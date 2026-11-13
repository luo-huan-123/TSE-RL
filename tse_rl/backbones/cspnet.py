"""CSPNet F/D decomposition, history encoder and supervised pretraining."""

import copy
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from gymnasium.spaces import Dict
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


def square(x):
    return x * x


def safe_log(x, eps=1e-06):
    return torch.log(torch.clamp(x, min=eps))


class Expression(nn.Module):
    def __init__(self, expression_fn):
        super().__init__()
        self.expression_fn = expression_fn

    def forward(self, *x):
        return self.expression_fn(*x)


class Conv2dNormWeight(nn.Conv2d):
    def __init__(self, *args, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super().__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super().forward(x)


class CSPNetFeatureExtractor(nn.Module):
    def __init__(
        self,
        chunk_size: int,
        num_electrodes: int,
        num_filters_t: int = 20,
        filter_size_t: int = 25,
        num_filters_s: int = 2,
        filter_size_s: int = -1,
        pool_size_1: int = 100,
        pool_stride_1: int = 25,
        dropout: float = 0.5,
        min_policy_dim: int = 150,
    ):
        super().__init__()
        assert filter_size_t <= chunk_size, "Temporal filter size error"
        if filter_size_s <= 0:
            filter_size_s = num_electrodes
        self.chunk_size = chunk_size
        self.num_electrodes = num_electrodes
        self.num_filters_t = num_filters_t
        self.num_filters_s = num_filters_s
        self.filter_size_s = filter_size_s
        self.pool_size_1 = pool_size_1
        self.pool_stride_1 = pool_stride_1
        self.dropout_p = dropout
        self.F_out = num_filters_t * num_filters_s
        self.N_feat = (chunk_size - pool_size_1) // pool_stride_1 + 1
        self.pooled_flat_dim = self.F_out * self.N_feat
        self.use_pre_pool = self.pooled_flat_dim < min_policy_dim
        self.conv_t = nn.Conv2d(
            1,
            num_filters_t,
            (filter_size_t, 1),
            padding=(filter_size_t // 2, 0),
            bias=False,
        )
        self.bn_t = nn.BatchNorm2d(num_filters_t)
        self.conv_spatial = nn.Conv2d(
            num_filters_t,
            self.F_out,
            (1, filter_size_s),
            groups=num_filters_t,
            bias=False,
        )
        self.bn_spatial = nn.BatchNorm2d(self.F_out)
        self.act_square = Expression(square)
        self.pool = nn.AvgPool2d((pool_size_1, 1), stride=(pool_stride_1, 1))
        self.log = Expression(safe_log)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = x.permute(0, 1, 3, 2)
        x = self.conv_t(x)
        x = self.bn_t(x)
        x = self.conv_spatial(x)
        x = self.bn_spatial(x)
        x = self.act_square(x)
        if self.use_pre_pool:
            return x
        else:
            x = self.pool(x)
            x = self.log(x)
            return x


class CSPNetDiscriminator(nn.Module):
    def __init__(self, feature_extractor_config: dict, num_classes: int):
        super().__init__()
        self.cfg = feature_extractor_config
        F_out = self.cfg.get("num_filters_t", 20) * self.cfg.get("num_filters_s", 2)
        chunk_size = self.cfg["chunk_size"]
        pool_size_1 = self.cfg.get("pool_size_1", 100)
        pool_stride_1 = self.cfg.get("pool_stride_1", 25)
        N_feat = (chunk_size - pool_size_1) // pool_stride_1 + 1
        min_policy_dim = self.cfg.get("min_policy_dim", 150)
        pooled_flat_dim = F_out * N_feat
        self.use_pre_pool = pooled_flat_dim < min_policy_dim
        self.pool = nn.AvgPool2d((pool_size_1, 1), stride=(pool_stride_1, 1))
        self.log = Expression(safe_log)
        self.drop = nn.Dropout(self.cfg.get("dropout", 0.5))
        self.classifier = nn.Sequential(
            Conv2dNormWeight(F_out, num_classes, (N_feat, 1), max_norm=0.5),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, feats):
        x = feats
        x = self.drop(x)
        if self.use_pre_pool:
            x = self.pool(x)
            x = self.log(x)
            x = self.drop(x)
        log_probs = self.classifier(x)
        log_probs = log_probs[:, :, 0, 0]
        return torch.exp(log_probs)


class HistoryEncoder(nn.Module):
    def __init__(
        self, chn, time_sample_num, history_len, feature_dim=64, fe_kwargs: dict = None
    ):
        super().__init__()
        self.history_len = history_len
        self.feature_extractor = CSPNetFeatureExtractor(**fe_kwargs)
        with torch.no_grad():
            dummy = torch.zeros(1, 1, chn, time_sample_num)
            feats = self.feature_extractor(dummy)
            self.per_window_feature_size = feats.flatten(1).shape[1]
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.per_window_feature_size,
            nhead=4,
            dim_feedforward=128,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.fc_out = nn.Linear(self.per_window_feature_size, feature_dim)

    def forward(self, history_obs):
        B, N, _, C, T = history_obs.shape
        x = history_obs.view(B * N, 1, C, T)
        feats = self.feature_extractor(x)
        vec = feats.flatten(1)
        x = vec.view(B, N, -1)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        x = self.fc_out(x)
        return x


class CombinedFeatureExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: Dict,
        cnn_feature_dim: int,
        history_feature_dim: int,
        pretrained_feature_extractor: nn.Module,
        fe_kwargs: dict,
    ):
        super().__init__(
            observation_space, features_dim=cnn_feature_dim + history_feature_dim + 1
        )
        self.current_obs_extractor = pretrained_feature_extractor
        self.history_obs_extractor = HistoryEncoder(
            chn=observation_space["current_obs"].shape[1],
            time_sample_num=observation_space["current_obs"].shape[2],
            history_len=observation_space["history_obs"].shape[0],
            feature_dim=history_feature_dim,
            fe_kwargs=fe_kwargs,
        )
        self.history_obs_extractor.feature_extractor.load_state_dict(
            pretrained_feature_extractor.state_dict(), strict=False
        )

    def forward(self, observations: dict) -> torch.Tensor:
        feats = self.current_obs_extractor(observations["current_obs"])
        current_features = feats.flatten(1)
        history_features = self.history_obs_extractor(observations["history_obs"])
        progress_features = observations["trial_progress"]
        combined_features = torch.cat(
            [current_features, history_features, progress_features], dim=1
        )
        return combined_features


def pretrain_discriminator(
    feature_extractor, discriminator, config, epochs=300, lr=0.001, batch_size=64
):
    print("\n--- [Stage 1] Pre-training Discriminator (Sliding Window Averaging) ---")
    device = next(feature_extractor.parameters()).device
    data_name = config.get("data_name")
    if data_name is None:
        raise ValueError("Config must include 'data_name'.")
    train_data, train_labels = (config["train_data"], config["train_labels"])
    test_data, test_labels = (config["selection_data"], config["selection_labels"])
    train_dataset = TensorDataset(
        torch.from_numpy(train_data.astype(np.float32)),
        torch.from_numpy(train_labels).long(),
    )
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    test_dataset = TensorDataset(
        torch.from_numpy(test_data.astype(np.float32)),
        torch.from_numpy(test_labels).long(),
    )
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
    params = list(feature_extractor.parameters()) + list(discriminator.parameters())
    optimizer = optim.Adam(
        params, lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0001
    )
    criterion = nn.CrossEntropyLoss()
    best_test_acc = -1.0
    best_feature_extractor_wts = None
    best_discriminator_wts = None
    sampling_rate = config.get("sampling_rate", 250)
    window_size = int(config.get("window_sec", 1.0) * sampling_rate)
    step_size = int(config.get("step_sec", 0.1) * sampling_rate)
    for epoch in range(epochs):
        feature_extractor.train()
        discriminator.train()
        running_loss, train_correct, train_total = (0.0, 0, 0)
        for full_trials, labels in train_loader:
            full_trials, labels = (full_trials.to(device), labels.to(device))
            window_outputs = []
            for start in range(0, full_trials.shape[-1] - window_size + 1, step_size):
                window = full_trials[:, :, start : start + window_size].unsqueeze(1)
                feats = feature_extractor(window)
                probs = discriminator(feats)
                window_outputs.append(probs.unsqueeze(2))
            window_outputs = torch.cat(window_outputs, dim=2)
            averaged_predictions = torch.mean(window_outputs, dim=2)
            optimizer.zero_grad()
            loss = criterion(averaged_predictions, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(averaged_predictions, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        train_acc = 100 * train_correct / train_total
        feature_extractor.eval()
        discriminator.eval()
        test_correct, test_total = (0, 0)
        with torch.no_grad():
            for full_trials, labels in test_loader:
                full_trials, labels = (full_trials.to(device), labels.to(device))
                window_outputs = []
                for start in range(
                    0, full_trials.shape[-1] - window_size + 1, step_size
                ):
                    window = full_trials[:, :, start : start + window_size].unsqueeze(1)
                    feats = feature_extractor(window)
                    probs = discriminator(feats)
                    window_outputs.append(probs.unsqueeze(2))
                window_outputs = torch.cat(window_outputs, dim=2)
                averaged_predictions = torch.mean(window_outputs, dim=2)
                _, predicted = torch.max(averaged_predictions, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
        test_acc = 100 * test_correct / test_total
        print(
            f"Epoch {epoch + 1}/{epochs} | Loss: {running_loss / len(train_loader):.4f} | Train Acc: {train_acc:.2f}% | Selection Acc: {test_acc:.2f}%"
        )
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_feature_extractor_wts = copy.deepcopy(feature_extractor.state_dict())
            best_discriminator_wts = copy.deepcopy(discriminator.state_dict())
            print(f"  -> New best model found with accuracy: {best_test_acc:.2f}%")
    print(
        f"--- [Stage 1] Pre-training complete. Best selection accuracy: {best_test_acc:.2f}% ---"
    )
    return (best_test_acc, best_feature_extractor_wts, best_discriminator_wts)
