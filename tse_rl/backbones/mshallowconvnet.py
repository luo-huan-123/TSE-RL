"""mshallownet F/D decomposition, history encoder and supervised pretraining."""

import copy
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from gymnasium.spaces import Dict
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1, **kwargs):
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)
        self.max_norm = max_norm

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(Conv2dWithConstraint, self).forward(x)


class LazyLinearWithConstraint(nn.LazyLinear):
    def __init__(self, *args, max_norm=1.0, **kwargs):
        super(LazyLinearWithConstraint, self).__init__(*args, **kwargs)
        self.max_norm = max_norm

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return self(x)


class MShallowFeatureExtractor(nn.Module):
    """
    m-ShallowConvNet-style feature extractor.
    Input: (batch, 1, num_electrodes, chunk_size)
    Output: (batch, feature_dim)  -- flattened per-window feature vector
    """

    def __init__(
        self,
        num_electrodes=22,
        chunk_size=250,
        depth=24,
        dropout_rate=0.5,
        max_norm=2.0,
        sampling_rate=250,
    ):
        super().__init__()
        self.num_electrodes = num_electrodes
        self.chunk_size = chunk_size
        self.depth = depth
        self.dropout_rate = dropout_rate
        kernel_size = max(1, int(sampling_rate * 0.12))
        pooling_size = int(max(1, sampling_rate * 0.3))
        hop = 0.7
        pooling_stride = max(1, int(pooling_size * (1 - hop)))
        kernel_size // 2
        self.temporal_conv = Conv2dWithConstraint(
            1, depth, kernel_size=(1, kernel_size), padding="same", max_norm=max_norm
        )
        self.spatial_conv = Conv2dWithConstraint(
            depth,
            depth,
            kernel_size=(num_electrodes, 1),
            padding="valid",
            max_norm=max_norm,
        )
        self.bn = nn.BatchNorm2d(depth)
        self.avg_pool = nn.AvgPool2d(
            kernel_size=(1, pooling_size), stride=(1, pooling_stride)
        )
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=dropout_rate)

    def _forward_features(self, x):
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        x = torch.square(x)
        x = self.avg_pool(x)
        x = torch.log(torch.clamp(x, min=1e-06))
        x = self.bn(x)
        return x

    def forward(self, x):
        x = self._forward_features(x)
        return x


class MShallowDiscriminator(nn.Module):
    """
    Simple discriminator/classifier that accepts the flattened features from MShallowFeatureExtractor.
    Input: features (batch, feature_dim)
    Output: softmax probabilities (batch, num_classes)
    """

    def __init__(self, feature_extractor_config: dict, num_classes=4, hidden_dim=128):
        super().__init__()
        MShallowFeatureExtractor(
            num_electrodes=feature_extractor_config.get("num_electrodes", 22),
            chunk_size=feature_extractor_config.get("chunk_size", 250),
            depth=feature_extractor_config.get("depth", 24),
            dropout_rate=feature_extractor_config.get("dropout_rate", 0.5),
            sampling_rate=feature_extractor_config.get("sampling_rate", 250),
        )
        pooling_size = int(
            max(1, feature_extractor_config.get("sampling_rate", 250) * 0.3)
        )
        hop = 0.7
        pooling_stride = max(1, int(pooling_size * (1 - hop)))
        self.bn = nn.BatchNorm2d(24)
        self.avg_pool = nn.AvgPool2d(
            kernel_size=(1, pooling_size), stride=(1, pooling_stride)
        )
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=0.5)
        self.classifier = LazyLinearWithConstraint(num_classes, max_norm=0.5)
        self._softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.flatten(x)
        x = self.dropout(x)
        logits = self.classifier(x)
        probs = self._softmax(logits)
        return probs


class HistoryEncoder(nn.Module):
    def __init__(self, chn, time_sample_num, history_len, feature_dim=64):
        super().__init__()
        self.history_len = history_len
        self.feature_extractor = MShallowFeatureExtractor(
            num_electrodes=chn, chunk_size=time_sample_num
        )
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, chn, time_sample_num)
            dummy_output = self.feature_extractor(dummy_input)
            self.per_window_feature_size = dummy_output.numel()
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
        batch_size, N, _, C, T = history_obs.shape
        x = history_obs.view(batch_size * N, 1, C, T)
        x = self.feature_extractor(x)
        x = x.view(batch_size, N, -1)
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
        )
        self.history_obs_extractor.feature_extractor.load_state_dict(
            pretrained_feature_extractor.state_dict(), strict=False
        )

    def forward(self, observations: dict) -> torch.Tensor:
        current_features = self.current_obs_extractor(observations["current_obs"])
        current_features = torch.flatten(current_features, start_dim=1)
        history_features = self.history_obs_extractor(observations["history_obs"])
        progress_features = observations["trial_progress"]
        combined_features = torch.cat(
            [current_features, history_features, progress_features], dim=1
        )
        return combined_features


def pretrain_discriminator(
    feature_extractor, discriminator, config, epochs=300, lr=0.001, batch_size=32
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
                features = feature_extractor(window)
                outputs = discriminator(features)
                window_outputs.append(outputs.unsqueeze(2))
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
                    features = feature_extractor(window)
                    outputs = discriminator(features)
                    window_outputs.append(outputs.unsqueeze(2))
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
