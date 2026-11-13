# Adapted F/D implementation. Retained upstream notice: third_party/FBCNet-LICENSE.
"""FBCNet F/D decomposition, history encoder and supervised pretraining."""

import copy
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from gymnasium.spaces import Dict
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, weight_norm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.weight_norm = weight_norm
        super().__init__(*args, **kwargs)

    def forward(self, x):
        if self.weight_norm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super().forward(x)


class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, weight_norm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.weight_norm = weight_norm
        super().__init__(*args, **kwargs)

    def forward(self, x):
        if self.weight_norm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super().forward(x)


class VarLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.var(dim=self.dim, keepdim=True)


class StdLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.std(dim=self.dim, keepdim=True)


class LogVarLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.log(
            torch.clamp(x.var(dim=self.dim, keepdim=True), 1e-06, 1000000.0)
        )


class MeanLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.mean(dim=self.dim, keepdim=True)


class MaxLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        ma, _ = x.max(dim=self.dim, keepdim=True)
        return ma


class swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class FBCNetFeatureExtractor(nn.Module):
    def __init__(
        self,
        num_electrodes: int,
        chunk_size: int,
        in_channels: int = 1,
        num_S: int = 32,
        stride_factor: int = 25,
        temporal: str = "LogVarLayer",
        weight_norm: bool = True,
    ):
        super().__init__()
        self.num_electrodes = num_electrodes
        self.in_channels = in_channels
        self.num_S = num_S
        self.stride_factor = stride_factor
        self.scb = nn.Sequential(
            Conv2dWithConstraint(
                in_channels,
                num_S * in_channels,
                (num_electrodes, 1),
                groups=in_channels,
                max_norm=2,
                weight_norm=weight_norm,
                padding=0,
            ),
            nn.BatchNorm2d(num_S * in_channels),
            swish(),
        )
        if temporal == "VarLayer":
            self.temporal_layer = VarLayer(dim=3)
        elif temporal == "StdLayer":
            self.temporal_layer = StdLayer(dim=3)
        elif temporal == "LogVarLayer":
            self.temporal_layer = LogVarLayer(dim=3)
        elif temporal == "MeanLayer":
            self.temporal_layer = MeanLayer(dim=3)
        elif temporal == "MaxLayer":
            self.temporal_layer = MaxLayer(dim=3)
        else:
            raise NotImplementedError(f"Unknown temporal layer: {temporal}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.scb(x)
        B, F, _, T = x.shape
        assert T % self.stride_factor == 0, (
            f"T({T}) must be divisible by stride_factor({self.stride_factor})"
        )
        x = x.reshape(B, F, self.stride_factor, T // self.stride_factor)
        x = self.temporal_layer(x)
        x = torch.flatten(x, start_dim=1)
        return x


class FBCNetDiscriminator(nn.Module):
    def __init__(
        self, feature_extractor_config: dict, num_classes: int, weight_norm: bool = True
    ):
        super().__init__()
        in_channels = feature_extractor_config.get("in_channels", 1)
        num_S = feature_extractor_config.get("num_S", 32)
        stride_factor = feature_extractor_config.get("stride_factor", 25)
        in_dim = num_S * in_channels * stride_factor
        self.last_layer = nn.Sequential(
            LinearWithConstraint(
                in_dim, num_classes, max_norm=0.5, weight_norm=weight_norm
            ),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, feat_vec: torch.Tensor) -> torch.Tensor:
        log_probs = self.last_layer(feat_vec)
        return torch.exp(log_probs)


class HistoryEncoder(nn.Module):
    def __init__(
        self,
        chn,
        time_sample_num,
        history_len,
        feature_dim=64,
        in_channels=1,
        num_S=32,
        temporal="LogVarLayer",
        stride_factor=25,
    ):
        super().__init__()
        self.history_len = history_len
        fe_cfg = dict(
            num_electrodes=chn,
            chunk_size=time_sample_num,
            in_channels=in_channels,
            num_S=num_S,
            stride_factor=stride_factor,
            temporal=temporal,
            weight_norm=True,
        )
        self.feature_extractor = FBCNetFeatureExtractor(**fe_cfg)
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, chn, time_sample_num)
            vec = self.feature_extractor(dummy)
            self.per_window_feature_size = vec.shape[1]
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
        vec = self.feature_extractor(x)
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
        in_channels=1,
        num_S=32,
        temporal="LogVarLayer",
        stride_factor=25,
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
            in_channels=in_channels,
            num_S=num_S,
            temporal=temporal,
            stride_factor=stride_factor,
        )
        self.history_obs_extractor.feature_extractor.load_state_dict(
            pretrained_feature_extractor.state_dict(), strict=False
        )

    def forward(self, observations: dict) -> torch.Tensor:
        current_features = self.current_obs_extractor(observations["current_obs"])
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
                feat_vec = feature_extractor(window)
                probs = discriminator(feat_vec)
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
                    feat_vec = feature_extractor(window)
                    probs = discriminator(feat_vec)
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
