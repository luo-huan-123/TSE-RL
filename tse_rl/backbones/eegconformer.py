# Adapted F/D implementation. Retained upstream notice: third_party/TorchEEG-LICENSE.
"""EEGConformer F/D decomposition, history encoder and supervised pretraining."""

import copy
import math
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from einops import rearrange
from einops.layers.torch import Rearrange
from gymnasium.spaces import Dict
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class PatchEmbedding(nn.Module):
    def __init__(
        self, num_electrodes: int, hid_channels: int = 40, dropout: float = 0.5
    ):
        super().__init__()
        self.shallownet = nn.Sequential(
            nn.Conv2d(1, hid_channels, (1, 25), (1, 1)),
            nn.Conv2d(hid_channels, hid_channels, (num_electrodes, 1), (1, 1)),
            nn.BatchNorm2d(hid_channels),
            nn.ELU(),
            nn.AvgPool2d((1, 75), (1, 15)),
            nn.Dropout(dropout),
        )
        self.projection = nn.Sequential(
            nn.Conv2d(hid_channels, hid_channels, (1, 1), stride=(1, 1)),
            Rearrange("b e (h) (w) -> b (h w) e"),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.shallownet(x)
        x = self.projection(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, hid_channels: int, heads: int, dropout: float):
        super().__init__()
        self.hid_channels = hid_channels
        self.heads = heads
        self.keys = nn.Linear(hid_channels, hid_channels)
        self.queries = nn.Linear(hid_channels, hid_channels)
        self.values = nn.Linear(hid_channels, hid_channels)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(hid_channels, hid_channels)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.heads)
        energy = torch.einsum("bhqd, bhkd -> bhqk", queries, keys)
        if mask is not None:
            fill_value = torch.finfo(energy.dtype).min
            energy = energy.masked_fill(~mask, fill_value)
        d_k = queries.shape[-1]
        att = F.softmax(energy / math.sqrt(d_k), dim=-1)
        att = self.att_drop(att)
        out = torch.einsum("bhqk, bhkd -> bhqd", att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.fn(x, **kwargs) + x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, hid_channels: int, expansion: int = 4, dropout: float = 0.0):
        super().__init__(
            nn.Linear(hid_channels, expansion * hid_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expansion * hid_channels, hid_channels),
        )


class TransformerEncoderBlock(nn.Sequential):
    def __init__(
        self,
        hid_channels: int,
        heads: int,
        dropout: float,
        forward_expansion: int,
        forward_dropout: float,
    ):
        super().__init__(
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(hid_channels),
                    MultiHeadAttention(hid_channels, heads, dropout),
                    nn.Dropout(dropout),
                )
            ),
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(hid_channels),
                    FeedForwardBlock(
                        hid_channels,
                        expansion=forward_expansion,
                        dropout=forward_dropout,
                    ),
                    nn.Dropout(dropout),
                )
            ),
        )


class TransformerEncoder(nn.Sequential):
    def __init__(
        self,
        depth: int,
        hid_channels: int,
        heads: int = 10,
        dropout: float = 0.5,
        forward_expansion: int = 4,
        forward_dropout: float = 0.5,
    ):
        super().__init__(
            *[
                TransformerEncoderBlock(
                    hid_channels, heads, dropout, forward_expansion, forward_dropout
                )
                for _ in range(depth)
            ]
        )


class ClassificationHead(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        hid_channels: int = 32,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channels, hid_channels * 8),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hid_channels * 8, hid_channels),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hid_channels, num_classes),
        )

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        x = self.fc(x)
        return x


class EEGConformerFeatureExtractor(nn.Module):
    def __init__(
        self, num_electrodes: int, chunk_size: int, hid_channels: int = 40, **kwargs
    ):
        super().__init__()
        self.num_electrodes = num_electrodes
        self.hid_channels = hid_channels
        self.stem = nn.Sequential(
            nn.Conv2d(1, hid_channels, (1, 25), (1, 1)),
            nn.Conv2d(hid_channels, hid_channels, (num_electrodes, 1), (1, 1)),
            nn.BatchNorm2d(hid_channels),
            nn.ELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stem(x)


class EEGConformerTail(nn.Module):
    def __init__(
        self,
        hid_channels: int = 40,
        depth: int = 6,
        heads: int = 10,
        dropout: float = 0.5,
        forward_expansion: int = 4,
        forward_dropout: float = 0.5,
    ):
        super().__init__()
        self.post_pool = nn.Sequential(
            nn.AvgPool2d((1, 75), (1, 15)),
            nn.Dropout(dropout),
            nn.Conv2d(hid_channels, hid_channels, (1, 1), stride=(1, 1)),
            Rearrange("b e (h) (w) -> b (h w) e"),
        )
        self.encoder = TransformerEncoder(
            depth=depth,
            hid_channels=hid_channels,
            heads=heads,
            dropout=dropout,
            forward_expansion=forward_expansion,
            forward_dropout=forward_dropout,
        )

    def forward(self, pre_pool_feats: torch.Tensor) -> torch.Tensor:
        x = self.post_pool(pre_pool_feats)
        x = self.encoder(x)
        return x


class EEGConformerDiscriminator(nn.Module):
    def __init__(
        self,
        feature_extractor_config: dict,
        num_classes: int = 4,
        cls_channels: int = 32,
        cls_dropout: float = 0.5,
        depth: int = 6,
        heads: int = 10,
        dropout: float = 0.5,
        forward_expansion: int = 4,
        forward_dropout: float = 0.5,
    ):
        super().__init__()
        hid_channels = feature_extractor_config.get("hid_channels", 40)
        self.tail = EEGConformerTail(
            hid_channels=hid_channels,
            depth=depth,
            heads=heads,
            dropout=dropout,
            forward_expansion=forward_expansion,
            forward_dropout=forward_dropout,
        )
        temp_fe = EEGConformerFeatureExtractor(**feature_extractor_config)
        with torch.no_grad():
            dummy = torch.zeros(
                1,
                1,
                feature_extractor_config["num_electrodes"],
                feature_extractor_config["chunk_size"],
            )
            pre_pool = temp_fe(dummy)
            toks = self.tail(pre_pool)
            flat_dim = toks.flatten(start_dim=1).shape[1]
        self.cls = ClassificationHead(
            in_channels=flat_dim,
            num_classes=num_classes,
            hid_channels=cls_channels,
            dropout=cls_dropout,
        )

    def forward(self, pre_pool_feats: torch.Tensor) -> torch.Tensor:
        tokens = self.tail(pre_pool_feats)
        logits = self.cls(tokens)
        return torch.softmax(logits, dim=1)


class HistoryEncoder(nn.Module):
    def __init__(
        self,
        chn,
        time_sample_num,
        history_len,
        feature_dim=64,
        hid_channels: int = 40,
        depth: int = 6,
        heads: int = 10,
        dropout: float = 0.5,
        forward_expansion: int = 4,
        forward_dropout: float = 0.5,
    ):
        super().__init__()
        self.history_len = history_len
        fe_cfg = dict(
            num_electrodes=chn, chunk_size=time_sample_num, hid_channels=hid_channels
        )
        self.feature_extractor = EEGConformerFeatureExtractor(**fe_cfg)
        self.tail = EEGConformerTail(
            hid_channels=hid_channels,
            depth=depth,
            heads=heads,
            dropout=dropout,
            forward_expansion=forward_expansion,
            forward_dropout=forward_dropout,
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, chn, time_sample_num)
            pre_pool = self.feature_extractor(dummy)
            tokens = self.tail(pre_pool)
            self.per_window_feature_size = tokens.flatten(1).shape[1]
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
        pre_pool = self.feature_extractor(x)
        toks = self.tail(pre_pool)
        x = toks.view(batch_size, N, -1)
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
        pretrained_tail: nn.Module = None,
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
        if pretrained_tail is not None:
            self.history_obs_extractor.tail.load_state_dict(
                pretrained_tail.state_dict(), strict=False
            )

    def forward(self, observations: dict) -> torch.Tensor:
        pre_pool = self.current_obs_extractor(observations["current_obs"])
        current_features = torch.flatten(pre_pool, start_dim=1)
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
