# Adapted F/D implementation. Retained upstream notice: third_party/TorchEEG-LICENSE.
"""ATCNet F/D decomposition, history encoder and supervised pretraining."""

import copy
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from gymnasium.spaces import Dict
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class ATCNetFeatureExtractor(nn.Module):
    """
    This class serves as the feature extractor, containing the 'conv_block' from ATCNet.
    It takes a raw EEG window and outputs a feature map.
    """

    def __init__(
        self,
        num_electrodes=22,
        chunk_size=1000,
        F1=16,
        D=2,
        conv_pool_size=7,
        dropout=0.5,
    ):
        super().__init__()
        F2 = F1 * D
        self.conv_block = nn.Sequential(
            nn.Conv2d(
                1,
                F1,
                (1, int(chunk_size / 2 + 1)),
                stride=1,
                padding="same",
                bias=False,
            ),
            nn.BatchNorm2d(F1, False),
            nn.Conv2d(F1, F2, (num_electrodes, 1), padding=0, groups=F1),
            nn.BatchNorm2d(F2, False),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout2d(dropout),
            nn.Conv2d(F2, F2, (1, 16), bias=False, padding="same"),
            nn.BatchNorm2d(F2, False),
            nn.ELU(),
            nn.AvgPool2d((1, conv_pool_size)),
            nn.Dropout2d(dropout),
        )

    def forward(self, x):
        return self.conv_block(x)


class ATCNetDiscriminator(nn.Module):
    """
    This class contains the temporal processing part of ATCNet (MSA, TCN) and the final classifier.
    It takes the feature map from ATCNetFeatureExtractor as input.
    """

    def __init__(
        self,
        feature_extractor_config: dict,
        num_classes=4,
        num_windows=3,
        tcn_depth=2,
        tcn_kernel_size=4,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_windows = num_windows
        self.tcn_depth = tcn_depth
        self.tcn_kernel_size = tcn_kernel_size
        self._build_model(feature_extractor_config)

    def _build_model(self, fe_config):
        temp_fe = ATCNetFeatureExtractor(**fe_config)
        with torch.no_grad():
            dummy_input = torch.zeros(
                1, 1, fe_config["num_electrodes"], fe_config["chunk_size"]
            )
            x = temp_fe(dummy_input)
            x = x[:, :, -1, :]
            x = x.permute(0, 2, 1)
            self.__chan_dim, self.__embed_dim = x.shape[1:]
            for i in range(self.num_windows):
                self.add_module(
                    f"msa{i}",
                    nn.MultiheadAttention(
                        embed_dim=self.__embed_dim, num_heads=2, batch_first=True
                    ),
                )
                self.add_module(f"msa_drop{i}", nn.Dropout(0.3))
                tcn_input_channels = self.__chan_dim - self.num_windows + 1
                for j in range(self.tcn_depth):
                    tcn_block = nn.Sequential(
                        nn.Conv1d(
                            tcn_input_channels, 32, self.tcn_kernel_size, padding="same"
                        ),
                        nn.BatchNorm1d(32),
                        nn.ELU(),
                        nn.Dropout(0.3),
                        nn.Conv1d(
                            32, tcn_input_channels, self.tcn_kernel_size, padding="same"
                        ),
                        nn.BatchNorm1d(tcn_input_channels),
                        nn.ELU(),
                        nn.Dropout(0.3),
                    )
                    self.add_module(f"tcn_{i}_{j}", tcn_block)
                self.add_module(
                    f"dense{i}", nn.Linear(self.__embed_dim, self.num_classes)
                )

    def forward(self, x):
        x = x[:, :, -1, :]
        x = x.permute(0, 2, 1)
        sw_concat = 0
        for i in range(self.num_windows):
            st = i
            end = x.shape[1] - self.num_windows + i + 1
            x2 = x[:, st:end, :]
            msa_module = self.get_submodule(f"msa{i}")
            msa_drop_module = self.get_submodule(f"msa_drop{i}")
            x2_att, _ = msa_module(x2, x2, x2)
            x2 = x2 + msa_drop_module(x2_att)
            for j in range(self.tcn_depth):
                tcn_module = self.get_submodule(f"tcn_{i}_{j}")
                x2_tcn = tcn_module(x2)
                x2 = x2 + x2_tcn
            dense_module = self.get_submodule(f"dense{i}")
            classified_output = dense_module(x2[:, -1, :])
            sw_concat += classified_output
        avg_output = sw_concat / self.num_windows
        return torch.softmax(avg_output, dim=1)


class HistoryEncoder(nn.Module):
    def __init__(self, chn, time_sample_num, history_len, feature_dim=64):
        super().__init__()
        self.history_len = history_len
        self.feature_extractor = ATCNetFeatureExtractor(
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
