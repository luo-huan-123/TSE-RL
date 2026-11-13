"""EEGNet F/D decomposition, history encoder and supervised pretraining."""

import copy
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from gymnasium.spaces import Dict
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class EEGNetFeatureExtractor(nn.Module):
    def __init__(self, classes_num, input_ch, input_time, dropout_size):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.ZeroPad2d(padding=(31, 32, 0, 0)),
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(1, 64), bias=False),
            nn.BatchNorm2d(8),
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=8,
                out_channels=16,
                kernel_size=(input_ch, 1),
                groups=8,
                bias=False,
            ),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
        )

    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, classes_num, input_ch, input_time, dropout_size):
        super().__init__()
        self.dropout = nn.Dropout(dropout_size)
        self.block_3 = nn.Sequential(
            nn.ZeroPad2d((7, 8, 0, 0)),
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=(1, 16),
                groups=16,
                bias=False,
            ),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 16)),
        )
        with torch.no_grad():
            dummy_input = torch.zeros(1, 16, 1, input_time // 4)
            out = self.block_3(dummy_input)
            n_outputs = out.flatten(1).shape[1]
        self.clf = nn.Linear(n_outputs, classes_num)

    def forward(self, x):
        x = self.dropout(x)
        x = self.block_3(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.clf(x)
        return torch.softmax(x, dim=-1)


class HistoryEncoder(nn.Module):
    def __init__(self, chn, time_sample_num, history_len, feature_dim=64):
        super().__init__()
        self.history_len = history_len
        self.feature_extractor = nn.Sequential(
            nn.ZeroPad2d((31, 32, 0, 0)),
            nn.Conv2d(1, 8, (1, 64), bias=False),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, (chn, 1), groups=8, bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
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
    feature_extractor, discriminator, config, epochs=300, lr=0.001, batch_size=288
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
