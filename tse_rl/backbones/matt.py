# Adapted F/D implementation. Retained upstream notice: third_party/MAtt-LICENSE.
"""Matt F/D decomposition, history encoder and supervised pretraining."""

import copy
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from gymnasium.spaces import Dict
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


def regularized_cov(X: torch.Tensor, eps: float = 0.0001) -> torch.Tensor:
    """
    X: (B, C, T)  ->  S: (B, C, C)  SPD
    """
    Xc = X - X.mean(dim=2, keepdim=True)
    B, C, T = Xc.shape
    S = Xc @ Xc.transpose(1, 2) / (T - 1)
    I = torch.eye(C, device=X.device, dtype=X.dtype).unsqueeze(0).expand(B, C, C)
    S = S + eps * I
    return S


def matrix_logm_spd(S: torch.Tensor, eps: float = 1e-06) -> torch.Tensor:
    orig_dtype = S.dtype
    Sd = S.double()
    Sd = 0.5 * (Sd + Sd.transpose(1, 2))
    L, U = torch.linalg.eigh(Sd)
    L = torch.clamp(L, min=eps)
    logL = torch.log(L)
    Yd = U @ torch.diag_embed(logL) @ U.transpose(1, 2)
    return Yd.to(orig_dtype)


def stiefel_retraction(W: torch.Tensor) -> torch.Tensor:
    Q, R = torch.linalg.qr(W, mode="reduced")
    s = torch.sign(torch.diagonal(R, dim1=-2, dim2=-1))
    s = torch.where(s == 0, torch.ones_like(s), s)
    Q = Q * s
    return Q


class MAttFeatureExtractor(nn.Module):
    def __init__(
        self,
        chunk_size: int,
        num_electrodes: int,
        num_heads: int = 4,
        proj_dim: int = 16,
    ):
        super().__init__()
        self.T = chunk_size
        self.C = num_electrodes
        self.H = num_heads
        self.d = proj_dim
        self.raw_projs = nn.ParameterList(
            [nn.Parameter(torch.randn(self.C, self.d) * 0.02) for _ in range(self.H)]
        )
        for p in self.raw_projs:
            nn.init.orthogonal_(p)

    def forward(self, x):
        if x.dim() == 4:
            x = x.squeeze(1)
        B, C, T = x.shape
        assert C == self.C and T == self.T, (
            f"Expect (B,{self.C},{self.T}), got {x.shape}"
        )
        S = regularized_cov(x, eps=0.0001)
        heads = []
        for h in range(self.H):
            Wh = stiefel_retraction(self.raw_projs[h])
            SW = torch.matmul(S, Wh)
            Sh = torch.matmul(Wh.t().unsqueeze(0), SW)
            Lh = matrix_logm_spd(Sh, eps=1e-06)
            Lh = Lh - Lh.mean(dim=(1, 2), keepdim=True)
            fnorm = Lh.norm(dim=(1, 2), keepdim=True).clamp_min(1e-06)
            Lh = Lh / fnorm
            Lh = torch.nan_to_num(Lh, nan=0.0, posinf=1000.0, neginf=-1000.0)
            heads.append(Lh)
        feats = torch.stack(heads, dim=1)
        return feats


class MAttDiscriminator(nn.Module):
    def __init__(
        self,
        feature_extractor_config: dict,
        num_classes: int,
        att_hidden: int = 128,
        mlp_hidden: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.cfg = feature_extractor_config
        self.H = self.cfg.get("num_heads", 4)
        self.d = self.cfg.get("proj_dim", 16)
        self.F = self.d * self.d
        self.K = num_classes
        self.dropout = nn.Dropout(dropout)
        self.att_mlp = nn.Sequential(
            nn.Linear(self.F, att_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(att_hidden, 1),
        )
        self.cls_mlp = nn.Sequential(
            nn.Linear(self.F, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, num_classes),
        )

    def forward(self, feats):
        """
        feats: (B,H,d,d) from FE
        """
        B, H, d, _ = feats.shape
        V = feats.reshape(B, H, -1)
        scores = self.att_mlp(V).squeeze(-1)
        alpha = torch.softmax(scores, dim=1).unsqueeze(-1)
        V_agg = (V * alpha).sum(dim=1)
        V_agg = self.dropout(V_agg)
        logits = self.cls_mlp(V_agg)
        probs = torch.softmax(logits, dim=1)
        return probs


class HistoryEncoder(nn.Module):
    def __init__(
        self, chn, time_sample_num, history_len, feature_dim=64, fe_kwargs: dict = None
    ):
        super().__init__()
        self.history_len = history_len
        self.feature_extractor = MAttFeatureExtractor(**fe_kwargs)
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
