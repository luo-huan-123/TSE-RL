# Adapted F/D implementation. Retained upstream notice: third_party/TSLANet-LICENSE.
"""TSLANet F/D decomposition, history encoder and supervised pretraining."""

import copy
import math
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from gymnasium.spaces import Dict
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


def count_parameters(m):
    return sum((p.numel() for p in m.parameters() if p.requires_grad))


from torch.nn.init import trunc_normal_


def drop_path(
    x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True
):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


class DropToken(nn.Module):
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = float(p)

    def forward(self, x):
        if self.p == 0.0 or not self.training:
            return x
        B, N, E = x.shape
        keep = (torch.rand(B, N, 1, device=x.device) > self.p).float()
        keep = keep / (1.0 - self.p)
        return x * keep


class ICB(nn.Module):
    def __init__(self, in_features, hidden_features, drop=0.0):
        super().__init__()
        self.conv1 = nn.Conv1d(in_features, hidden_features, 1)
        self.conv2 = nn.Conv1d(in_features, hidden_features, 3, 1, 1)
        self.conv3 = nn.Conv1d(hidden_features, in_features, 1)
        self.drop = nn.Dropout(drop)
        self.act = nn.GELU()

    def forward(self, x):
        x = x.transpose(1, 2)
        x1 = self.conv1(x)
        x1_1 = self.act(x1)
        x1_2 = self.drop(x1_1)
        x2 = self.conv2(x)
        x2_1 = self.act(x2)
        x2_2 = self.drop(x2_1)
        out1 = x1 * x2_2
        out2 = x2 * x1_2
        x = self.conv3(out1 + out2)
        x = x.transpose(1, 2)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, seq_len, patch_size=40, in_chans=22, embed_dim=96):
        super().__init__()
        stride = patch_size // 2
        num_patches = int((seq_len - patch_size) / stride + 1)
        assert num_patches > 0, (
            f"num_patches must be >0, got 0 with seq_len={seq_len}, patch_size={patch_size}"
        )
        self.num_patches = num_patches
        self.proj = nn.Conv1d(
            in_chans, embed_dim, kernel_size=patch_size, stride=stride
        )

    def forward(self, x):
        x_out = self.proj(x).flatten(2).transpose(1, 2)
        return x_out


class AdaptiveSpectralBlock(nn.Module):
    def __init__(self, dim, adaptive_filter=True):
        super().__init__()
        self.adaptive_filter = adaptive_filter
        self.complex_weight_high = nn.Parameter(
            torch.randn(dim, 2, dtype=torch.float32) * 0.02
        )
        self.complex_weight = nn.Parameter(
            torch.randn(dim, 2, dtype=torch.float32) * 0.02
        )
        trunc_normal_(self.complex_weight_high, std=0.02)
        trunc_normal_(self.complex_weight, std=0.02)
        self.threshold_param = nn.Parameter(torch.rand(1))

    def create_adaptive_high_freq_mask(self, x_fft):
        B, _, _ = x_fft.shape
        energy = torch.abs(x_fft).pow(2).sum(dim=-1)
        flat_energy = energy.view(B, -1)
        median_energy = flat_energy.median(dim=1, keepdim=True)[0].view(B, 1)
        epsilon = 1e-06
        normalized_energy = energy / (median_energy + epsilon)
        adaptive_mask = (
            (normalized_energy > self.threshold_param).float() - self.threshold_param
        ).detach() + self.threshold_param
        adaptive_mask = adaptive_mask.unsqueeze(-1)
        return adaptive_mask

    def forward(self, x_in):
        B, N, C = x_in.shape
        dtype = x_in.dtype
        x = x_in.to(torch.float32)
        x_fft = torch.fft.rfft(x, dim=1, norm="ortho")
        weight = torch.view_as_complex(self.complex_weight)
        x_weighted = x_fft * weight
        if self.adaptive_filter:
            freq_mask = self.create_adaptive_high_freq_mask(x_fft)
            x_masked = x_fft * freq_mask.to(x.device)
            weight_high = torch.view_as_complex(self.complex_weight_high)
            x_weighted2 = x_masked * weight_high
            x_weighted += x_weighted2
        x = torch.fft.irfft(x_weighted, n=N, dim=1, norm="ortho")
        x = x.to(dtype)
        return x


class TSLANetLayer(nn.Module):
    def __init__(
        self, dim, mlp_ratio=2.0, drop=0.3, drop_path=0.0, norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.ASB = AdaptiveSpectralBlock(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.ICB = ICB(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.ICB(self.norm2(self.ASB(self.norm1(x)))))
        return x


class TSLANetFeatureExtractor(nn.Module):
    def __init__(
        self,
        chunk_size: int,
        num_electrodes: int,
        patch_size: int = 40,
        emb_dim: int = 96,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.patch_embed = PatchEmbed(
            seq_len=chunk_size,
            patch_size=patch_size,
            in_chans=num_electrodes,
            embed_dim=emb_dim,
        )
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, emb_dim), requires_grad=True
        )
        trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        if x.dim() == 4:
            x = x.squeeze(1)
        assert x.dim() == 3, f"Expected (B,C,T), got {x.shape}"
        x = self.patch_embed(x)
        x = x + self.pos_embed
        return x


class TSLANetDiscriminator(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        num_classes: int,
        depth: int = 3,
        dropout_rate: float = 0.3,
        mlp_ratio: float = 2.0,
        drop_token: float = 0.2,
        drop_path_rate: float = 0.2,
    ):
        super().__init__()
        self.drop_token = DropToken(drop_token)
        self.pos_drop = nn.Dropout(p=dropout_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.tsla_blocks = nn.ModuleList(
            [
                TSLANetLayer(
                    dim=emb_dim,
                    mlp_ratio=mlp_ratio,
                    drop=dropout_rate,
                    drop_path=dpr[i],
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(emb_dim)
        self.head = nn.Linear(emb_dim, num_classes)

    def forward(self, tokens_pre_drop, return_logits: bool = False):
        x = self.drop_token(tokens_pre_drop)
        x = self.pos_drop(x)
        for blk in self.tsla_blocks:
            x = blk(x)
        x = self.norm(x)
        x = x.mean(1)
        logits = self.head(x)
        if return_logits:
            return logits
        probs = torch.softmax(logits, dim=1)
        return probs


def augment_windows(batch_x, max_shift=12, noise_std=0.01):
    if max_shift > 0:
        shifts = np.random.randint(-max_shift, max_shift + 1, size=batch_x.shape[0])
        for i, s in enumerate(shifts):
            if s != 0:
                batch_x[i] = np.roll(batch_x[i], shift=s, axis=-1)
    if noise_std > 0:
        noise = np.random.randn(*batch_x.shape).astype(np.float32) * noise_std
        batch_x = batch_x + noise
    return batch_x


class HistoryEncoder(nn.Module):
    def __init__(
        self,
        chn,
        time_sample_num,
        history_len,
        feature_dim=64,
        patch_size=40,
        emb_dim=96,
    ):
        super().__init__()
        self.history_len = history_len
        self.feature_extractor = TSLANetFeatureExtractor(
            chunk_size=time_sample_num,
            num_electrodes=chn,
            patch_size=patch_size,
            emb_dim=emb_dim,
        )
        with torch.no_grad():
            dummy = torch.zeros(1, chn, time_sample_num)
            tokens = self.feature_extractor(dummy)
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
        B, N, _, C, T = history_obs.shape
        x = history_obs.view(B * N, 1, C, T)
        tokens = self.feature_extractor(x)
        vec = tokens.flatten(1)
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
        patch_size=40,
        emb_dim=96,
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
            patch_size=patch_size,
            emb_dim=emb_dim,
        )
        self.history_obs_extractor.feature_extractor.load_state_dict(
            pretrained_feature_extractor.state_dict(), strict=False
        )

    def forward(self, observations: dict) -> torch.Tensor:
        tokens = self.current_obs_extractor(observations["current_obs"])
        current_features = tokens.flatten(1)
        history_features = self.history_obs_extractor(observations["history_obs"])
        progress_features = observations["trial_progress"]
        combined_features = torch.cat(
            [current_features, history_features, progress_features], dim=1
        )
        return combined_features


class WarmupCosineLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self, optimizer, warmup_epochs, max_epochs, min_lr=1e-05, last_epoch=-1
    ):
        self.warmup = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup:
            return [
                base_lr * (self.last_epoch + 1) / max(1, self.warmup)
                for base_lr in self.base_lrs
            ]
        e = self.last_epoch - self.warmup
        E = max(1, self.max_epochs - self.warmup)
        return [
            self.min_lr + (base_lr - self.min_lr) * (1 + math.cos(math.pi * e / E)) / 2
            for base_lr in self.base_lrs
        ]


def pretrain_discriminator(
    feature_extractor,
    discriminator,
    config,
    epochs=300,
    base_lr=0.0003,
    batch_size=64,
    label_smoothing=0.1,
    weight_decay=0.05,
    max_grad_norm=1.0,
    augment=True,
):
    print(
        "\n--- [Stage 1] Pre-training Discriminator (Sliding Window Averaging, with strong regularization) ---"
    )
    device = next(feature_extractor.parameters()).device
    data_name = config.get("data_name")
    if data_name is None:
        raise ValueError("Config must include 'data_name'.")
    train_data, train_labels = (config["train_data"], config["train_labels"])
    test_data, test_labels = (config["selection_data"], config["selection_labels"])
    print("Train:", train_data.shape, train_labels.shape)
    print("Selection:", test_data.shape, test_labels.shape)
    train_dataset = TensorDataset(
        torch.from_numpy(train_data.astype(np.float32)),
        torch.from_numpy(train_labels).long(),
    )
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )
    test_dataset = TensorDataset(
        torch.from_numpy(test_data.astype(np.float32)),
        torch.from_numpy(test_labels).long(),
    )
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
    params = list(feature_extractor.parameters()) + list(discriminator.parameters())
    optimizer = optim.AdamW(params, lr=base_lr, weight_decay=weight_decay)
    scheduler = WarmupCosineLR(
        optimizer, warmup_epochs=5, max_epochs=epochs, min_lr=1e-05
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    best_test_acc = -1.0
    best_feature_extractor_wts = None
    best_discriminator_wts = None
    epochs_no_improve = 0
    early_stop_patience = 30
    sampling_rate = config.get("sampling_rate", 250)
    window_size = int(config.get("window_sec", 1.0) * sampling_rate)
    step_size = int(config.get("step_sec", 0.1) * sampling_rate)
    print(
        f"Feature params: {count_parameters(feature_extractor):,} | Disc params: {count_parameters(discriminator):,}"
    )
    for epoch in range(epochs):
        feature_extractor.train()
        discriminator.train()
        running_loss, train_correct, train_total = (0.0, 0, 0)
        for full_trials, labels in train_loader:
            full_trials, labels = (full_trials.to(device), labels.to(device))
            window_logits_list = []
            for start in range(0, full_trials.shape[-1] - window_size + 1, step_size):
                window = full_trials[:, :, start : start + window_size]
                if augment:
                    window_np = window.detach().cpu().numpy()
                    window_np = augment_windows(window_np, max_shift=12, noise_std=0.01)
                    window = torch.from_numpy(window_np).to(device)
                tokens = feature_extractor(window.unsqueeze(1))
                logits = discriminator(tokens, return_logits=True)
                window_logits_list.append(logits.unsqueeze(2))
            window_logits = torch.cat(window_logits_list, dim=2)
            mean_logits = torch.mean(window_logits, dim=2)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(mean_logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(params, max_grad_norm)
            optimizer.step()
            running_loss += loss.item()
            preds = mean_logits.argmax(dim=1)
            train_total += labels.size(0)
            train_correct += (preds == labels).sum().item()
        train_acc = 100 * train_correct / train_total
        feature_extractor.eval()
        discriminator.eval()
        test_correct, test_total = (0, 0)
        with torch.no_grad():
            for full_trials, labels in test_loader:
                full_trials, labels = (full_trials.to(device), labels.to(device))
                window_logits_list = []
                for start in range(
                    0, full_trials.shape[-1] - window_size + 1, step_size
                ):
                    window = full_trials[:, :, start : start + window_size].unsqueeze(1)
                    tokens = feature_extractor(window)
                    logits = discriminator(tokens, return_logits=True)
                    window_logits_list.append(logits.unsqueeze(2))
                window_logits = torch.cat(window_logits_list, dim=2)
                mean_logits = torch.mean(window_logits, dim=2)
                preds = mean_logits.argmax(dim=1)
                test_total += labels.size(0)
                test_correct += (preds == labels).sum().item()
        test_acc = 100 * test_correct / test_total
        scheduler.step()
        print(
            f"Epoch {epoch + 1:03d}/{epochs} | Loss: {running_loss / max(1, len(train_loader)):.4f} | Train Acc: {train_acc:.2f}% | Selection Acc: {test_acc:.2f}%"
        )
        if test_acc > best_test_acc + 1e-06:
            best_test_acc = test_acc
            best_feature_extractor_wts = copy.deepcopy(feature_extractor.state_dict())
            best_discriminator_wts = copy.deepcopy(discriminator.state_dict())
            epochs_no_improve = 0
            print(f"  -> New best model found with accuracy: {best_test_acc:.2f}%")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                print(
                    f"Early stopping at epoch {epoch + 1}. Best Acc: {best_test_acc:.2f}%"
                )
                break
    print(
        f"--- [Stage 1] Pre-training complete. Best selection accuracy: {best_test_acc:.2f}% ---"
    )
    return (best_test_acc, best_feature_extractor_wts, best_discriminator_wts)
