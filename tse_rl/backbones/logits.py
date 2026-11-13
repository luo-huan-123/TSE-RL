"""Pre-softmax F/D outputs used by Liu2024 supervised training."""

import torch


def forward_logits(
    backbone: str, discriminator, features: torch.Tensor
) -> torch.Tensor:
    """Return pre-softmax classifier values without changing frozen backbone files."""
    if backbone == "EEGNet":
        x = discriminator.dropout(features)
        x = discriminator.block_3(x)
        x = discriminator.dropout(x)
        return discriminator.clf(x.flatten(1))
    if backbone == "TSLANet":
        return discriminator(features, return_logits=True)
    if backbone == "EEGConformer":
        return discriminator.cls(discriminator.tail(features))
    if backbone == "FBCNet":
        return discriminator.last_layer[0](features)
    if backbone == "ATCNet":
        x = features[:, :, -1, :].permute(0, 2, 1)
        total = 0
        for index in range(discriminator.num_windows):
            start = index
            stop = x.shape[1] - discriminator.num_windows + index + 1
            branch = x[:, start:stop, :]
            attention, _ = discriminator.get_submodule(f"msa{index}")(
                branch, branch, branch
            )
            branch = branch + discriminator.get_submodule(f"msa_drop{index}")(attention)
            for depth in range(discriminator.tcn_depth):
                branch = branch + discriminator.get_submodule(f"tcn_{index}_{depth}")(
                    branch
                )
            total = total + discriminator.get_submodule(f"dense{index}")(
                branch[:, -1, :]
            )
        return total / discriminator.num_windows
    if backbone == "M-ShallowConvNet":
        return discriminator.classifier(
            discriminator.dropout(discriminator.flatten(features))
        )
    if backbone == "mAtt":
        batch, heads, _, _ = features.shape
        values = features.reshape(batch, heads, -1)
        scores = discriminator.att_mlp(values).squeeze(-1)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        pooled = discriminator.dropout((values * weights).sum(dim=1))
        return discriminator.cls_mlp(pooled)
    if backbone == "CSPNet":
        x = discriminator.drop(features)
        if discriminator.use_pre_pool:
            x = discriminator.drop(discriminator.log(discriminator.pool(x)))
        return discriminator.classifier[0](x)[:, :, 0, 0]
    raise KeyError(backbone)
