"""Construct the eight paper backbones and their matching policy encoders."""

import importlib
import copy
import torch

MODULES = {
    "EEGNet": "eegnet",
    "TSLANet": "tslanet",
    "EEGConformer": "eegconformer",
    "FBCNet": "fbcnet",
    "ATCNet": "atcnet",
    "M-ShallowConvNet": "mshallowconvnet",
    "mAtt": "matt",
    "CSPNet": "cspnet",
}


def model_factories(module, backbone: str, CLASSES, CHANNELS, WINDOW_SAMPLES):
    if backbone == "EEGNet":

        def make_fe():
            return module.EEGNetFeatureExtractor(
                CLASSES, CHANNELS, WINDOW_SAMPLES, 0.25
            )

        def make_disc():
            return module.Discriminator(CLASSES, CHANNELS, WINDOW_SAMPLES, 0.25)

        policy_extra = {}
        unfreeze = lambda model: (
            model.policy.features_extractor.current_obs_extractor.block_2.parameters()
        )
    elif backbone == "TSLANet":
        fe_config = {
            "chunk_size": WINDOW_SAMPLES,
            "num_electrodes": CHANNELS,
            "patch_size": 40,
            "emb_dim": 96,
        }
        disc_config = {
            "emb_dim": 96,
            "num_classes": CLASSES,
            "depth": 3,
            "dropout_rate": 0.3,
            "mlp_ratio": 2.0,
            "drop_token": 0.2,
            "drop_path_rate": 0.2,
        }
        make_fe = lambda: module.TSLANetFeatureExtractor(**fe_config)
        make_disc = lambda: module.TSLANetDiscriminator(**disc_config)
        policy_extra = {"patch_size": 40, "emb_dim": 96}
        unfreeze = lambda model: (
            model.policy.features_extractor.current_obs_extractor.parameters()
        )
    elif backbone == "EEGConformer":
        fe_config = {
            "num_electrodes": CHANNELS,
            "chunk_size": WINDOW_SAMPLES,
            "hid_channels": 40,
            "depth": 6,
            "heads": 10,
            "dropout": 0.5,
            "forward_expansion": 4,
            "forward_dropout": 0.5,
        }
        disc_config = {
            "feature_extractor_config": fe_config,
            "num_classes": CLASSES,
            "cls_channels": 32,
            "cls_dropout": 0.5,
        }
        make_fe = lambda: module.EEGConformerFeatureExtractor(**fe_config)
        make_disc = lambda: module.EEGConformerDiscriminator(**disc_config)
        policy_extra = {"pretrained_tail_from_discriminator": True}
        unfreeze = lambda model: (
            model.policy.features_extractor.current_obs_extractor.stem.parameters()
        )
    elif backbone == "FBCNet":
        fe_config = {
            "num_electrodes": CHANNELS,
            "chunk_size": WINDOW_SAMPLES,
            "in_channels": 1,
            "num_S": 32,
            "stride_factor": (
                10 if WINDOW_SAMPLES % 10 == 0 and WINDOW_SAMPLES < 1000 else 4
            ),
            "temporal": "LogVarLayer",
            "weight_norm": True,
        }
        disc_config = {
            "feature_extractor_config": fe_config,
            "num_classes": CLASSES,
            "weight_norm": True,
        }
        make_fe = lambda: module.FBCNetFeatureExtractor(**fe_config)
        make_disc = lambda: module.FBCNetDiscriminator(**disc_config)
        policy_extra = {
            "in_channels": 1,
            "num_S": 32,
            "temporal": "LogVarLayer",
            "stride_factor": (
                10 if WINDOW_SAMPLES % 10 == 0 and WINDOW_SAMPLES < 1000 else 4
            ),
        }
        unfreeze = lambda model: (
            model.policy.features_extractor.current_obs_extractor.parameters()
        )
    elif backbone == "ATCNet":
        fe_config = {"num_electrodes": CHANNELS, "chunk_size": WINDOW_SAMPLES}
        disc_config = {"feature_extractor_config": fe_config, "num_classes": CLASSES}
        make_fe = lambda: module.ATCNetFeatureExtractor(**fe_config)
        make_disc = lambda: module.ATCNetDiscriminator(**disc_config)
        policy_extra = {}
        unfreeze = lambda model: (
            model.policy.features_extractor.current_obs_extractor.conv_block.parameters()
        )
    elif backbone == "M-ShallowConvNet":
        fe_config = {"num_electrodes": CHANNELS, "chunk_size": WINDOW_SAMPLES}
        disc_config = {"feature_extractor_config": fe_config, "num_classes": CLASSES}
        make_fe = lambda: module.MShallowFeatureExtractor(**fe_config)
        make_disc = lambda: module.MShallowDiscriminator(**disc_config)
        policy_extra = {}
        unfreeze = lambda model: (
            model.policy.features_extractor.current_obs_extractor.spatial_conv.parameters()
        )
    elif backbone == "mAtt":
        fe_config = {
            "chunk_size": WINDOW_SAMPLES,
            "num_electrodes": CHANNELS,
            "num_heads": 4,
            "proj_dim": 13,
        }
        disc_config = {
            "feature_extractor_config": fe_config,
            "num_classes": CLASSES,
            "att_hidden": 256,
            "mlp_hidden": 256,
            "dropout": 0.2,
        }
        make_fe = lambda: module.MAttFeatureExtractor(**fe_config)
        make_disc = lambda: module.MAttDiscriminator(**disc_config)
        policy_extra = {"fe_kwargs": fe_config}
        unfreeze = lambda model: (
            model.policy.features_extractor.current_obs_extractor.parameters()
        )
    elif backbone == "CSPNet":
        fe_config = {
            "chunk_size": WINDOW_SAMPLES,
            "num_electrodes": CHANNELS,
            "num_filters_t": 20,
            "filter_size_t": 25,
            "num_filters_s": 2,
            "filter_size_s": -1,
            "pool_size_1": 100,
            "pool_stride_1": 5,
            "dropout": 0.5,
            "min_policy_dim": 150,
        }
        disc_config = {"feature_extractor_config": fe_config, "num_classes": CLASSES}
        make_fe = lambda: module.CSPNetFeatureExtractor(**fe_config)
        make_disc = lambda: module.CSPNetDiscriminator(**disc_config)
        policy_extra = {"fe_kwargs": fe_config}
        unfreeze = lambda model: (
            model.policy.features_extractor.current_obs_extractor.parameters()
        )
    else:
        raise KeyError(backbone)
    return (make_fe, make_disc, policy_extra, unfreeze)


def build(name, classes, channels, samples, device="cpu"):
    module = importlib.import_module(f"tse_rl.backbones.{MODULES[name]}")
    make_f, make_d, extra, unfreeze = model_factories(
        module, name, classes, channels, samples
    )
    feature, discriminator = make_f().to(device), make_d().to(device)
    # Initialize lazy layers before state dictionaries are saved or compared.
    feature.eval()
    discriminator.eval()
    with torch.no_grad():
        discriminator(feature(torch.zeros(2, 1, channels, samples, device=device)))
    return module, feature, discriminator, extra, unfreeze


def policy_kwargs(module, feature, discriminator, extra, channels, samples):
    feature = copy.deepcopy(feature).cpu().eval()
    with torch.no_grad():
        size = feature(torch.zeros(1, 1, channels, samples)).flatten(1).shape[1]
    extra = dict(extra)
    if extra.pop("pretrained_tail_from_discriminator", False):
        extra["pretrained_tail"] = copy.deepcopy(discriminator.tail).cpu()
    return dict(
        features_extractor_class=module.CombinedFeatureExtractor,
        features_extractor_kwargs=dict(
            cnn_feature_dim=size,
            history_feature_dim=64,
            pretrained_feature_extractor=feature,
            **extra,
        ),
    )
