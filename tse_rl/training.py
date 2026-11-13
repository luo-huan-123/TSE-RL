"""Three-stage training with separately stored frozen reward models."""

import hashlib
import json
import random
from pathlib import Path
import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import FloatSchedule
from .backbones import build, policy_kwargs
from .data import load_prepared
from .environment import WindowSelectionEnv
from .evaluation import evaluate_policy


def seed_everything(seed, threads=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def dump(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2) + "\n")


def data_fingerprint(path):
    digest = hashlib.sha256()
    for source in (Path(path), Path(path).with_suffix(".json")):
        with source.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def stage1(args):
    seed_everything(args.seed, args.threads)
    arrays, meta = load_prepared(args.data)
    if args.batch_size is None:
        args.batch_size = {
            "EEGNet": 288,
            "TSLANet": 64,
            "EEGConformer": 32,
            "FBCNet": 64,
            "ATCNet": 32,
            "M-ShallowConvNet": 32,
            "mAtt": 64,
            "CSPNet": 64,
        }[args.backbone]
    output = Path(args.run_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new run directory to avoid mixing experiments.")
    output.mkdir(parents=True, exist_ok=True)
    samples = int(args.window_sec * meta["sampling_rate"])
    module, feature, disc, _, _ = build(
        args.backbone, len(meta["classes"]), meta["channels"], samples, args.device
    )
    config = dict(
        sampling_rate=meta["sampling_rate"],
        window_sec=args.window_sec,
        step_sec=args.step_sec,
        history_len=args.history_len,
        data_name=meta["dataset"],
        subject_id=meta["subject"],
        train_data=arrays["train_x"],
        train_labels=arrays["train_y"],
        selection_data=arrays["selection_x"],
        selection_labels=arrays["selection_y"],
    )
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must be positive.")
    from .windows import window_starts

    list(
        window_starts(
            meta["samples"], samples, int(args.step_sec * meta["sampling_rate"])
        )
    )
    if meta["dataset"] == "Liu2024" and args.backbone != "TSLANet":
        from .backbones.logits import forward_logits

        # Keep the Liu2024 supervised objective on mean window logits.
        disc.forward = lambda features: forward_logits(args.backbone, disc, features)
    best, f_state, d_state = module.pretrain_discriminator(
        feature, disc, config, epochs=args.epochs, batch_size=args.batch_size
    )
    torch.save(dict(feature=f_state, discriminator=d_state), output / "stage1.pt")
    record = {k: v for k, v in config.items() if not k.endswith(("_data", "_labels"))}
    record.update(
        backbone=args.backbone,
        classes=len(meta["classes"]),
        channels=meta["channels"],
        samples=samples,
        seed=args.seed,
        selection=meta["selection"],
        protocol=meta["protocol"],
        data_fingerprint=data_fingerprint(args.data),
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    dump(output / "config.json", record)
    dump(output / "stage1_metrics.json", {"best_selection_accuracy": best})


def restore(args):
    root = Path(args.run_dir)
    cfg = json.loads((root / "config.json").read_text())
    if data_fingerprint(args.data) != cfg["data_fingerprint"]:
        raise ValueError("The prepared data or manifest differs from Stage 1.")
    seed_everything(cfg["seed"], args.threads)
    arrays, _ = load_prepared(args.data)
    module, feature, disc, extra, unfreeze = build(
        cfg["backbone"], cfg["classes"], cfg["channels"], cfg["samples"], args.device
    )
    state = torch.load(root / "stage1.pt", map_location=args.device, weights_only=True)
    feature.load_state_dict(state["feature"])
    disc.load_state_dict(state["discriminator"])
    feature.eval().requires_grad_(False)
    disc.eval().requires_grad_(False)
    return root, cfg, arrays, module, feature, disc, extra, unfreeze


def environment(arrays, split, feature, disc, cfg, train=False):
    return WindowSelectionEnv(
        arrays[f"{split}_x"],
        arrays[f"{split}_y"] if train else None,
        feature,
        disc,
        sampling_rate=cfg["sampling_rate"],
        window_sec=cfg["window_sec"],
        step_sec=cfg["step_sec"],
        history_len=cfg["history_len"],
        reward_enabled=train,
    )


class SelectCheckpoint(BaseCallback):
    def __init__(self, env, labels, path, frequency):
        super().__init__()
        self.env, self.labels, self.path, self.frequency = env, labels, path, frequency
        self.best, self.records = -1.0, []

    def check(self):
        report = evaluate_policy(self.model, self.env, self.labels)
        report.pop("predictions")
        report["timesteps"] = self.model.num_timesteps
        self.records.append(report)
        print(
            f"Selection @ {self.model.num_timesteps}: {report['accuracy']:.2f}%",
            flush=True,
        )
        if report["accuracy"] > self.best:
            self.best = report["accuracy"]
            self.model.save(self.path)

    def _on_training_start(self):
        self.check()

    def _on_step(self):
        return True

    def _on_rollout_start(self):
        # Called after the preceding PPO update; checks actual trained parameters.
        if self.num_timesteps and (
            not self.records
            or self.num_timesteps - self.records[-1]["timesteps"] >= self.frequency
        ):
            self.check()

    def _on_training_end(self):
        self.check()


def train_policy(args, stage):
    root, cfg, arrays, module, feature, disc, extra, unfreeze = restore(args)
    if (root / f"stage{stage}.zip").exists():
        raise FileExistsError(
            f"Stage {stage} already exists. Use a separate run directory."
        )
    if (
        args.steps < 1
        or args.n_steps < 2
        or args.ppo_epochs < 1
        or args.batch_size < 2
        or args.n_envs < 1
    ):
        raise ValueError("Use positive steps/epochs/envs and n-steps/batch-size >= 2.")
    train = DummyVecEnv(
        [
            lambda: environment(arrays, "train", feature, disc, cfg, True)
            for _ in range(args.n_envs)
        ]
    )
    selection = environment(arrays, "selection", feature, disc, cfg)
    if stage == 2:
        kwargs = policy_kwargs(
            module, feature, disc, extra, cfg["channels"], cfg["samples"]
        )
        model = MaskablePPO(
            "MultiInputPolicy",
            train,
            policy_kwargs=kwargs,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.ppo_epochs,
            learning_rate=args.lr,
            seed=cfg["seed"],
            device=args.device,
            verbose=1,
        )
    else:
        model = MaskablePPO.load(root / "stage2.zip", env=train, device=args.device)
        if model.n_envs != args.n_envs:
            raise ValueError("Stage 3 must use the same n-envs as Stage 2.")
        # Loading recreates the rollout buffer. Keep its shape aligned to n_steps.
        model.n_steps, model.batch_size, model.n_epochs = (
            args.n_steps,
            args.batch_size,
            args.ppo_epochs,
        )
        model.rollout_buffer = model.rollout_buffer_class(
            model.n_steps,
            model.observation_space,
            model.action_space,
            device=model.device,
            gamma=model.gamma,
            gae_lambda=model.gae_lambda,
            n_envs=model.n_envs,
            **model.rollout_buffer_kwargs,
        )
        model.learning_rate = args.lr
        model.lr_schedule = FloatSchedule(args.lr)
        for group in model.policy.optimizer.param_groups:
            group["lr"] = args.lr
    if stage == 2:
        # PPO initializes submodules during construction; restore pretrained F afterwards.
        encoder = model.policy.features_extractor
        encoder.current_obs_extractor.load_state_dict(feature.state_dict())
        encoder.history_obs_extractor.feature_extractor.load_state_dict(
            feature.state_dict(), strict=False
        )
        if hasattr(encoder.history_obs_extractor, "tail"):
            encoder.history_obs_extractor.tail.load_state_dict(disc.tail.state_dict())
    current = model.policy.features_extractor.current_obs_extractor
    current.requires_grad_(False)
    if stage == 3:
        for parameter in unfreeze(model):
            parameter.requires_grad_(True)
    before = {k: v.detach().cpu().clone() for k, v in current.named_parameters()}
    callback = SelectCheckpoint(
        selection, arrays["selection_y"], root / f"stage{stage}", args.eval_every
    )
    model.learn(
        total_timesteps=args.steps, callback=callback, reset_num_timesteps=(stage == 2)
    )
    model.save(root / f"stage{stage}_last")
    changed = [
        k
        for k, v in current.named_parameters()
        if not torch.equal(before[k], v.detach().cpu())
    ]
    dump(
        root / f"stage{stage}_metrics.json",
        dict(
            selection=cfg["selection"],
            records=callback.records,
            requested_timesteps=args.steps,
            completed_timesteps=model.num_timesteps,
            current_feature_parameters_changed=changed,
            learning_rate=model.policy.optimizer.param_groups[0]["lr"],
        ),
    )
    train.close()
    selection.close()


def evaluate(args):
    root, cfg, arrays, _, feature, disc, _, _ = restore(args)
    model = MaskablePPO.load(root / f"stage{args.stage}.zip", device=args.device)
    env = environment(arrays, "test", feature, disc, cfg)
    report = evaluate_policy(model, env, arrays["test_y"])
    report["checkpoint_selection"] = cfg["selection"]
    destination = root / f"stage{args.stage}_evaluation.json"
    dump(destination, report)
    print(json.dumps({k: v for k, v in report.items() if k != "predictions"}, indent=2))
    env.close()
