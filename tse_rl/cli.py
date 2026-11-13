"""Portable command-line entry points for data, training and evaluation."""

import argparse
from .backbones import MODULES
from .data import DATASETS, ALIASES


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m tse_rl")
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser(
        "prepare", help="Download through MOABB and prepare a split"
    )
    prep.add_argument(
        "--dataset", choices=list(DATASETS) + list(ALIASES), default="BNCI2014001"
    )
    prep.add_argument(
        "--subject", type=int, default=1, help="MOABB uses one-based subject IDs"
    )
    prep.add_argument(
        "--protocol",
        choices=["within-subject", "loso", "patient-fold"],
        default="within-subject",
    )
    prep.add_argument("--fold", type=int, choices=range(5), default=0)
    prep.add_argument("--seed", type=int, default=42)
    prep.add_argument(
        "--resample", type=int, help="Override the dataset's default sampling rate"
    )
    prep.add_argument("--data-root", default="data/raw")
    prep.add_argument("--output", default="data/prepared/bnci001_s1.npz")
    prep.add_argument(
        "--limit-trials", type=int, default=0, help="Per-split cap for smoke tests only"
    )
    for name in ("stage1", "stage2", "stage3", "evaluate"):
        sub = commands.add_parser(name)
        sub.add_argument("--data", default="data/prepared/bnci001_s1.npz")
        sub.add_argument("--run-dir", default="runs/bnci001_s1_eegnet")
        sub.add_argument(
            "--device",
            default="cpu",
            help="cpu, cuda, or a CUDA device index such as cuda:0",
        )
        sub.add_argument("--threads", type=int, default=1)
        if name == "stage1":
            sub.add_argument("--backbone", choices=list(MODULES), default="EEGNet")
            sub.add_argument("--epochs", type=int, default=300)
            sub.add_argument("--batch-size", type=int)
            sub.add_argument("--seed", type=int, default=42)
            sub.add_argument("--window-sec", type=float, default=1.0)
            sub.add_argument("--step-sec", type=float, default=0.1)
            sub.add_argument("--history-len", type=int, default=5)
        elif name in ("stage2", "stage3"):
            sub.add_argument(
                "--steps", type=int, default=40000 if name == "stage2" else 20000
            )
            sub.add_argument("--n-steps", type=int, default=128)
            sub.add_argument("--n-envs", type=int, default=4)
            sub.add_argument("--batch-size", type=int, default=64)
            sub.add_argument("--ppo-epochs", type=int, default=10)
            sub.add_argument(
                "--lr", type=float, default=3e-4 if name == "stage2" else 3e-5
            )
            sub.add_argument("--eval-every", type=int, default=512)
        else:
            sub.add_argument("--stage", type=int, choices=[2, 3], default=3)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        from .data import prepare

        if args.limit_trials < 0 or (args.resample is not None and args.resample < 1):
            parser.error(
                "limit-trials must be nonnegative and resample must be positive"
            )
        if not args.output.endswith(".npz"):
            parser.error("--output must end in .npz")
        prepare(args)
    else:
        from . import training

        if args.threads < 1:
            parser.error("threads must be positive")
        if args.command == "stage1":
            if args.history_len < 1:
                parser.error("history-len must be positive")
            training.stage1(args)
        elif args.command in ("stage2", "stage3"):
            training.train_policy(args, int(args.command[-1]))
        else:
            training.evaluate(args)
