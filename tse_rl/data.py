"""MOABB acquisition, preprocessing and explicit trial/subject splits."""

from pathlib import Path
import json
import os
import numpy as np

DATASETS = {
    "BNCI2014001": ("BNCI2014_001", 250, 4.0, 32.0),
    "BNCI2014002": ("BNCI2014_002", 512, 5.0, 32.0),
    "BNCI2015001": ("BNCI2015_001", 512, 5.0, 32.0),
    "OpenBMI": ("Lee2019_MI", 1000, 4.0, 32.0),
    "Liu2024": ("Liu2024", 250, 4.0, 30.0),
}
ALIASES = {"Lee2019": "OpenBMI", "Lee2019_MI": "OpenBMI"}


def dataset_instance(name, data_root):
    from moabb import datasets

    name = ALIASES.get(name, name)
    root = str(Path(data_root).expanduser().resolve())
    Path(root).mkdir(parents=True, exist_ok=True)
    os.environ["MNE_DATA"] = root
    # Set all download keys in process, without changing the user's MNE config.
    for key in ("BNCI", "LEE2019-MI", "LIU2024"):
        os.environ[f"MNE_DATASETS_{key}_PATH"] = root
    cls = getattr(datasets, DATASETS[name][0])
    if name == "OpenBMI":
        return cls(train_run=True, test_run=True, sessions=(1, 2))
    return cls()


def within_subject_indices(name, metadata):
    """Use the original session/run split, before creating any sliding windows."""
    sessions = list(dict.fromkeys(metadata.session.astype(str)))
    if name == "BNCI2014002":
        runs = metadata.run.astype(str)
        train = np.flatnonzero(runs.str.endswith("train"))
        test = np.flatnonzero(runs.str.endswith("test"))
        if not len(train) or not len(test):
            raise ValueError(
                "MOABB run metadata must identify training and testing runs."
            )
    elif name == "Liu2024":
        raise ValueError(
            "Liu2024 uses patient-fold or loso; it has only one session per patient."
        )
    else:
        if len(sessions) < 2:
            raise ValueError("Two sessions are required for a within-subject split.")
        train = np.flatnonzero(metadata.session.astype(str) == sessions[0])
        test = np.flatnonzero(metadata.session.astype(str) == sessions[1])
    return train, test


def prepare(args):
    import moabb
    from moabb.paradigms import MotorImagery

    name = ALIASES.get(args.dataset, args.dataset)
    dataset = dataset_instance(name, args.data_root)
    _, default_rate, duration, high = DATASETS[name]
    rate = args.resample or default_rate
    subjects = (
        [args.subject] if args.protocol == "within-subject" else dataset.subject_list
    )
    if args.subject not in dataset.subject_list:
        raise ValueError(f"Subject must be one of {dataset.subject_list}")
    if name == "Liu2024" and args.protocol == "within-subject":
        raise ValueError("Use --protocol patient-fold or --protocol loso for Liu2024.")
    if args.protocol == "patient-fold" and name != "Liu2024":
        raise ValueError("patient-fold is defined for Liu2024 only.")
    paradigm = MotorImagery(
        n_classes=len(dataset.event_id),
        fmin=8.0,
        fmax=high,
        tmin=0,
        tmax=duration,
        resample=rate,
    )
    # MOABB returns EEG amplitudes in microvolts and excludes non-EEG channels.
    if name == "Liu2024":
        from .liu2024 import load_trials

        x, label_names, metadata = load_trials(dataset, subjects, rate)
    else:
        x, label_names, metadata = paradigm.get_data(dataset, subjects=subjects)
    labels, y = np.unique(label_names, return_inverse=True)
    x = np.asarray(x, dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError("The downloaded/preprocessed EEG contains non-finite values.")
    if args.protocol == "within-subject":
        train, test = within_subject_indices(name, metadata)
    else:
        eligible = np.ones(len(y), dtype=bool)
        if name != "Liu2024":
            # Cross-subject experiments use the first session of each participant.
            for subject in subjects:
                rows = metadata.subject.to_numpy() == subject
                first = metadata.loc[rows, "session"].iloc[0]
                eligible[rows] = (metadata.loc[rows, "session"] == first).to_numpy()
        if args.protocol == "loso":
            test_subjects = [args.subject]
        else:
            from .liu2024 import PATIENT_FOLDS

            test_subjects = list(PATIENT_FOLDS[args.fold])
        held_out = metadata.subject.isin(test_subjects).to_numpy()
        train, test = (
            np.flatnonzero(eligible & ~held_out),
            np.flatnonzero(eligible & held_out),
        )
    original_train = train.copy()
    selection = test.copy()
    # This switch is only for functional smoke tests, and is recorded in the manifest.
    if args.limit_trials:
        train, selection, test = [
            indices[: args.limit_trials] for indices in (train, selection, test)
        ]
    if min(map(len, (train, selection, test))) == 0:
        raise ValueError("Every split must contain at least one trial.")
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.with_suffix(".json").exists():
        raise FileExistsError(f"Choose a new output path: {target}")
    np.savez_compressed(
        target,
        train_x=x[train],
        train_y=y[train],
        selection_x=x[selection],
        selection_y=y[selection],
        test_x=x[test],
        test_y=y[test],
    )
    manifest = dict(
        dataset=name,
        moabb_class=DATASETS[name][0],
        moabb_version=moabb.__version__,
        protocol=args.protocol,
        selection="test",
        subject=args.subject,
        fold=args.fold,
        seed=args.seed,
        sampling_rate=rate,
        channels=int(x.shape[1]),
        classes=labels.tolist(),
        samples=int(x.shape[2]),
        filter_hz=[8.0, high],
        epoch_seconds=[0, duration],
        unit="microvolt",
        limit_trials=args.limit_trials,
        original_train_count=len(original_train),
        indices={
            k: v.tolist()
            for k, v in dict(train=train, selection=selection, test=test).items()
        },
        trial_metadata=metadata.astype(str).to_dict(orient="records"),
    )
    target.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "prepared": str(target),
                "counts": {
                    k: len(v)
                    for k, v in dict(
                        train=train, selection=selection, test=test
                    ).items()
                },
                "shape": x.shape,
            },
            indent=2,
        )
    )


def load_prepared(path):
    path = Path(path)
    meta = json.loads(path.with_suffix(".json").read_text())
    with np.load(path, allow_pickle=False) as values:
        arrays = {key: values[key] for key in values.files}
    for split in ("train", "selection", "test"):
        x, y = arrays[f"{split}_x"], arrays[f"{split}_y"]
        if x.ndim != 3 or len(x) != len(y) or not np.isfinite(x).all():
            raise ValueError(f"Invalid {split} array")
    return arrays, meta
