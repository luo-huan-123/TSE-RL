"""Liu2024 preprocessing using MOABB downloads and source trial markers."""

from io import BytesIO
import hashlib
from pathlib import Path
import zipfile
import mne
import numpy as np
import pandas as pd
from scipy.io import loadmat
from moabb.datasets.download import data_dl

PATIENT_FOLDS = (
    (2, 5, 7, 11, 16, 18, 22, 29, 32, 48),
    (3, 9, 14, 19, 23, 31, 33, 45, 47, 49),
    (13, 15, 17, 24, 36, 40, 41, 43, 44, 46),
    (1, 4, 6, 12, 20, 26, 34, 37, 39, 50),
    (8, 10, 21, 25, 27, 28, 30, 35, 38, 42),
)
SOURCE_URL = "https://ndownloader.figshare.com/files/38516555"
SOURCE_MD5 = "4336b1ddbde42f89edb82f513a5c0721"


def trial_markers(payload):
    source = loadmat(
        BytesIO(payload),
        struct_as_record=False,
        squeeze_me=True,
        variable_names=["eeg"],
    )["eeg"]
    values, labels = np.asarray(source.rawdata), np.asarray(source.label).reshape(-1)
    if (
        values.shape != (40, 33, 4000)
        or labels.shape != (40,)
        or not np.isin(labels, [1, 2]).all()
    ):
        raise ValueError("Invalid Liu2024 source trial arrays.")
    starts = []
    for trial, trigger in enumerate(values[:, 32, :]):
        onset = np.flatnonzero(trigger == 2)
        if len(onset) != 1 or onset[0] + 2000 > 4000:
            raise ValueError(f"Invalid MI onset in source trial {trial}.")
        starts.append(trial * 4000 + int(onset[0]))
    return np.asarray(starts), np.where(labels == 1, "left_hand", "right_hand")


def load_trials(dataset, subjects, sampling_rate=250):
    """Use source labels/markers instead of ambiguous EDF trigger samples."""
    archive = Path(data_dl(SOURCE_URL, dataset.code))
    digest = hashlib.md5()
    with archive.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != SOURCE_MD5:
        raise ValueError("Unexpected Liu2024 source archive checksum.")
    electrodes, _ = dataset.data_infos()
    channels = [
        name
        for name in pd.read_csv(electrodes, sep="\t")["name"].astype(str)
        if name not in {"HEOL", "HEOR", "VEOR"}
    ]
    if len(channels) != 29 or len(set(channels)) != 29:
        raise ValueError("Expected 29 unique EEG electrodes.")
    parts, labels, rows = [], [], []
    with zipfile.ZipFile(archive) as source:
        for subject in subjects:
            filename = f"sub-{subject:02d}_task-motor-imagery_eeg.mat"
            matches = [
                name for name in source.namelist() if name.endswith("/" + filename)
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"Expected exactly one MAT source for subject {subject}."
                )
            starts, names = trial_markers(source.read(matches[0]))
            raw = mne.io.read_raw_edf(
                dataset.data_path(subject)[0],
                preload=True,
                infer_types=True,
                stim_channel="",
                verbose=False,
            )
            if raw.info["sfreq"] != 500:
                raise ValueError("Expected a 500 Hz source recording.")
            raw.pick(channels)
            raw.filter(8.0, 30.0, method="fir", phase="zero", verbose=False)
            raw.resample(sampling_rate, npad="auto", verbose=False)
            samples = raw.get_data() * 1e6
            starts = np.rint(starts * sampling_rate / 500).astype(int)
            length = 4 * sampling_rate
            trials = np.stack(
                [samples[:, start : start + length] for start in starts]
            ).astype(np.float32)
            if trials.shape != (40, 29, length):
                raise ValueError(
                    f"Unexpected prepared dimensions for subject {subject}."
                )
            parts.append(trials)
            labels.extend(names)
            rows.extend(dict(subject=subject, session="0", run="0") for _ in names)
    return np.concatenate(parts), np.asarray(labels), pd.DataFrame(rows)
