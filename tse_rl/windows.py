"""Trial-aware concatenated EEG replay and causal window extraction."""

from dataclasses import dataclass
import numpy as np


def window_starts(samples, window, step):
    if window < 1 or step < 1 or samples < window:
        raise ValueError("Require positive window/step and at least one full window.")
    return range(0, samples - window + 1, step)


@dataclass
class EEGStream:
    samples: np.ndarray
    boundaries: np.ndarray

    def windows(self, window, step):
        """Yield (trial, absolute start, EEG); windows never cross trial boundaries."""
        for trial, (left, right) in enumerate(self.boundaries):
            for start in window_starts(right - left, window, step):
                absolute = left + start
                yield trial, absolute, self.samples[:, absolute : absolute + window]


def concatenate_trials(trials):
    if trials.ndim != 3 or not len(trials):
        raise ValueError("Expected nonempty (trials, channels, samples) EEG.")
    count, channels, length = trials.shape
    starts = np.arange(count, dtype=np.int64) * length
    samples = trials.transpose(1, 0, 2).reshape(channels, -1)
    return EEGStream(samples, np.column_stack([starts, starts + length]))
