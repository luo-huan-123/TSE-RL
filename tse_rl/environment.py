"""FSM-guided window selection with the paper's rewards and aggregation."""

from collections import deque
from enum import IntEnum
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from .windows import concatenate_trials, window_starts


class Action(IntEnum):
    DISCARD = 0
    SUSTAIN = 1
    ONSET = 2
    OFFSET = 3


ACTION_WEIGHTS = {Action.SUSTAIN: 1.0, Action.ONSET: 1.5, Action.OFFSET: 0.8}
MASKS = np.array([[1, 0, 1, 0], [1, 1, 1, 0], [1, 1, 0, 1]], dtype=bool)


def aggregate(segment):
    if not segment:
        return -1
    probabilities, actions = zip(*segment)
    weights = np.array([ACTION_WEIGHTS[a] for a in actions])
    return int(np.average(np.stack(probabilities), axis=0, weights=weights).argmax())


class WindowSelectionEnv(gym.Env):
    """One episode per known trial; class labels are never policy observations.

    Labels are required only for supervised training rewards. Set reward_enabled
    to False for label-free inference, then score predictions outside the env.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        data,
        labels,
        feature,
        discriminator,
        sampling_rate=250,
        window_sec=1.0,
        step_sec=0.1,
        history_len=5,
        reward_enabled=True,
    ):
        super().__init__()
        self.data, self.labels = data, labels
        self.stream = concatenate_trials(data)
        self.feature, self.discriminator = feature.eval(), discriminator.eval()
        self.device = next(feature.parameters()).device
        self.rate, self.history_len = sampling_rate, history_len
        self.window = int(window_sec * sampling_rate)
        self.step_size = int(step_sec * sampling_rate)
        self.starts = list(window_starts(data.shape[-1], self.window, self.step_size))
        self.reward_enabled = reward_enabled
        if reward_enabled and labels is None:
            raise ValueError("Training rewards require labels.")
        channels = data.shape[1]
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Dict(
            {
                "current_obs": spaces.Box(
                    -np.inf, np.inf, (1, channels, self.window), np.float32
                ),
                "history_obs": spaces.Box(
                    -np.inf, np.inf, (history_len, 1, channels, self.window), np.float32
                ),
                "trial_progress": spaces.Box(0.0, 1.0, (1,), np.float32),
                "action_mask": spaces.Box(0, 1, (4,), np.int8),
            }
        )
        self.history = deque(maxlen=history_len)

    def action_masks(self):
        return MASKS[self.phase].copy()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.trial = (
            int(options["trial_id"])
            if options and "trial_id" in options
            else int(self.np_random.integers(len(self.data)))
        )
        self.index, self.phase, self.last_keep = 0, 0, -1
        self.segment, self.actions = [], []
        self.finished = False
        self.history.clear()
        self.history.extend(
            np.zeros(self.observation_space["history_obs"].shape, np.float32)
        )
        return self._observation(), {}

    def _window(self):
        start = self.stream.boundaries[self.trial, 0] + self.starts[self.index]
        return self.stream.samples[:, start : start + self.window][None].astype(
            np.float32
        )

    def _observation(self):
        progress = self.starts[self.index] / max(1, self.data.shape[-1] - self.window)
        return {
            "current_obs": self._window(),
            "history_obs": np.asarray(self.history, np.float32),
            "trial_progress": np.array([progress], np.float32),
            "action_mask": self.action_masks().astype(np.int8),
        }

    def step(self, action):
        if self.finished:
            raise RuntimeError("Call reset before stepping a completed trial.")
        action = int(action)
        if action not in range(4) or not self.action_masks()[action]:
            raise ValueError(f"Action {action} is invalid in FSM phase {self.phase}.")
        window = self._window()
        with torch.no_grad():
            probabilities = (
                self.discriminator(
                    self.feature(torch.from_numpy(window[None]).to(self.device))
                )[0]
                .cpu()
                .numpy()
            )
        reward = 0.0
        if self.reward_enabled:
            q = float(probabilities[self.labels[self.trial]])
            reward += 2 * (q - 0.5) if action else 0.2 * (0.5 - q)
            reward += 0.1 if action > 0 and action == self.last_keep else 0
            reward += (
                0.2
                if self.last_keep == Action.ONSET and action == Action.SUSTAIN
                else 0
            )
            reward -= 0.1 if action == Action.DISCARD and self.phase != 0 else 0
        self.actions.append(action)
        if action:
            self.segment.append((probabilities, action))
            self.last_keep = action
        if self.phase == 0 and action == Action.ONSET:
            self.phase = 1
        elif self.phase == 1 and action == Action.SUSTAIN:
            self.phase = 2
        self.history.append(window)
        self.finished = action == Action.OFFSET or self.index == len(self.starts) - 1
        info = {}
        if self.finished:
            prediction = aggregate(self.segment)
            if self.reward_enabled:
                correct = prediction == self.labels[self.trial]
                if action == Action.OFFSET:
                    reward += (5 if correct else -3) * (
                        1 + 0.1 * max(0, len(self.segment) - 3)
                    )
                elif not self.segment:
                    reward -= 5
                else:
                    reward += 2 if correct else -3
            info = dict(
                prediction=prediction,
                kept_windows=len(self.segment),
                actions=self.actions.copy(),
                decision_time=(self.starts[self.index] + self.window) / self.rate,
                actual_offset=action == Action.OFFSET,
            )
            # Keep a full-size terminal observation; do not synthesize an Offset.
        else:
            self.index += 1
        return self._observation(), float(reward), self.finished, False, info
