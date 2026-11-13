"""Functional checks for imports, F/D interfaces, FSM and replay boundaries."""

import importlib
import pkgutil
import unittest
import numpy as np
import torch
import tse_rl
from tse_rl.backbones import MODULES, build, policy_kwargs
from tse_rl.environment import WindowSelectionEnv, aggregate
from tse_rl.windows import concatenate_trials


class ConstantFeature(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return x


class ConstantHead(torch.nn.Module):
    def forward(self, x):
        return torch.tensor([[0.8, 0.2]], device=x.device).repeat(len(x), 1)


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_import_every_module(self):
        for entry in pkgutil.walk_packages(tse_rl.__path__, tse_rl.__name__ + "."):
            importlib.import_module(entry.name)

    def test_backbones_and_matching_history_encoders(self):
        for name in MODULES:
            with self.subTest(backbone=name):
                torch.manual_seed(42)
                module, feature, head, extra, _ = build(name, 4, 22, 250)
                x = torch.randn(2, 1, 22, 250)
                output = head(feature(x))
                self.assertEqual(output.shape, (2, 4))
                self.assertTrue(torch.isfinite(output).all())
                torch.testing.assert_close(output.sum(1), torch.ones(2))
                (-output[:, 0].clamp_min(1e-8).log().mean()).backward()
                self.assertTrue(
                    all(
                        torch.isfinite(p.grad).all()
                        for p in feature.parameters()
                        if p.grad is not None
                    )
                )
                env = WindowSelectionEnv(
                    np.zeros((2, 22, 1000), np.float32),
                    None,
                    feature,
                    head,
                    reward_enabled=False,
                )
                kwargs = policy_kwargs(module, feature, head, extra, 22, 250)
                encoder = kwargs["features_extractor_class"](
                    env.observation_space, **kwargs["features_extractor_kwargs"]
                ).eval()
                obs, _ = env.reset()
                values = {
                    key: torch.from_numpy(value[None]).float()
                    for key, value in obs.items()
                }
                embedding = encoder(values)
                self.assertEqual(embedding.shape, (1, encoder.features_dim))
                self.assertTrue(torch.isfinite(embedding).all())

    def test_bnci002_split_uses_recording_role(self):
        import pandas as pd
        from tse_rl.data import within_subject_indices

        for runs in (
            ["0train"] * 100 + ["1test"] * 60,
            [f"{i}train" for i in range(5) for _ in range(20)]
            + [f"{i}test" for i in range(5, 8) for _ in range(20)],
        ):
            metadata = pd.DataFrame({"session": ["0"] * 160, "run": runs})
            train, test = within_subject_indices("BNCI2014002", metadata)
            np.testing.assert_array_equal(train, np.arange(100))
            np.testing.assert_array_equal(test, np.arange(100, 160))

    def make_env(self, labels=True):
        return WindowSelectionEnv(
            np.zeros((2, 2, 40), np.float32),
            np.array([0, 1]) if labels else None,
            ConstantFeature(),
            ConstantHead(),
            sampling_rate=10,
            reward_enabled=labels,
        )

    def test_fsm_terminal_reward_and_real_offset(self):
        env = self.make_env()
        obs, _ = env.reset(options={"trial_id": 0})
        self.assertEqual(env.action_masks().tolist(), [True, False, True, False])
        with self.assertRaises(ValueError):
            env.step(3)
        env.step(2)
        self.assertEqual(env.action_masks().tolist(), [True, True, True, False])
        env.step(1)
        self.assertEqual(env.action_masks().tolist(), [True, True, False, True])
        obs, reward, done, _, info = env.step(3)
        self.assertTrue(done)
        self.assertAlmostEqual(reward, 5.6, places=5)
        self.assertEqual(info["actions"], [2, 1, 3])
        self.assertEqual(info["prediction"], 0)
        self.assertTrue(env.observation_space.contains(obs))

    def test_exhaustion_does_not_fabricate_offset(self):
        env = self.make_env(False)
        env.reset()
        done = False
        while not done:
            obs, reward, done, _, info = env.step(0)
            self.assertEqual(reward, 0)
        self.assertFalse(info["actual_offset"])
        self.assertEqual(info["prediction"], -1)
        self.assertEqual(len(info["actions"]), 31)
        self.assertTrue(env.observation_space.contains(obs))

    def test_weighted_aggregation(self):
        self.assertEqual(
            aggregate([(np.array([0.8, 0.2]), 2), (np.array([0.1, 0.9]), 3)]), 0
        )
        self.assertEqual(aggregate([]), -1)

    def test_stream_never_crosses_trials(self):
        trials = np.stack([np.zeros((2, 40)), np.ones((2, 40))]).astype(np.float32)
        stream = concatenate_trials(trials)
        windows = list(stream.windows(10, 1))
        self.assertEqual(len(windows), 62)
        for trial, start, window in windows:
            self.assertTrue(np.all(window == trial))
            self.assertLessEqual(start + 10, stream.boundaries[trial, 1])


if __name__ == "__main__":
    unittest.main()
