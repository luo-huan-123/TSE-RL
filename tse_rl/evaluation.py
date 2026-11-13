"""Deterministic masked-policy evaluation on every supplied trial."""

import numpy as np


def evaluate_policy(model, env, labels):
    rows = []
    for trial in range(len(env.data)):
        obs, _ = env.reset(options={"trial_id": trial})
        done = False
        while not done:
            action, _ = model.predict(
                obs, action_masks=env.action_masks(), deterministic=True
            )
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        rows.append(dict(trial=trial, target=int(labels[trial]), **info))
    correct = [r["prediction"] == r["target"] for r in rows]
    return dict(
        accuracy=100 * float(np.mean(correct)),
        trials=len(rows),
        mean_kept_windows=float(np.mean([r["kept_windows"] for r in rows])),
        mean_decision_time_s=float(np.mean([r["decision_time"] for r in rows])),
        actual_offset_trials=sum(r["actual_offset"] for r in rows),
        predictions=rows,
    )
