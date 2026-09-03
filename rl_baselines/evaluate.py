from __future__ import annotations
import numpy as np


def evaluate_policy_success(model, env, n_episodes=20, seeds=None):
    returns = []
    lengths = []
    success = []
    episodes = []
    rng = np.random.get_state()
    try:
        for ep in range(n_episodes):
            obs, _ = env.reset(seed=(seeds[ep] if seeds else 100000 + ep))
            ret = 0.0
            ever = False
            for t in range(env.horizon):
                action, _ = model.predict(obs, deterministic=True)
                obs, r, term, trunc, info = env.step(action)
                ret += r
                ever |= bool(info["is_success"])
                if term or trunc:
                    break
            returns.append(ret)
            lengths.append(t + 1)
            success.append(ever)
            episodes.append(info.get("episode_diagnostics", {}))
    finally:
        np.random.set_state(rng)

    def mean(k):
        x = [float(e[k]) for e in episodes if k in e]
        return float(np.mean(x)) if x else None

    lift_rate = float(np.mean([float(e.get("object_lift_delta", 0)) >= 0.004 for e in episodes]))
    grasp_count = sum(bool(e.get("ever_grasped")) for e in episodes)
    p_lift_given_grasp = sum(
        float(e.get("object_lift_delta", 0)) >= 0.004 for e in episodes if e.get("ever_grasped")
    ) / max(1, grasp_count)
    median_lift = float(np.median([float(e.get("object_lift_delta", 0)) for e in episodes]))
    return {
        "episodes": n_episodes,
        "success_rate": float(np.mean(success)),
        "final_success_rate": float(
            np.mean([bool(e.get("final_success", False)) for e in episodes])
        ),
        "mean_return": float(np.mean(returns)),
        "median_return": float(np.median(returns)),
        "mean_return_per_step": mean("return_per_step"),
        "mean_episode_length": float(np.mean(lengths)),
        "grasp_rate": mean("ever_grasped"),
        "lift_rate": lift_rate,
        "p_lift_given_grasp": float(p_lift_given_grasp),
        "median_object_lift_delta": median_lift,
        "mean_grasp_duration": mean("grasp_duration_steps"),
        "mean_max_object_height": mean("max_object_height"),
        "mean_object_lift_delta": mean("object_lift_delta"),
        "mean_action_norm": mean("mean_action_norm"),
        "mean_action_saturation_fraction": mean("action_saturation_fraction"),
        "returns": returns,
        "lengths": lengths,
        "episode_diagnostics": episodes,
    }
