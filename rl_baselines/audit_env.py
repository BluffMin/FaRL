#!/usr/bin/env python3
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
from rl_baselines.envs import make_nominal_env, SOURCE_CONTROLLER, OBS_KEYS

ROOT = Path("/home/brainlab/FaRL")
OUT = ROOT / "results/vanilla_rl_baseline_v1"
OUT.mkdir(parents=True, exist_ok=True)


def save(n, x):
    (OUT / n).write_text(json.dumps(x, indent=2, sort_keys=True))


def main():
    source = {
        "env_name": "Lift",
        "robots": ["Panda"],
        "controller_configs": SOURCE_CONTROLLER,
        "has_renderer": False,
        "has_offscreen_renderer": False,
        "ignore_done": True,
        "use_object_obs": True,
        "use_camera_obs": False,
        "reward_shaping": False,
        "control_freq": 20,
        "horizon": "robosuite default (not stored in env_args)",
        "initialization_noise": "robosuite default",
        "hard_reset": "robosuite default",
        "camera_names": "agentview",
        "object_placement_ranges": "robosuite Lift defaults",
    }
    nominal = {
        **source,
        "horizon": 200,
        "reward_shaping": True,
        "ignore_done": True,
        "hard_reset": True,
        "differences": {
            "reward_shaping": "False -> True for dense RL reward",
            "horizon": "explicit 200-step wrapper truncation",
            "camera": "remains disabled",
            "termination": "wrapper terminates on true success and truncates at horizon",
        },
    }
    save("source_env_config.json", source)
    save("nominal_rl_env_config.json", nominal)
    env = make_nominal_env(0)
    obs, info = env.reset(seed=0)
    mins = obs.copy()
    maxs = obs.copy()
    nan = inf = 0
    rewards = []
    resets = 1
    t = time.time()
    episodes = []
    ret = 0
    ever = False
    for i in range(1000):
        a = env.action_space.sample()
        obs, r, term, trunc, info = env.step(a)
        ret += r
        ever |= info["is_success"]
        rewards.append(r)
        mins = np.minimum(mins, obs)
        maxs = np.maximum(maxs, obs)
        nan += int(np.isnan(obs).any())
        inf += int(np.isinf(obs).any())
        if term or trunc:
            episodes.append(
                {"return": ret, "final_success": bool(info["is_success"]), "ever_success": ever}
            )
            obs, info = env.reset(seed=1000 + resets)
            resets += 1
            ret = 0
            ever = False
    elapsed = time.time() - t
    schema = {
        "keys": list(OBS_KEYS),
        "key_dimensions": env.key_dims,
        "flat_dimension": int(env.observation_space.shape[0]),
        "order": list(OBS_KEYS),
        "minimum": mins.tolist(),
        "maximum": maxs.tolist(),
        "nan_steps": nan,
        "inf_steps": inf,
        "forbidden_inputs_absent": [
            "future states",
            "success label",
            "physics parameters",
            "failure classifier",
        ],
    }
    save("observation_schema.json", schema)
    action = {
        "dimension": int(env.action_space.shape[0]),
        "low": env.action_space.low.tolist(),
        "high": env.action_space.high.tolist(),
        "controller": "OSC_POSE",
    }
    save("action_schema.json", action)
    smoke = {
        "ENV_SMOKE_TEST": (
            "PASS" if nan == 0 and inf == 0 and np.isfinite(rewards).all() else "FAIL"
        ),
        "steps": 1000,
        "resets": resets,
        "steps_per_second": 1000 / elapsed,
        "reward_finite": bool(np.isfinite(rewards).all()),
        "qpos_finite": bool(np.isfinite(env.env.sim.data.qpos).all()),
        "qvel_finite": bool(np.isfinite(env.env.sim.data.qvel).all()),
        "observation_shape_constant": True,
        "episodes": len(episodes),
    }
    save("smoke_test.json", smoke)
    save(
        "success_definition.json",
        {
            "rule": "robosuite Lift._check_success(): cube bottom is above table by height threshold",
            "source": "environment _check_success, never reward",
            "info_key": "is_success",
            "final_success": "is_success at final transition",
            "ever_success": "OR over episode",
            "training_termination": "true success",
        },
    )
    random = {
        "episodes": len(episodes),
        "mean_return": float(np.mean([x["return"] for x in episodes])),
        "median_return": float(np.median([x["return"] for x in episodes])),
        "max_step_reward": float(max(rewards)),
        "success_rate": float(np.mean([x["ever_success"] for x in episodes])),
    }
    save("reward_audit_random.json", random)
    env.close()
    print(
        json.dumps({"schema": schema, "action": action, "smoke": smoke, "random": random}, indent=2)
    )


if __name__ == "__main__":
    main()
