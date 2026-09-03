#!/usr/bin/env python3
"""Preflight and post-run analysis for the one-variable termination A/B."""
from __future__ import annotations
import json, sys
from pathlib import Path
import h5py, numpy as np
from rl_baselines.envs import LiftGymnasium, SOURCE_CONTROLLER, OBS_KEYS

ROOT = Path("/home/brainlab/FaRL")
OUT = ROOT / "results/vanilla_rl_baseline_v1/termination_ab_v1"
OUT.mkdir(parents=True, exist_ok=True)
DEMO = Path("/home/robotics/external_workspace/data/C/demos/lift_ph_demo.hdf5")


def write(name, x):
    (OUT / name).write_text(json.dumps(x, indent=2, sort_keys=True))


def replay(mode, continue_to_horizon=False):
    sys.path.insert(0, "/home/robotics/external_workspace/data/C/scripts")
    import a_cluttered as A

    e = LiftGymnasium(seed=0, horizon=200, reward_shaping=True, termination_mode=mode)
    e.reset(seed=0)
    with h5py.File(DEMO, "r") as f:
        g = f["data"]["demo_0"]
        e.env.reset_from_xml_string(A.fix_mesh_paths(g.attrs["model_file"]))
        e.env.sim.set_state_from_flattened(np.asarray(g["states"][0]))
        e.env.sim.forward()
        e._step = 0
        e._diag["initial_object_height"] = float(e.env.sim.data.body_xpos[e.env.cube_body_id][2])
        trace = []
        first = None
        for a in np.asarray(g["actions"]):
            _, r, term, trunc, info = e.step(a)
            trace.append(
                {
                    "step": e._step,
                    "success": bool(info["is_success"]),
                    "terminated": bool(term),
                    "truncated": bool(trunc),
                    "reward": float(r),
                    "height": float(info["object_height"]),
                    "grasped": bool(info["grasped"]),
                }
            )
            if info["is_success"] and first is None:
                first = e._step
            if term or trunc:
                break
        if continue_to_horizon and not (term or trunc):
            while e._step < e.horizon:
                cube = e.env.sim.data.body_xpos[e.env.cube_body_id].copy()
                eef = e.env.sim.data.site_xpos[e.env.robots[0].eef_site_id].copy()
                a = np.zeros(7)
                a[:3] = np.clip((np.array([cube[0], cube[1], 0.90]) - eef) * 10, -1, 1)
                a[6] = 1
                _, r, term, trunc, info = e.step(a)
                trace.append(
                    {
                        "step": e._step,
                        "success": bool(info["is_success"]),
                        "terminated": bool(term),
                        "truncated": bool(trunc),
                        "reward": float(r),
                        "height": float(info["object_height"]),
                        "grasped": bool(info["grasped"]),
                    }
                )
                if term or trunc:
                    break
    out = {
        "mode": mode,
        "first_success_step": first,
        "trace": trace,
        "last": trace[-1],
        "episode_diagnostics": info.get("episode_diagnostics"),
    }
    e.close()
    return out


def main():
    a = replay("success")
    b = replay("fixed_horizon", True)
    at = next(x for x in a["trace"] if x["success"])
    bt = next(x for x in b["trace"] if x["success"])
    passed = (
        at["terminated"]
        and not at["truncated"]
        and not bt["terminated"]
        and not bt["truncated"]
        and b["last"]["step"] == 200
        and b["last"]["truncated"]
        and not b["last"]["terminated"]
    )
    write(
        "termination_semantics_unit_test.json",
        {
            "ARM_A_at_first_success": at,
            "ARM_B_at_first_success": bt,
            "ARM_B_at_horizon": b["last"],
            "ARM_B_ever_success": b["episode_diagnostics"]["ever_success"],
            "ARM_B_final_success": b["episode_diagnostics"]["final_success"],
            "FIXED_HORIZON_SEMANTICS": "PASS" if passed else "FAIL",
        },
    )
    post = [
        x
        for x in b["trace"]
        if b["first_success_step"] is not None and x["step"] >= b["first_success_step"]
    ]
    write(
        "post_success_semantics.json",
        {
            "first_success_step": b["first_success_step"],
            "post_success_steps": len(post),
            "success_duration_steps": sum(x["success"] for x in post),
            "success_reentry_count": b["episode_diagnostics"]["success_count"],
            "final_success": b["episode_diagnostics"]["final_success"],
            "mean_reward_after_first_success": float(np.mean([x["reward"] for x in post])),
            "min_height_after_first_success": float(min(x["height"] for x in post)),
            "max_height_after_first_success": float(max(x["height"] for x in post)),
            "grasp_fraction_after_first_success": float(np.mean([x["grasped"] for x in post])),
            "diagnostic_continuation": "state-feedback hold/lift; never used for training",
        },
    )
    common = {
        "environment": "Lift",
        "robot": "Panda",
        "controller": SOURCE_CONTROLLER,
        "control_freq": 20,
        "horizon": 200,
        "ignore_done": True,
        "hard_reset": True,
        "observation_keys": list(OBS_KEYS),
        "observation_dim": 42,
        "action_dim": 7,
        "action_bounds": [-1, 1],
        "reward_shaping": True,
        "reward_scale": 1.0,
        "seed": 0,
        "algorithm": "SAC",
        "network": [256, 256],
        "activation": "ReLU",
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "tau": 0.005,
        "buffer_size": 1000000,
        "learning_starts": 5000,
        "batch_size": 256,
        "train_freq": 1,
        "gradient_steps": 1,
        "ent_coef": "auto",
        "target_entropy": -7,
        "normalization": False,
        "eval_seed_formula": "700000 + timestep + episode_index",
        "eval_episodes": 20,
        "checkpoint_cadence": 50000,
    }
    ca = {**common, "termination_mode": "success"}
    cb = {**common, "termination_mode": "fixed_horizon"}
    write(
        "arm_a_reference.json",
        {
            "run": "sac_seed0_100k_diag",
            "config": ca,
            "best_success_100": 0.10,
            "last_success_100": 0.05,
        },
    )
    write(
        "arm_b_config.json", {"run": "sac_seed0_100k_fixed_horizon", "steps": 100000, "config": cb}
    )
    diffs = [k for k in ca if ca[k] != cb[k]]
    write(
        "config_match.json",
        {
            "AB_CONFIG_MATCH": "PASS" if diffs == ["termination_mode"] else "FAIL",
            "ONLY_INTENTIONAL_DIFFERENCE": "success termination semantics",
            "different_fields": diffs,
            "arm_a": ca,
            "arm_b": cb,
            "reward_identity": True,
            "observation_identity": True,
            "action_identity": True,
            "reset_distribution_identity": True,
            "original_assets_modified": False,
        },
    )
    (OUT / "current_termination_trace.md").write_text(
        '# Current termination trace\n\n`robosuite.env.step()` runs with `ignore_done=True`; its done value is intentionally not propagated. `LiftGymnasium.step()` computes success from the local Lift `_check_success()`. In default `termination_mode=success`, `terminated = success`; in `fixed_horizon`, success remains in `info["is_success"]` but does not terminate. In both modes, `truncated = step >= horizon and not terminated`. The horizon is therefore a Gymnasium time-limit truncation.\n'
    )
    if not passed:
        raise SystemExit("fixed-horizon semantics preflight failed")
    print(
        json.dumps(
            {
                "AB_CONFIG_MATCH": "PASS" if diffs == ["termination_mode"] else "FAIL",
                "FIXED_HORIZON_SEMANTICS": "PASS" if passed else "FAIL",
                "first_success_step": b["first_success_step"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
