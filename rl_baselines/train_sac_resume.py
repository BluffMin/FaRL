#!/usr/bin/env python3
"""True SB3 SAC continuation from a model and replay-buffer checkpoint."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np, torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CallbackList
from rl_baselines.callbacks import SuccessEvalCallback, LiveMetricsCallback
from rl_baselines.envs import make_nominal_env
from rl_baselines.evaluate import evaluate_policy_success
from rl_baselines.progress import VanillaSACProgressCallback


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent-model", required=True)
    ap.add_argument("--parent-replay", required=True)
    ap.add_argument("--parent-step", type=int, default=100000)
    ap.add_argument("--target-total-steps", type=int, default=300000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--termination-mode", choices=["success", "fixed_horizon"], default="success")
    args = ap.parse_args()
    if args.target_total_steps <= args.parent_step:
        raise SystemExit("target must exceed parent step")
    out = Path(args.out)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    train = make_nominal_env(args.seed, termination_mode=args.termination_mode)
    ev = make_nominal_env(10000 + args.seed, termination_mode=args.termination_mode)
    model = SAC.load(args.parent_model, env=train, device="cpu")
    model.load_replay_buffer(args.parent_replay)
    if model.num_timesteps != args.parent_step:
        raise RuntimeError(f"model step {model.num_timesteps} != parent step {args.parent_step}")
    if model.replay_buffer.size() < args.parent_step:
        raise RuntimeError(f"replay buffer too small: {model.replay_buffer.size()}")
    live = LiveMetricsCallback(out, 1000, print_live=False)
    checkpoints = [120000, 140000, 160000, 180000, 200000, 225000, 250000, 275000, 300000]
    evcb = SuccessEvalCallback(ev, out, 10000, 20, 50000, verbose=0, checkpoint_steps=checkpoints)
    progress = VanillaSACProgressCallback(live, args.parent_step, args.target_total_steps, 1000)
    before_updates = int(getattr(model, "_n_updates", 0))
    started = time.time()
    additional = args.target_total_steps - args.parent_step
    model.learn(
        total_timesteps=additional,
        callback=CallbackList([evcb, live, progress]),
        reset_num_timesteps=False,
        log_interval=10,
        progress_bar=False,
    )
    elapsed = time.time() - started
    model.save(out / "checkpoints" / "latest_model")
    model.save_replay_buffer(out / "checkpoints" / "replay_buffer.pkl")
    seeds = [900000 + i for i in range(100)]
    last = evaluate_policy_success(model, ev, 100, seeds)
    best = SAC.load(out / "checkpoints" / "best_model.zip", env=ev, device="cpu")
    best_eval = evaluate_policy_success(best, ev, 100, seeds)
    comparison = {
        "episodes_per_model": 100,
        "paired_seeds": seeds,
        "best_model": best_eval,
        "last_model": last,
    }
    (out / "final_model_comparison.json").write_text(json.dumps(comparison, indent=2))
    summary = {
        "resume_mode": "TRUE_CONTINUATION",
        "termination_mode": args.termination_mode,
        "parent_step": args.parent_step,
        "target_total_steps": args.target_total_steps,
        "additional_steps": additional,
        "final_num_timesteps": model.num_timesteps,
        "elapsed_seconds": elapsed,
        "steps_per_second": additional / elapsed,
        "replay_buffer_size": model.replay_buffer.size(),
        "gradient_updates_added": int(getattr(model, "_n_updates", 0)) - before_updates,
        "entropy_coefficient": float(model.log_ent_coef.detach().exp()),
        "best_model_evaluation": best_eval,
        "last_model_evaluation": last,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    train.close()
    ev.close()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
