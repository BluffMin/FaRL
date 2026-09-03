#!/usr/bin/env python3
"""Attachable tqdm display for a managed SAC run without touching training."""
from __future__ import annotations
import argparse, csv, json, time
from pathlib import Path
from tqdm.auto import tqdm

BASE = Path("/home/brainlab/FaRL/results/vanilla_rl_baseline_v1")


def last_csv(path):
    try:
        with path.open() as f:
            rows = list(csv.DictReader(f))
        return rows[-1] if rows else {}
    except (FileNotFoundError, OSError, csv.Error):
        return {}


def alive(pid):
    return bool(pid) and Path(f"/proc/{pid}").exists()


def val(x, fmt=".3g"):
    if x in (None, ""):
        return "NA"
    try:
        return format(float(x), fmt)
    except (TypeError, ValueError):
        return str(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_name", nargs="?")
    ap.add_argument("--interval", type=float, default=2.0)
    args = ap.parse_args()
    active_path = BASE / "active_run.json"
    active = json.loads(active_path.read_text())
    name = args.run_name or active["run_name"]
    run = BASE / name
    status = json.loads((run / "run_status.json").read_text())
    total = int(status.get("target_total_steps", status.get("steps_requested", 100000)))
    initial = last_csv(run / "live_metrics.csv")
    last_step = int(float(initial.get("step", 0) or 0))
    bar = tqdm(total=total, initial=last_step, unit="step", dynamic_ncols=True, desc=name)
    try:
        while True:
            live = last_csv(run / "live_metrics.csv")
            ev = last_csv(run / "learning_curve.csv")
            step = int(float(live.get("step", last_step) or last_step))
            bar.update(max(0, step - last_step))
            last_step = max(last_step, step)
            bar.set_postfix(
                {
                    "ever": val(
                        live.get("rolling_ever_success_rate", live.get("rolling_success_rate"))
                    ),
                    "final": val(live.get("rolling_final_success_rate")),
                    "eval_succ": val(ev.get("success_rate")),
                    "grasp": val(live.get("rolling_grasp_rate")),
                    "lift_mm": val(
                        1000 * float(live["rolling_object_lift_delta"])
                        if live.get("rolling_object_lift_delta")
                        else None
                    ),
                    "lift": val(live.get("rolling_lift_rate")),
                    "lift|grasp": val(live.get("rolling_p_lift_given_grasp")),
                    "first_s": val(live.get("rolling_first_success_step")),
                    "return": val(live.get("rolling_return")),
                    "actor": val(live.get("actor_loss")),
                    "critic": val(live.get("critic_loss")),
                    "alpha": val(live.get("entropy_coefficient")),
                }
            )
            bar.refresh()
            status = json.loads((run / "run_status.json").read_text())
            if status.get("status") not in ("RUNNING", "STARTING") or not alive(
                status.get("training_pid")
            ):
                break
            time.sleep(max(0.5, args.interval))
    finally:
        bar.n = last_step
        bar.refresh()
        bar.close()
    print(f"status={status.get('status')} step={last_step}/{total} run={name}")


if __name__ == "__main__":
    main()
