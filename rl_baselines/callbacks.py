from __future__ import annotations
import csv, json
from collections import deque
import numpy as np, torch
from pathlib import Path
from stable_baselines3.common.callbacks import BaseCallback
from rl_baselines.evaluate import evaluate_policy_success


class SuccessEvalCallback(BaseCallback):
    def __init__(
        self,
        eval_env,
        out_dir,
        eval_freq=10000,
        n_eval=20,
        checkpoint_freq=50000,
        verbose=1,
        checkpoint_steps=None,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.out = Path(out_dir)
        self.freq = eval_freq
        self.n_eval = n_eval
        self.ckpt = checkpoint_freq
        self.checkpoint_steps = set(checkpoint_steps or [])
        self.rows = []
        self.best = (-1.0, -1.0, float("-inf"), -1.0, float("-inf"))

    def _on_step(self):
        if self.num_timesteps % self.ckpt == 0 or self.num_timesteps in self.checkpoint_steps:
            self.model.save(self.out / "checkpoints" / f"model_{self.num_timesteps}")
            self.model.save(self.out / "checkpoints" / "latest_model")
            self.model.save_replay_buffer(self.out / "checkpoints" / "replay_buffer.pkl")
        if self.num_timesteps % self.freq == 0:
            m = evaluate_policy_success(
                self.model,
                self.eval_env,
                self.n_eval,
                [700000 + self.num_timesteps + i for i in range(self.n_eval)],
            )
            row = {
                "step": self.num_timesteps,
                **{
                    k: v
                    for k, v in m.items()
                    if k not in ("returns", "lengths", "episode_diagnostics")
                },
            }
            self.rows.append(row)
            with (self.out / "learning_curve.csv").open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(row))
                w.writeheader()
                w.writerows(self.rows)
            score = (
                m["success_rate"],
                m["lift_rate"],
                m.get("median_object_lift_delta") or -1.0,
                m["grasp_rate"] or -1.0,
                m["mean_return"],
            )
            if score > self.best:
                self.best = score
                self.model.save(self.out / "checkpoints" / "best_model")
            if self.verbose:
                print("EVAL", json.dumps(row))
        return True

    def _on_training_end(self):
        self.model.save(self.out / "checkpoints" / "last_model")
        self.model.save_replay_buffer(self.out / "checkpoints" / "replay_buffer.pkl")


class LiveMetricsCallback(BaseCallback):
    """Print and persist compact rolling metrics while tqdm is active."""

    def __init__(self, out_dir, print_freq=1000, print_live=True):
        super().__init__(0)
        self.out = Path(out_dir)
        self.freq = int(print_freq)
        self.print_live = print_live
        self.rows = []
        self.episodes = deque(maxlen=100)
        self.all_episodes = []

    def _on_step(self):
        for info, done in zip(self.locals.get("infos", []), self.locals.get("dones", [])):
            if done and info.get("episode_diagnostics"):
                e = dict(info["episode_diagnostics"])
                e["end_step"] = self.num_timesteps
                self.episodes.append(e)
                self.all_episodes.append(e)
                with (self.out / "episode_metrics.csv").open("w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(e))
                    w.writeheader()
                    w.writerows(self.all_episodes)
        if self.num_timesteps % self.freq:
            return True
        values = self.model.logger.name_to_value
        ep = list(self.model.ep_info_buffer)
        succ = list(self.model.ep_success_buffer)
        diag = list(self.episodes)

        def finite_value(name):
            x = values.get(name)
            return None if x is None else float(x)

        def avg(name):
            return float(np.mean([x[name] for x in diag])) if diag else None

        def avg_defined(name):
            values = [x.get(name) for x in diag if x.get(name) is not None]
            return float(np.mean(values)) if values else None

        policy_std = None
        try:
            obs = torch.as_tensor(self.locals["new_obs"], device=self.model.device)
            _, log_std, _ = self.model.actor.get_action_dist_params(obs)
            policy_std = float(log_std.exp().mean().detach().cpu())
        except Exception:
            pass
        row = {
            "step": self.num_timesteps,
            "rolling_episodes": len(ep),
            "rolling_return": float(sum(float(x["r"]) for x in ep) / len(ep)) if ep else None,
            "rolling_success_rate": float(sum(bool(x) for x in succ) / len(succ)) if succ else None,
            "rolling_episode_length": avg("episode_length"),
            "rolling_return_per_step": avg("return_per_step"),
            "rolling_base_reward": avg("mean_base_reward"),
            "rolling_lift_progress_reward": avg("mean_lift_progress_reward"),
            "rolling_max_object_height": avg("max_object_height"),
            "rolling_object_lift_delta": avg("object_lift_delta"),
            "rolling_grasp_rate": avg("ever_grasped"),
            "rolling_lift_rate": (
                float(np.mean([x["object_lift_delta"] >= 0.004 for x in diag])) if diag else None
            ),
            "rolling_p_lift_given_grasp": (
                float(
                    sum(x["object_lift_delta"] >= 0.004 for x in diag if x["ever_grasped"])
                    / max(1, sum(x["ever_grasped"] for x in diag))
                )
                if diag
                else None
            ),
            "rolling_grasp_duration": avg("grasp_duration_steps"),
            "actor_loss": finite_value("train/actor_loss"),
            "critic_loss": finite_value("train/critic_loss"),
            "entropy_coefficient": finite_value("train/ent_coef"),
            "mean_policy_action_std": policy_std,
            "action_norm": avg("mean_action_norm"),
            "action_saturation_fraction": avg("action_saturation_fraction"),
            "updates": finite_value("train/n_updates"),
        }
        row.update(
            rolling_ever_success_rate=avg("ever_success"),
            rolling_final_success_rate=avg("final_success"),
            rolling_first_success_step=avg_defined("success_step"),
        )
        row.update(
            translation_abs_mean=avg("translation_abs_mean"),
            rotation_abs_mean=avg("rotation_abs_mean"),
            gripper_mean=avg("gripper_mean"),
            gripper_abs_mean=avg("gripper_abs_mean"),
        )
        self.rows.append(row)
        with (self.out / "live_metrics.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            w.writeheader()
            w.writerows(self.rows)
        eval_success = None
        curve = self.out / "learning_curve.csv"
        if curve.exists():
            er = list(csv.DictReader(curve.open()))
            eval_success = float(er[-1]["success_rate"]) if er else None
        explore = [
            {
                "step": r["step"],
                "ent_coef": r["entropy_coefficient"],
                "mean_policy_action_std": r["mean_policy_action_std"],
                "action_saturation": r["action_saturation_fraction"],
                "eval_success": eval_success if r is self.rows[-1] else None,
                "grasp_rate": r["rolling_grasp_rate"],
                "max_object_height": r["rolling_max_object_height"],
            }
            for r in self.rows
        ]
        with (self.out / "exploration_diagnostics.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(explore[0]))
            w.writeheader()
            w.writerows(explore)
        pretty = " | ".join(
            f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()
        )
        if self.print_live:
            print(f"\nLIVE {pretty}", flush=True)
        return True
