"""Nominal robosuite Lift construction and Gymnasium compatibility."""

from __future__ import annotations
import os

os.environ.setdefault("PYGLFW_LIBRARY", "/workspace/collectenv/lib/libglfw.so.3")
os.environ.setdefault("MUJOCO_PY_MUJOCO_PATH", "/workspace/mujoco-2.1.1")
os.environ.setdefault("MUJOCO_GL", "osmesa")
import gymnasium as gym
import numpy as np
import robosuite as suite

SOURCE_CONTROLLER = {
    "type": "OSC_POSE",
    "input_max": 1,
    "input_min": -1,
    "output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
    "output_min": [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
    "kp": 150,
    "damping": 1,
    "impedance_mode": "fixed",
    "kp_limits": [0, 300],
    "damping_limits": [0, 10],
    "position_limits": None,
    "orientation_limits": None,
    "uncouple_pos_ori": True,
    "control_delta": True,
    "interpolation": None,
    "ramp_ratio": 0.2,
}
OBS_KEYS = ("object-state", "robot0_proprio-state")


class LiftGymnasium(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        seed=0,
        horizon=200,
        reward_shaping=True,
        termination_mode="success",
        reward_mode="control",
        lambda_lift=1.0,
    ):
        super().__init__()
        self.horizon = int(horizon)
        self._step = 0
        self._seed = int(seed)
        if termination_mode not in ("success", "fixed_horizon"):
            raise ValueError(f"invalid termination_mode={termination_mode!r}")
        self.termination_mode = termination_mode
        if reward_mode not in ("control", "signed_lift_progress"):
            raise ValueError(f"invalid reward_mode={reward_mode!r}")
        self.reward_mode = reward_mode
        self.lambda_lift = float(lambda_lift)
        self.env = suite.make(
            env_name="Lift",
            robots=["Panda"],
            controller_configs=dict(SOURCE_CONTROLLER),
            has_renderer=False,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            use_object_obs=True,
            reward_shaping=reward_shaping,
            reward_scale=1.0,
            control_freq=20,
            horizon=horizon,
            ignore_done=True,
            hard_reset=True,
        )
        low, high = self.env.action_spec
        self.action_space = gym.spaces.Box(
            np.asarray(low, np.float32), np.asarray(high, np.float32), dtype=np.float32
        )
        obs = self.env.reset()
        self.key_dims = {k: int(np.asarray(obs[k]).size) for k in OBS_KEYS}
        n = sum(self.key_dims.values())
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (n,), np.float32)

    def flatten(self, obs):
        return np.concatenate([np.asarray(obs[k], np.float32).ravel() for k in OBS_KEYS]).astype(
            np.float32, copy=False
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step = 0
        if seed is not None:
            np.random.seed(seed)
        obs = self.env.reset()
        self._diag = {
            "episode_return": 0.0,
            "ever_success": False,
            "previous_success": False,
            "success_count": 0,
            "success_duration_steps": 0,
            "ever_grasped": False,
            "first_grasp_step": None,
            "grasp_count": 0,
            "grasp_duration_steps": 0,
            "height_after_grasp_sum": 0.0,
            "height_after_grasp_count": 0,
            "max_height_after_grasp": None,
            "post_grasp_z_action_sum": 0.0,
            "post_grasp_positive_z_count": 0,
            "post_grasp_action_count": 0,
            "max_object_height": float(self.env.sim.data.body_xpos[self.env.cube_body_id][2]),
            "initial_object_height": float(self.env.sim.data.body_xpos[self.env.cube_body_id][2]),
            "min_eef_object_distance": float("inf"),
            "success_step": None,
            "first_measurable_lift_step": None,
            "action_norm_sum": 0.0,
            "max_action_norm": 0.0,
            "action_saturated": 0,
            "action_sum": np.zeros(self.action_space.shape[0]),
            "action_abs_sum": np.zeros(self.action_space.shape[0]),
            "action_sq_sum": np.zeros(self.action_space.shape[0]),
            "action_sat_dim": np.zeros(self.action_space.shape[0]),
            "reaching_return": 0.0,
            "grasp_return": 0.0,
            "success_return": 0.0,
        }
        self._diag.update(base_return=0.0, lift_progress_return=0.0)
        return self.flatten(obs), {"is_success": bool(self.env._check_success())}

    def step(self, action):
        raw = np.asarray(action, np.float32)
        a = np.clip(raw, self.action_space.low, self.action_space.high)
        z_before = float(self.env.sim.data.body_xpos[self.env.cube_body_id][2])
        grasp_before = bool(
            self.env._check_grasp(gripper=self.env.robots[0].gripper, object_geoms=self.env.cube)
        )
        z0 = self._diag["initial_object_height"]
        z_success = float(self.env.table_offset[2] + 0.04)
        p_before = float(np.clip((z_before - z0) / max(z_success - z0, 1e-8), 0, 1))
        obs, base_reward, _, info = self.env.step(a)
        self._step += 1
        success = bool(self.env._check_success())
        terminated = success and self.termination_mode == "success"
        truncated = self._step >= self.horizon and not terminated
        height = float(self.env.sim.data.body_xpos[self.env.cube_body_id][2])
        eef = np.asarray(self.env.sim.data.site_xpos[self.env.robots[0].eef_site_id])
        obj = np.asarray(self.env.sim.data.body_xpos[self.env.cube_body_id])
        dist = float(np.linalg.norm(eef - obj))
        grasp = bool(
            self.env._check_grasp(gripper=self.env.robots[0].gripper, object_geoms=self.env.cube)
        )
        reach = (1 - np.tanh(10 * dist)) / 2.25 if not success else 0.0
        grasp_r = 0.25 / 2.25 if grasp and not success else 0.0
        success_r = 1.0 if success else 0.0
        p_after = float(np.clip((height - z0) / max(z_success - z0, 1e-8), 0, 1))
        progress_reward = (
            self.lambda_lift * (p_after - p_before)
            if self.reward_mode == "signed_lift_progress" and (grasp_before or grasp)
            else 0.0
        )
        reward = float(base_reward) + progress_reward
        self._diag["base_return"] += float(base_reward)
        self._diag["lift_progress_return"] += float(progress_reward)
        d = self._diag
        d["episode_return"] += float(reward)
        d["ever_success"] |= success
        d["success_count"] += int(success and not d["previous_success"])
        d["success_duration_steps"] += int(success)
        d["previous_success"] = success
        d["ever_grasped"] |= grasp
        d["first_grasp_step"] = (
            self._step if grasp and d["first_grasp_step"] is None else d["first_grasp_step"]
        )
        d["grasp_count"] += int(grasp)
        d["grasp_duration_steps"] += int(grasp)
        d["max_object_height"] = max(d["max_object_height"], height)
        d["min_eef_object_distance"] = min(d["min_eef_object_distance"], dist)
        d["success_step"] = (
            self._step if success and d["success_step"] is None else d["success_step"]
        )
        d["first_measurable_lift_step"] = (
            self._step
            if height >= d["initial_object_height"] + 0.004
            and d["first_measurable_lift_step"] is None
            else d["first_measurable_lift_step"]
        )
        if grasp:
            d["height_after_grasp_sum"] += height
            d["height_after_grasp_count"] += 1
            d["max_height_after_grasp"] = (
                height
                if d["max_height_after_grasp"] is None
                else max(d["max_height_after_grasp"], height)
            )
            d["post_grasp_z_action_sum"] += float(a[2])
            d["post_grasp_positive_z_count"] += int(a[2] > 0)
            d["post_grasp_action_count"] += 1
        norm = float(np.linalg.norm(a))
        d["action_norm_sum"] += norm
        d["max_action_norm"] = max(d["max_action_norm"], norm)
        d["action_saturated"] += int(np.any(np.abs(a) >= 1 - 1e-6))
        d["action_sum"] += a
        d["action_abs_sum"] += np.abs(a)
        d["action_sq_sum"] += a * a
        d["action_sat_dim"] += np.abs(a) >= 1 - 1e-6
        d["reaching_return"] += reach
        d["grasp_return"] += grasp_r
        d["success_return"] += success_r
        info = dict(info)
        info.update(
            is_success=success,
            grasped=grasp,
            action_clipped=bool(np.any(a != raw)),
            action_saturated=bool(np.any(np.abs(a) >= 1 - 1e-6)),
            action_norm=norm,
            object_height=height,
            object_lift_delta=d["max_object_height"] - d["initial_object_height"],
            eef_object_distance=dist,
            reward_reaching=reach,
            reward_grasp=grasp_r,
            reward_success=success_r,
        )
        info.update(
            base_reward=float(base_reward),
            lift_progress_reward=float(progress_reward),
            total_reward=float(reward),
            lift_progress_before=p_before,
            lift_progress_after=p_after,
        )
        if terminated or truncated:
            n = self._step
            mean = d["action_sum"] / n
            var = np.maximum(0, d["action_sq_sum"] / n - mean * mean)
            info["episode_diagnostics"] = {
                "episode_return": d["episode_return"],
                "episode_length": n,
                "return_per_step": d["episode_return"] / n,
                "final_success": success,
                "ever_success": d["ever_success"],
                "success_count": d["success_count"],
                "success_duration_steps": d["success_duration_steps"],
                "max_object_height": d["max_object_height"],
                "final_object_height": height,
                "initial_object_height": d["initial_object_height"],
                "object_lift_delta": d["max_object_height"] - d["initial_object_height"],
                "ever_grasped": d["ever_grasped"],
                "first_grasp_step": d["first_grasp_step"],
                "grasp_count": d["grasp_count"],
                "grasp_duration_steps": d["grasp_duration_steps"],
                "mean_object_height_after_grasp": (
                    d["height_after_grasp_sum"] / d["height_after_grasp_count"]
                    if d["height_after_grasp_count"]
                    else None
                ),
                "max_object_height_after_grasp": d["max_height_after_grasp"],
                "post_grasp_z_action_mean": (
                    d["post_grasp_z_action_sum"] / d["post_grasp_action_count"]
                    if d["post_grasp_action_count"]
                    else None
                ),
                "post_grasp_positive_z_action_fraction": (
                    d["post_grasp_positive_z_count"] / d["post_grasp_action_count"]
                    if d["post_grasp_action_count"]
                    else None
                ),
                "min_eef_object_distance": d["min_eef_object_distance"],
                "success_step": d["success_step"],
                "first_measurable_lift_step": d["first_measurable_lift_step"],
                "mean_action_norm": d["action_norm_sum"] / n,
                "max_action_norm": d["max_action_norm"],
                "action_saturation_fraction": d["action_saturated"] / n,
                "action_mean_per_dimension": mean.tolist(),
                "action_std_per_dimension": np.sqrt(var).tolist(),
                "action_abs_mean_per_dimension": (d["action_abs_sum"] / n).tolist(),
                "action_saturation_per_dimension": (d["action_sat_dim"] / n).tolist(),
                "translation_abs_mean": float(np.mean(d["action_abs_sum"][:3] / n)),
                "rotation_abs_mean": float(np.mean(d["action_abs_sum"][3:6] / n)),
                "gripper_mean": float(mean[6]),
                "gripper_abs_mean": float(d["action_abs_sum"][6] / n),
                "cumulative_reaching_reward": d["reaching_return"],
                "cumulative_grasp_reward": d["grasp_return"],
                "cumulative_success_reward": d["success_return"],
            }
            info["episode_diagnostics"].update(
                base_return=d["base_return"],
                lift_progress_return=d["lift_progress_return"],
                mean_base_reward=d["base_return"] / n,
                mean_lift_progress_reward=d["lift_progress_return"] / n,
            )
        return self.flatten(obs), float(reward), terminated, truncated, info

    def close(self):
        self.env.close()


def make_nominal_env(
    seed=0, horizon=200, termination_mode="success", reward_mode="control", lambda_lift=1.0
):
    return LiftGymnasium(
        seed=seed,
        horizon=horizon,
        reward_shaping=True,
        termination_mode=termination_mode,
        reward_mode=reward_mode,
        lambda_lift=lambda_lift,
    )
