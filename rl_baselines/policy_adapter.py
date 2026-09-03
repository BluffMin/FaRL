"""Minimal failure-classifier-free SB3 policy adapter."""

from __future__ import annotations
from stable_baselines3 import SAC, PPO


class VanillaRLPolicy:
    def __init__(self, path, algorithm="SAC", device="cpu"):
        self.model = (SAC if algorithm.upper() == "SAC" else PPO).load(path, device=device)

    def reset(self):
        return None

    def act(self, obs, deterministic=True):
        return self.model.predict(obs, deterministic=deterministic)[0]
