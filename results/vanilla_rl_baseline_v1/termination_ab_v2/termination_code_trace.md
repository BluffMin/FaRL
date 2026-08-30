# Termination code trace

`rl_baselines/envs.py:35` ignores robosuite done, recomputes success, sets `terminated = success and termination_mode == "success"`, and sets the horizon as `truncated`. SB3 `DummyVecEnv` converts the five-tuple to done plus `TimeLimit.truncated`. Installed `ReplayBuffer` stores timeout separately and samples `done * (1-timeout)`. SAC uses `(1-done)` in its target.
