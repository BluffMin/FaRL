# Reproduction checklist

- [ ] `.venv_rl` imports and dependency versions match the manifest
- [ ] MuJoCo 2.1.1 path, OSMesa, GLFW, and LD_LIBRARY_PATH are set
- [ ] Environment reset/step smoke passes
- [ ] Observation is float32, shape `(42,)`, ordered object then proprio
- [ ] Action is 7-D Box[-1,1] and positive gripper closes
- [ ] Success is cube z > table z + 0.04 m
- [ ] Fixed success does not terminate; step 200 truncates
- [ ] Scripted controller still solves Lift
- [ ] SAC config exactly matches `final_sac_config.json`
- [ ] Fresh parent uses fixed horizon from step 0
- [ ] Parent `last_model.zip` and `replay_buffer.pkl` are both present
- [ ] Continuation restores model, optimizers, entropy, timestep, replay
- [ ] tqdm and CSV metrics agree
- [ ] Final deterministic seeds are 900000–900099
- [ ] Best and last both receive 100-episode evaluation
- [ ] Checkpoint SHA-256 values match
- [ ] Policy actions are recomputed from the current observation
