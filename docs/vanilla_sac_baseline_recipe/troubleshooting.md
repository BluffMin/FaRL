# Troubleshooting

- **Return and grasp rise, lift stays zero:** inspect fixed-horizon semantics, P(lift|grasp), and post-grasp z action. Do not tune the network first.
- **Scripted controller fails:** treat this as an environment/controller/runtime problem.
- **Scripted succeeds but SAC never reaches:** audit observation ordering, action scaling/sign, reward, and resets.
- **SAC grasps under success termination but will not lift:** verify a one-variable fixed-horizon run before entropy or architecture changes.
- **Fixed horizon succeeds:** do not reintroduce reward shaping without a separate causal need.
- **tqdm shows completed immediately:** inspect `run_status.json` and requested total; ensure a new run name.
- **Concurrent-run refusal:** inspect `active_run.json` and whether its PID exists; never bypass a genuinely running job.
- **Continuation performs unlike parent:** confirm replay buffer was loaded and `reset_num_timesteps=False`; model-only loading is warm-start.
- **MuJoCo import/GL error:** export the exact variables in `environment_manifest.json`; use the project launcher.
