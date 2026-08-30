# FaRL vanilla SAC baseline recipe

> **Working recipe — status: `SEED0_WORKING_BASELINE = ESTABLISHED`; `MULTISEED_REPLICATION = PENDING`.** Use robosuite Lift, Panda/PandaGripper, OSC_POSE at 20 Hz, 42-D low-dimensional state, SAC MLP 256×256, original shaped Lift reward, fixed-horizon semantics, horizon 200, and 300k effective interactions. Load `/home/brainlab/FaRL/results/vanilla_rl_baseline_v1/sac_seed0_300k_fixed_horizon_from_100k/checkpoints/best_model.zip` for downstream deterministic nominal control. Primary metric: `ever_success`. This is a seed-0 baseline, not a robust multiseed claim.

## 1. Purpose and scope

This policy supplies a nominal closed-loop task policy, a control for failure-aware recovery, feedback continuation after interventions, and a failure generator under A/B/C perturbations. It is **not failure-conditioned** and must not receive classifier output, ground-truth failure labels, hidden physics, or future trajectory data.

## 2. Proven status and the critical provenance correction

The successful chain was **not** the success-termination 100k control followed by a semantic switch. Saved run metadata proves this chain:

`fresh fixed horizon 0→100k` (`sac_seed0_100k_fixed_horizon`) → 100k `last_model.zip` + 100k replay → `fixed horizon 100k→300k` true continuation (`sac_seed0_300k_fixed_horizon_from_100k`).

Thus fixed-horizon semantics were used from step 0 through 300k. The two-stage continuation restored actor, critics/targets, optimizers, entropy state, timestep, and replay. It added 200k transitions and 200k updates. Python/NumPy/environment RNG streams were not separately restored. A monolithic one-stage command was not run, but **fresh fixed horizon from random initialization is validated through the two-stage trajectory**.

## 3. Runtime

Known-good host is aarch64, Python 3.9.23. Training used CPU: torch 2.8.0+cpu, SB3 2.7.1, Gymnasium 1.1.1, NumPy 1.23.3, SciPy 1.9.1, robosuite 1.2.0 and free-mujoco-py 2.1.6 against `/workspace/mujoco-2.1.1`. Robomimic and legacy Gym are not required. `.venv_rl` isolates modern SB3/Gymnasium while exposing historical robosuite dependencies under `/workspace/collectenv`. Required OSMesa/GLFW/MuJoCo variables are exact in `environment_manifest.json`. Low-dimensional SAC is CPU-valid and environment-step limited.

## 4. Environment

`rl_baselines/envs.py` builds Lift with Panda, PandaGripper, OSC_POSE, object observations enabled, cameras disabled, shaped reward scale 1, 20 Hz, horizon 200, `ignore_done=True`, and `hard_reset=True`. Default cube placement is uniform x/y ±0.03 m around table offset `[0,0,0.8]`, z offset 0.01. Controller translation commands map to ±0.05 m and rotation to ±0.5 rad.

The original FaRL collector (`/home/robotics/.../a_cluttered.py`) is multimodal and enables offscreen agent/wrist cameras; it accepts recorded `env_kwargs`. Vanilla RL intentionally uses nominal state-only observations and no rendering. `/home/robotics` and `/workspace/collectenv` remain read-only.

## 5. Observation and action

Observation is float32 `(42,)`: `object-state` indices 0:10 then `robot0_proprio-state` 10:42. It contains cube pose and relative position plus joint, EEF, and gripper state. There is no normalization, clipping, RGB, failure label, or hidden physics. See `observation_schema.json`.

Action is `[dx,dy,dz,dRx,dRy,dRz,gripper]`, Box[-1,1]^7. Positive gripper closes. SAC uses a squashed Gaussian mapped to these bounds; the wrapper clips once. Audits found no double scaling, sign inversion, or bound mismatch.

## 6. Success and reward

Local robosuite `_check_success()` is exactly `cube_height > table_height + 0.04`; here the table is 0.8 m. Grasp, persistence, and low velocity are not required. Report `ever_success` primarily and `final_success` secondarily.

Before success the original shaped reward is `(1-tanh(10*distance) + 0.25*grasp)/2.25`; success replaces this with `2.25/2.25 = 1`. There is no continuous pre-threshold lift term. The signed lift-progress experiment reached best 5%, last 1% and is `NOT_SUPPORTED_WITH_THIS_SHAPING`; it is excluded.

## 7. Termination semantics

Original control turns success into `terminated=True`; horizon is `truncated=True`. Fixed horizon keeps `is_success=True` but returns `terminated=False,truncated=False` until step 200, which truncates. SB3 masks timeout dones and bootstraps through horizons. It also bootstraps through fixed-horizon success but not control terminal success. This matches the general robosuite fixed-horizon convention, with local code authoritative.

## 8. Why termination mattered

At matched 300k best checkpoints and 100 seeds, control had success 0%, grasp 100%, lift 2%, P(lift|grasp) 2%, mean lift delta 0.310 mm. Fixed best had ever-success **92%**, final-success 74%, grasp 93%, lift **92%**, P(lift|grasp) **98.9%**. Fixed last remained 88% success, 74% final-success, 94% grasp, 90% lift. Paired differences were +92 points (95% CI +86 to +97) for success and +90 (+84 to +95) for lift. Verdict: `FIXED_HORIZON_SUPPORTED` for this single-seed causal screen.

Mechanistic observations, not a causal decomposition: reward≈1 replay proxy rose from 0.0163% to 23.88%; post-grasp mean z action shifted −0.865→−0.587 and positive-z incidence increased. Fixed horizon jointly changes bootstrap targets, successful-state replay occupancy, and repeated post-success reward.

## 9. SAC and numerical health

SAC `MlpPolicy`, actor/critic 256×256 ReLU, lr 3e-4, replay 1e6, learning starts 5k, batch 256, gamma .99, tau .005, train/update frequency 1/1, `ent_coef=auto`, target entropy −7, CPU, no normalization. Actor/critic/entropy remained finite, parameters updated, replay and timeout handling were valid. Falling alpha alone was consistent with normal autotuning and is not evidence of collapse; do not manually tune entropy by default.

## 10. Evaluation and checkpointing

Policies are deterministic for evaluation. Online evaluation uses 20 episodes at 10k cadence with seeds `700000 + step + episode_index`; final best/last use 100 matched seeds 900000–900099. Selection tuple is success, lift, median physical lift, grasp, then return. Never select by shaped return alone: the failed control reached 95–100% grasp while success stayed 0%. Best fixed checkpoint is effective step 280k; last is 300k.

## 11. Monitoring and behavioral funnel

Monitor return, ever/final success, grasp rate/duration, lift, P(lift|grasp), lift mm, actor/critic loss, alpha, action norm/saturation, translation/rotation/gripper statistics, and post-grasp z action. The tqdm callback displays step/total, ETA/fps and the same core fields persisted in `live_metrics.csv`; evaluations use `tqdm.write` and persist to `learning_curve.csv`. Diagnose Reach→Contact→Grasp→Lift→Success, not return alone.

## 12. Historical debugging path

| Phase | Actual experiment | Lesson |
|---|---|---|
| A | 20k sanity | numerical SAC works; `PASS_NUMERIC_ONLY` |
| B | 100k success-termination control | best 10%, last 5%; reach/grasp only |
| C | control true continuation to 300k | 0% success with 95–100% grasp; `PLATEAU_AT_GRASP` |
| D | fresh 300k signed lift-progress | small intermittent gain, regression; not supported |
| E | fresh fixed 100k + fixed continuation to 300k | best 92%, last 88%; supported |

## 13. Loading and closed-loop use

```python
from rl_baselines.envs import LiftGymnasium
from rl_baselines.policy_adapter import VanillaRLPolicy

env = LiftGymnasium(seed=0, horizon=200, termination_mode="fixed_horizon")
policy = VanillaRLPolicy("/home/brainlab/FaRL/results/vanilla_rl_baseline_v1/sac_seed0_300k_fixed_horizon_from_100k/checkpoints/best_model.zip", algorithm="SAC", device="cpu")
policy.reset()
obs, info = env.reset(seed=0)
for _ in range(200):
    action = policy.act(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
env.close()
```

The required loop is current `obs_t → policy → action_t → env.step → obs_(t+1)`. Never splice a recorded historical action suffix after recovery. A real checkpoint smoke showed changed observations produced changed in-bound actions; `tests/test_policy_adapter_closed_loop.py` guards this contract. Normalization is absent, so there is no VecNormalize state to load.

## 14. Continuation and run registry

A true SAC continuation requires the model zip (policy, targets, optimizers, entropy, timestep) **and replay buffer**; normalization state would also be required if used. Model-only loading is a warm-start. `active_run.json` identifies the latest registered run; managers reject a live concurrent PID and append terminal records to `run_history.jsonl`. Run directories contain `run_status.json`, `live_metrics.csv`, `learning_curve.csv`, `episode_metrics.csv`, `summary.json`, final comparisons, and `checkpoints/{best,last,latest}_model.zip` plus replay where saved.

## 15. Common failure modes — do not repeat

1. Do not judge learning from shaped return alone.
2. Do not call grasp task completion.
3. Do not call low alpha exploration collapse by itself.
4. Do not change reward, termination, entropy, and architecture together.
5. Do not report only last checkpoint.
6. Do not call model-only loading true continuation.
7. Do not encode horizon as true termination.
8. Do not terminate a fixed-horizon episode at success.
9. Do not use recorded action suffixes as closed-loop feedback.
10. Do not claim signed lift-progress solved Lift.
11. Do not claim seed-0 92% is multiseed.

## 16. Forward recommendation and limitations

Use the best fixed-horizon seed-0 checkpoint, low-dimensional observations, original reward, 200-step fixed horizon, OSC_POSE 20 Hz, 256×256 SAC, and automatic entropy for downstream FaRL development. Seeds 1 and 2 remain pending. The very large 0.65–0.72 m maximum lift deltas satisfy official Lift success but need a secondary throw/overshoot/stability quality audit. Do not invalidate the official result without that evidence.

Replication status: control seed 0 completed (best 0%, last 0%); fixed seed 0 completed (best 92%, last 88%, best lift 92%, P(lift|grasp) 98.9%). Control/fixed seeds 1 and 2 are `PENDING`.

## 17. Commands and indexes

Exact commands are in [command_reference.md](vanilla_sac_baseline_recipe/command_reference.md), artifacts in [artifact_index.md](vanilla_sac_baseline_recipe/artifact_index.md), checklist in [reproduction_checklist.md](vanilla_sac_baseline_recipe/reproduction_checklist.md), and troubleshooting in [troubleshooting.md](vanilla_sac_baseline_recipe/troubleshooting.md). Machine-readable values and hashes are in the companion directory.

## 18. Git/source provenance

Repository branch `main`, HEAD `8533fc994f302e49ddc2255d38514bed0722d59f`. The working tree contains extensive untracked project outputs/code; no commit or push was performed. Per-file hashes and full status are captured in `source_provenance.json`.
