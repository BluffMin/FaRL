# FaRL Vanilla SAC Baseline

Reproducible low-dimensional SAC baseline for robosuite `Lift` with Panda,
`OSC_POSE`, and fixed-horizon execution.

The established seed-0 result reaches 92% ever-success at its best checkpoint
and 88% at the final checkpoint. This is a seed-0 result; seeds 1 and 2 remain
pending. Model checkpoints and replay buffers are intentionally not stored in
Git because of their size.

Start with [the canonical baseline recipe](docs/vanilla_sac_baseline_recipe.md).

## Layout

- `rl_baselines/`: environment wrapper, SAC training, evaluation, monitoring,
  continuation, and analysis code
- `scripts/`: user-facing training and monitoring commands
- `configs/vanilla_rl_baseline_v1.yaml`: experiment configuration
- `docs/vanilla_sac_baseline_recipe/`: machine-readable provenance and manifests
- `results/vanilla_rl_baseline_v1/`: curated lightweight reports and figures
- `tests/`: baseline and closed-loop policy adapter regression tests

## Environment

The historical runtime uses Python 3.9, robosuite 1.2.0, MuJoCo 2.1.1,
Gymnasium 1.1.1, Stable-Baselines3 2.7.1, and CPU PyTorch. Exact dependencies
and required environment variables are recorded in the recipe directory.

## Important provenance

The successful policy was trained with fixed-horizon semantics from step zero:

1. fresh fixed-horizon SAC from 0 to 100k;
2. true continuation with its model, optimizers, entropy state, timestep, and
   replay buffer from 100k to 300k.

Do not describe it as a success-termination model switched to fixed horizon.

## Code style

The Python sources use Black formatting with a 100-character line limit. To
check or refresh formatting without changing experiment semantics:

```bash
black --check rl_baselines tests
black rl_baselines tests
```
