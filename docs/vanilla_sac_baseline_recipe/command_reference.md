# Command reference

## Fresh run

```bash
cd /home/brainlab/FaRL
./scripts/run_vanilla_sac.sh <steps> <seed> <run_name> [--termination-mode success|fixed_horizon] [--reward-mode control|signed_lift_progress]
```

Exact successful parent command (from `run_status.json`):

```bash
./scripts/run_vanilla_sac.sh 100000 0 sac_seed0_100k_fixed_horizon --termination-mode fixed_horizon
```

Exact continuation helper:

```bash
./scripts/run_fixed_horizon_300k_continuation.sh
```

This helper uses the saved fixed 100k last model **and replay buffer**, target 300k, seed 0, and `fixed_horizon`. Do not rerun into an existing result directory.

## Monitor / inspect / plot

```bash
./scripts/watch_vanilla_sac.sh
./scripts/watch_vanilla_sac.sh <run_name>
./scripts/inspect_vanilla_sac.sh
.venv_rl/bin/python -m rl_baselines.plot_run --run results/vanilla_rl_baseline_v1/<run_name>
.venv_rl/bin/python -m rl_baselines.audit_env
.venv_rl/bin/python -m rl_baselines.reward_audit
```

There is no single verified command that recreates the complete two-stage provenance without overwriting names. Use new run names and first create a fresh fixed-horizon 100k parent, then invoke `run_vanilla_sac_from_checkpoint.sh <parent-last> 300000 <seed> <new-name> 100000 fixed_horizon`.
