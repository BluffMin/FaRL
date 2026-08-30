#!/usr/bin/env bash
set -euo pipefail
cd /home/brainlab/FaRL
exec ./scripts/run_vanilla_sac_from_checkpoint.sh \
  results/vanilla_rl_baseline_v1/sac_seed0_100k_fixed_horizon/checkpoints/last_model.zip \
  300000 \
  0 \
  sac_seed0_300k_fixed_horizon_from_100k \
  100000 \
  fixed_horizon
