#!/usr/bin/env bash
set -euo pipefail
cd /home/brainlab/FaRL
export PYGLFW_LIBRARY=/workspace/collectenv/lib/libglfw.so.3
export MUJOCO_PY_MUJOCO_PATH=/workspace/mujoco-2.1.1
export MUJOCO_GL=osmesa
export LD_LIBRARY_PATH=/workspace/collectenv/lib:/workspace/mujoco-2.1.1/bin:/workspace/mujoco-2.1.1/lib:${LD_LIBRARY_PATH:-}

steps=${1:-20000}
seed=${2:-0}
run_name=${3:-sac_${steps}_seed${seed}}
out="results/vanilla_rl_baseline_v1/${run_name}"
if (( $# >= 3 )); then shift 3; else shift "$#"; fi

exec .venv_rl/bin/python -u -m rl_baselines.run_manager \
  --steps "$steps" \
  --seed "$seed" \
  --run-name "$run_name" \
  "$@"
