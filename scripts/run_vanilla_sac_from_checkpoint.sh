#!/usr/bin/env bash
set -euo pipefail
cd /home/brainlab/FaRL
export PYGLFW_LIBRARY=/workspace/collectenv/lib/libglfw.so.3
export MUJOCO_PY_MUJOCO_PATH=/workspace/mujoco-2.1.1
export MUJOCO_GL=osmesa
export LD_LIBRARY_PATH=/workspace/collectenv/lib:/workspace/mujoco-2.1.1/bin:/workspace/mujoco-2.1.1/lib:${LD_LIBRARY_PATH:-}
parent_model=${1:?parent checkpoint required}
target=${2:?target total steps required}
seed=${3:?seed required}
run_name=${4:?run name required}
parent_step=${5:-100000}
termination_mode=${6:-success}
parent_replay="$(dirname "$parent_model")/replay_buffer.pkl"
exec .venv_rl/bin/python -u -m rl_baselines.resume_manager \
  --parent-model "$parent_model" \
  --parent-replay "$parent_replay" \
  --parent-step "$parent_step" \
  --target-total-steps "$target" \
  --seed "$seed" \
  --run-name "$run_name" \
  --termination-mode "$termination_mode"
