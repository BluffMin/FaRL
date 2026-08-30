#!/usr/bin/env bash
set -euo pipefail
cd /home/brainlab/FaRL

wait_run="sac_seed0_100k_fixed_horizon"
wait_manager_pid=2154543
status_file="results/vanilla_rl_baseline_v1/${wait_run}/run_status.json"
continuation_log="results/vanilla_rl_baseline_v1/sac_seed0_300k_from_last/continuation_console.log"

echo "WAITING_FOR=${wait_run} MANAGER_PID=${wait_manager_pid}"
while kill -0 "$wait_manager_pid" 2>/dev/null; do
  sleep 30
done

status=$(.venv_rl/bin/python -c "import json; print(json.load(open('${status_file}')).get('status'))")
exit_code=$(.venv_rl/bin/python -c "import json; print(json.load(open('${status_file}')).get('exit_code'))")
if [[ "$status" != "COMPLETED" || "$exit_code" != "0" ]]; then
  echo "ABORT_CONTINUATION: ${wait_run} status=${status} exit_code=${exit_code}"
  exit 2
fi

echo "PREDECESSOR_COMPLETED; STARTING_300K_CONTINUATION"
./scripts/run_vanilla_sac_from_checkpoint.sh \
  results/vanilla_rl_baseline_v1/sac_seed0_100k_diag/checkpoints/last_model.zip \
  300000 \
  0 \
  sac_seed0_300k_from_last 2>&1 | tee "$continuation_log"
