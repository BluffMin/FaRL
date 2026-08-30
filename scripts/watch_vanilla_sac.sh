#!/usr/bin/env bash
set -euo pipefail
cd /home/brainlab/FaRL
exec .venv_rl/bin/python -u -m rl_baselines.watch_tqdm "${1:-}"
