#!/usr/bin/env bash
set -euo pipefail
cd /home/brainlab/FaRL
exec .venv_rl/bin/python -m rl_baselines.inspect_run
