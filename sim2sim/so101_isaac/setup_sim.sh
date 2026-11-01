#!/usr/bin/env bash
# Requires uv 0.11.28 and the pinned checkouts created by setup_sources.sh.
set -euo pipefail
cd "$(dirname "$0")"
sim_root="$PWD"
[[ -f IsaacLab/uv.lock && -d leisaac/source/leisaac ]] || { echo 'Run setup_sources.sh first.' >&2; exit 1; }
export UV_PROJECT_ENVIRONMENT="$sim_root/.venv-sim"
uv sync --project IsaacLab --frozen --python 3.12.13 --extra isaacsim
uv pip install --python .venv-sim/bin/python -e leisaac/source/leisaac
uv pip install --python .venv-sim/bin/python 'h5py==3.16.0' 'scipy==1.17.0' 'pillow==12.3.0'
