#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MUJOCO_GL=disable
export PIP_CACHE_DIR="${BT_PROJECT_CACHE_DIR:-/tmp}/pancake-pip"
python -m pip install -r requirements.txt
python -m pip check
export WARP_CACHE_PATH="${BT_PROJECT_CACHE_DIR:-/tmp}/pancake-warp-3.13"
mkdir -p "$WARP_CACHE_PATH"
python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("No CUDA device: check H100 allocation/container before training")
print("GPU:", torch.cuda.get_device_name(0), "PyTorch:", torch.__version__, flush=True)
PY
exec python -u train.py --mode "${1:-smoke}" --device cuda
