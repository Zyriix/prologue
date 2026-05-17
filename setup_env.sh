#!/bin/bash
# Create a conda env for Prologue: python 3.12 + torch 2.9.1+cu128 + flash-attn 2.8.3.
# Install order (xformers -> nvcc -> flash-attn -> requirements) anchors torch and lets
# flash-attn source-build against the env's torch+nvcc. Requires conda and an NVIDIA
# driver supporting CUDA 12.8.
#
# Usage:
#   bash setup_env.sh                   # env "prologue" (python 3.12)
#   bash setup_env.sh my-env            # custom env name
#   bash setup_env.sh my-env 3.12       # custom python

set -euo pipefail

ENV_NAME=${1:-prologue}
PY_VER=${2:-3.12}

# Locate conda.sh (PATH or CONDA_ROOT).
_CONDA_SH=""
if command -v conda >/dev/null 2>&1; then
    _CONDA_SH="$(conda info --base)/etc/profile.d/conda.sh"
else
    for _root in "${CONDA_ROOT:-}" "$HOME/miniconda3" "$HOME/anaconda3" \
                 "/opt/conda" "/opt/miniconda3" "/opt/anaconda3"; do
        if [ -n "$_root" ] && [ -f "${_root}/etc/profile.d/conda.sh" ]; then
            _CONDA_SH="${_root}/etc/profile.d/conda.sh"; break
        fi
    done
fi
if [ -z "${_CONDA_SH}" ] || [ ! -f "${_CONDA_SH}" ]; then
    echo "[setup_env] ERROR: could not locate conda.sh. Set CONDA_ROOT=/path/to/miniconda3 and retry." >&2
    exit 1
fi
echo "[setup_env] sourcing ${_CONDA_SH}"
source "${_CONDA_SH}"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "[setup_env] ERROR: conda env '${ENV_NAME}' already exists." >&2
    echo "[setup_env]   Remove it first:  conda env remove -y -n ${ENV_NAME}" >&2
    echo "[setup_env]   Or pick a new name: bash setup_env.sh <new_env_name>" >&2
    exit 1
fi

# conda-forge-only to avoid defaults-channel ToS prompts.
echo "[setup_env] creating conda env '${ENV_NAME}' (python=${PY_VER}) from conda-forge..."
conda create -y -n "${ENV_NAME}" -c conda-forge --override-channels \
    "python=${PY_VER}" pip
conda activate "${ENV_NAME}"
python -m pip install --upgrade pip

# [1/4] xformers + torchvision anchor torch==2.9.1+cu128.
echo "[setup_env] [1/4] installing xformers + torchvision..."
pip install xformers==0.0.33.post2 torchvision==0.24.1

# [2/4] nvcc for flash-attn source build (kept inside $CONDA_PREFIX).
echo "[setup_env] [2/4] installing cuda-nvcc-tools=12.8.* (nvidia + conda-forge)..."
conda install -y -n "${ENV_NAME}" -c nvidia -c conda-forge --override-channels \
    "cuda-nvcc-tools=12.8.*"
conda deactivate
conda activate "${ENV_NAME}"
echo "[setup_env]   nvcc: $(command -v nvcc || echo MISSING)"
nvcc --version | tail -n 1

# [3/4] flash-attn: prebuilt wheel preferred, source build (ninja) fallback.
echo "[setup_env] [3/4] installing flash-attn==2.8.3..."
pip install ninja packaging psutil
pip install --no-build-isolation --no-cache-dir flash-attn==2.8.3

# [4/4] remaining (torch-agnostic) deps.
echo "[setup_env] [4/4] installing remaining deps from requirements.txt..."
pip install -r "$(dirname "$(readlink -f "$0")")/requirements.txt"

echo
echo "[setup_env] sanity check:"
python - <<'PY'
import torch, torchvision, xformers, accelerate, safetensors, omegaconf, einops
import numpy, scipy, torchmetrics, tqdm, PIL, requests, yaml, wandb
import flash_attn, thop, matplotlib
print(f"  torch        {torch.__version__}  (cuda={torch.version.cuda})")
print(f"  torchvision  {torchvision.__version__}")
print(f"  xformers     {xformers.__version__}")
print(f"  flash_attn   {flash_attn.__version__}")
print(f"  accelerate   {accelerate.__version__}")
print(f"  torchmetrics {torchmetrics.__version__}")
print(f"  numpy        {numpy.__version__}")
print(f"  scipy        {scipy.__version__}")
print(f"  thop         present")
print(f"  matplotlib   {matplotlib.__version__}")
PY

echo
echo "[setup_env] done. Activate with:"
echo "    conda activate ${ENV_NAME}"
