#!/bin/bash
set -exo pipefail

# CANN toolkit is installed outside conda (driver + toolkit from the Ascend
# installers); this only sources its environment.
# conda is optional: without it the venv is built from the python on PATH.
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    export CONDA_ENV="${CONDA_ENV:-raven-npu}"
    conda activate "$CONDA_ENV"
fi

if [ -z "${ASCEND_SET_ENV}" ]; then
    for candidate in /usr/local/Ascend/ascend-toolkit/set_env.sh /usr/local/Ascend/cann/set_env.sh; do
        [ -f "$candidate" ] && ASCEND_SET_ENV="$candidate" && break
    done
fi
[ -n "${ASCEND_SET_ENV}" ] || { echo "CANN set_env.sh not found; export ASCEND_SET_ENV=/path/to/set_env.sh" >&2; exit 1; }
source "$ASCEND_SET_ENV"

PREFIX="${PREFIX:-$HOME}/.cache"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PREFIX/uv}"

# No FlashAttention / MagiAttention builds: without flash_attn installed,
# utils/flash_attn.py falls back to its torch SDPA varlen path.
[ ! -d "venv/" ] && uv venv "venv/" --python="$(command -v python)"
uv pip install -r tools/requirements-npu-inference.txt \
    --python "venv/bin/python" \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    --index-strategy unsafe-best-match \
    --link-mode=hardlink
source "venv/bin/activate"

python -c "import torch, torch_npu; assert torch.npu.is_available(), 'torch_npu imported but no NPU is visible'; print('torch', torch.__version__, 'torch_npu', torch_npu.__version__, 'npus', torch.npu.device_count())"
