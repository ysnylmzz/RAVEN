#!/bin/bash
set -exo pipefail

# Ascend hosts ship npu-smi; everything else is treated as CUDA.
if command -v npu-smi >/dev/null 2>&1; then
    DEVICE=npu
else
    DEVICE=cuda
fi

# Activate the conda env and the repo venv when present; otherwise run in the
# Python environment that is already active (e.g. an Ascend container image).
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    if [ "$DEVICE" = npu ]; then
        export CONDA_ENV="${CONDA_ENV:-raven-npu}"
    else
        export CONDA_ENV="${CONDA_ENV:-raven}"
    fi
    conda activate "$CONDA_ENV"
fi
if [ -f venv/bin/activate ]; then
    source "venv/bin/activate"
fi

if [ "$DEVICE" = npu ]; then
    if [ -z "${ASCEND_SET_ENV}" ]; then
        for candidate in /usr/local/Ascend/ascend-toolkit/set_env.sh /usr/local/Ascend/cann/set_env.sh; do
            [ -f "$candidate" ] && ASCEND_SET_ENV="$candidate" && break
        done
    fi
    [ -n "${ASCEND_SET_ENV}" ] || { echo "CANN set_env.sh not found; export ASCEND_SET_ENV=/path/to/set_env.sh" >&2; exit 1; }
    source "$ASCEND_SET_ENV"
    # Without torch_npu the launcher would quietly fall back to CPU.
    python -c "import torch_npu" || { echo "torch_npu is not importable from $(command -v python); activate the environment that has it" >&2; exit 1; }

    # torch's count, not npu-smi's: it honours ASCEND_RT_VISIBLE_DEVICES and
    # counts both dies of a dual-die card.
    N=${N:-$(python -c "import torch, torch_npu; print(torch.npu.device_count())")}  # nproc per node
    # Ranks reach their first collective on a lazily-created group at different
    # times while loading 30B+ weights; HCCL's 120 s default link setup is too short.
    export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-1800}
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    LAUNCH_BLOCKING_VAR=ASCEND_LAUNCH_BLOCKING
else
    export CUDA_HOME="${CUDA_HOME:-$CONDA_PREFIX}"
    CUDA_TARGET_INCLUDES="$(printf ':%s' "$CUDA_HOME"/targets/*-linux/include)"
    CUDA_TARGET_LIBS="$(printf ':%s' "$CUDA_HOME"/targets/*-linux/lib)"
    export CPATH="${CUDA_TARGET_INCLUDES#:}${CPATH:+:$CPATH}"
    export LIBRARY_PATH="${CUDA_TARGET_LIBS#:}${LIBRARY_PATH:+:$LIBRARY_PATH}"
    export LD_LIBRARY_PATH="${CUDA_TARGET_LIBS#:}:$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

    N=${N:-${SLURM_GPUS_ON_NODE:-$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)}}  # nproc per node
    export NCCL_DEBUG=INFO
    export NCCL_IB_TIMEOUT=31
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    LAUNCH_BLOCKING_VAR=CUDA_LAUNCH_BLOCKING
fi
D=${D:-0}  # debug mode
export TOKENIZERS_PARALLELISM=false

NNODES=${NNODES:-${SLURM_NNODES:-1}}
NPROC_PER_NODE=$N
NODE_RANK=${NODE_RANK:-${SLURM_NODEID:-0}}
LOCAL_ADDR=${LOCAL_ADDR:-$(python -c "import socket; print(socket.gethostbyname('$(hostname)'))")}
if [ -z "${MASTER_ADDR}" ]; then
    if [ -n "${SLURM_JOB_NODELIST}" ]; then
        MASTER_ADDR=$(python -c "import hostlist; print(hostlist.expand_hostlist('$SLURM_JOB_NODELIST')[0])")
    else
        MASTER_ADDR="localhost"
    fi
fi
if [ -z "${MASTER_PORT}" ]; then
    if [ -n "${SLURM_JOB_ID}" ]; then
        MASTER_PORT=$(expr 10000 + $SLURM_JOB_ID % 20000)
    else
        MASTER_PORT=7890
    fi
fi

RDZV_BACKEND=static
if [ $NNODES -gt 1 ]; then
    RDZV_BACKEND=c10d
fi

if [ $D -eq 0 ]; then
    torchrun \
        --nnodes $NNODES \
        --nproc_per_node $NPROC_PER_NODE \
        --node_rank $NODE_RANK \
        --rdzv-endpoint $MASTER_ADDR:$MASTER_PORT \
        --rdzv-backend $RDZV_BACKEND \
        --local_addr=$LOCAL_ADDR \
        -m common.launch --config $@
else
    DEBUGPY_PORT=${DEBUGPY_PORT:-5678}
    export PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT=30
    export "$LAUNCH_BLOCKING_VAR"=1
    python -m debugpy --listen 0.0.0.0:$DEBUGPY_PORT --wait-for-client $(which torchrun) \
        --nnodes 1 \
        --nproc_per_node $D \
        --master-port $MASTER_PORT \
        --monitor-interval 0.1 \
        -m common.launch --config $@
fi
