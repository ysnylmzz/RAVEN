"""Distributed process-group bootstrap."""

from __future__ import annotations

import os
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist

from ..logging import get_logger
from ..seed import seed_everything
from . import ops

logger = get_logger()


def configure_distributed(config: Any) -> Any:
    """Initialize the default process group before anything else touches it.

    Only runs under a distributed launcher (torchrun sets WORLD_SIZE); plain
    single-process invocations skip initialization entirely.

    The accelerator is CUDA or Ascend NPU (``ops.get_device_type``); everything
    below says "device" for whichever one this process drives.

    Decisions baked in:
    - bind the local device *before* creating the process group so NCCL/HCCL
      communicators attach to the right card;
    - dual-backend default pg (``cpu:gloo,cuda:nccl`` / ``cpu:gloo,npu:hccl``,
      the form recommended by ``dcp.async_save``) so CPU collectives never ride
      the device collective stream;
    - ``device_id`` enables eager communicator init and device-aware ``barrier()``;
    - initializing here -- before persistence builds the Checkpointer and
      before any training collective is in flight -- keeps later
      ``dist.new_group`` calls (e.g. the checkpointer's async gloo group) out
      of mid-training timing races.

    Environment knobs (config node ``distributed``): ``timeout_minutes`` (30),
    ``cudnn_benchmark`` (false), ``deterministic`` (false; also pins the cuBLAS
    workspace for reproducible GEMMs and makes HCCL reductions deterministic),
    ``alloc_conf`` (``expandable_segments:True``, exported as both
    ``PYTORCH_CUDA_ALLOC_CONF`` and ``PYTORCH_NPU_ALLOC_CONF``; curbs
    fragmentation under video workloads' varying allocation sizes). Env vars
    use ``setdefault`` semantics -- an external ``export`` wins. They are set
    for both backends *before* the device type is resolved, because resolving
    it enumerates devices; they only take effect because this function is the
    first device-touching code in the launch chain: torch reads the device
    order at first enumeration and the allocator config at first allocation,
    never at import.
    """
    dist_config = config.get("distributed", {})
    # The env knobs below are read by torch/CUDA/CANN lazily (device order at
    # first driver enumeration, allocator config at device state init) -- they
    # only work if nothing has touched the device yet. Heavy libraries are all
    # imported lazily inside stage functions for exactly this reason; these
    # asserts fail loud if a future import regresses that discipline. Unlike
    # ops.get_device_type(), is_initialized() never enumerates devices.
    assert not torch.cuda.is_initialized(), "CUDA was initialized before configure_distributed; env knobs would be ignored"
    assert not (hasattr(torch, "npu") and torch.npu.is_initialized()), (
        "NPU was initialized before configure_distributed; env knobs would be ignored"
    )
    alloc_conf = dist_config.get("alloc_conf", "expandable_segments:True")
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", alloc_conf)
    os.environ.setdefault("PYTORCH_NPU_ALLOC_CONF", alloc_conf)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.backends.cudnn.benchmark = dist_config.get("cudnn_benchmark", False)
    if dist_config.get("deterministic", False):
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        os.environ["HCCL_DETERMINISTIC"] = "true"
        torch.use_deterministic_algorithms(True)

    # Same seed on every rank, before anything constructs a model: replicated
    # parameters (LoRA init, from-scratch init) must be born bit-identical
    # across ranks -- sharding trusts local values and never re-syncs them.
    # Per-rank streams are derived later by the engine (common.seed.RandomState).
    seed_everything(config.get("seed", 1019))

    if "WORLD_SIZE" not in os.environ:
        return config

    device = None
    device_module = ops.get_device_module()
    if device_module is not None:
        local_rank = ops.get_local_rank()
        device_module.set_device(local_rank)
        device = torch.device(ops.get_device_type(), local_rank)

    timeout_minutes = dist_config.get("timeout_minutes", 30)
    backend = ops.get_process_group_backend()
    dist.init_process_group(
        backend=backend,
        timeout=timedelta(minutes=timeout_minutes),
        device_id=device,
    )
    logger.info(
        "Distributed initialized: rank=%s/%s local_rank=%s backend=%s device=%s timeout=%smin",
        dist.get_rank(), dist.get_world_size(), os.environ.get("LOCAL_RANK", "0"), backend, device, timeout_minutes,
    )
    return config


def destroy_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
