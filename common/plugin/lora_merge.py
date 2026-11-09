"""Make a LoRA adapter cheap to run after placement (inference only).

An unmerged adapter costs two extra matmuls per wrapped linear, plus the casts
and the add that fold its output back into the base result. peft builds the
adapter in fp32, and on Ascend, where fp32 matmul does not use HF32 by default,
that path took about a quarter of a MiniMax-H3 rollout forward while holding
under 4% of the parameters. Two plugins remove that cost:

``CastLoraAfterPlacement`` keeps the adapter unmerged and casts it to the base
weights' dtype (bf16), so its matmuls run at bf16 speed and nothing is cast per
forward. The base weights stay exactly as loaded; only the adapter branch's
per-call arithmetic loses precision::

    placement:
      plugins:
        - module: common.plugin.lora_merge
          class_name: CastLoraAfterPlacement
          enabled: true   # false runs the adapter in fp32, as peft built it

``MergeLoraAfterPlacement`` folds ``W + scaling * B @ A`` into the base weights
and unloads the adapter. It is the cheapest, but rounding the sum to bf16 moves
each weight by up to half a bf16 step of ``W``. Where the delta is small next
to ``W``, most of its elements are smaller than that step and are lost outright;
the rest are quantized.
The result is a fixed perturbation of the weights that every token and every
chunk sees. On MiniMax-H3's streaming LoRA, clips longer than the trained 8 s
degraded after 8 s with the merge and stayed clean without it, so do not use it
for rollouts past the training length.

Both run after placement, so the arithmetic happens on the device. Sharded
(FSDP) parameters are rejected: the merge needs whole weights, and recasting
an FSDP-managed parameter would bypass FSDP's own dtype handling.
"""

from __future__ import annotations

from typing import Any

import torch

from ..distributed import ops
from ..logging import get_logger
from ..model.placement import PlacementPlugin

logger = get_logger()


class MergeLoraAfterPlacement(PlacementPlugin):
    def after_placement(self, state: dict[str, Any]) -> dict[str, Any]:
        name = state["name"]
        if not self.config.get("enabled", True):
            logger.info("[%s] LoRA merge disabled; the adapter runs unmerged", name)
            return state

        from peft.tuners.lora.layer import LoraLayer
        from torch.distributed.tensor import DTensor

        model = state["model"]
        assert hasattr(model, "peft_config"), (
            f"[{name}] MergeLoraAfterPlacement needs a peft-wrapped model; configure "
            "an adapter or drop the plugin"
        )
        assert not model.training, (
            f"[{name}] merging a LoRA freezes it into the base weights; it is for "
            "inference only (runtime.training: false)"
        )
        assert not any(isinstance(p, DTensor) for p in model.parameters()), (
            f"[{name}] parameters are sharded; the LoRA merge needs whole weights. "
            "Use it with placement.fsdp: null"
        )

        merged = 0
        with torch.no_grad():
            for module in model.modules():
                if not isinstance(module, LoraLayer):
                    continue
                weight = module.get_base_layer().weight
                for adapter in module.active_adapters:
                    if adapter not in module.lora_A:
                        continue
                    assert adapter not in getattr(module, "lora_variant", {}), (
                        f"[{name}] LoRA variants (DoRA, ...) are not supported by this merge"
                    )
                    assert not getattr(module, "lora_bias", {}).get(adapter, False), (
                        f"[{name}] lora_bias is not supported by this merge"
                    )
                    delta = module.get_delta_weight(adapter).float()
                    assert delta.shape == weight.shape, (
                        f"[{name}] LoRA delta {tuple(delta.shape)} does not match "
                        f"base weight {tuple(weight.shape)}"
                    )
                    weight.copy_(weight.float() + delta)
                    merged += 1
        assert merged > 0, f"[{name}] no active LoRA adapter found to merge"

        # unload() returns the wrapped model, which drops whatever the runtime
        # stage annotated on the PeftModel itself. param_dtype is the one
        # attribute forward-time code reads from the model root.
        param_dtype = getattr(model, "param_dtype", None)
        model = model.unload()
        if param_dtype is not None:
            model.param_dtype = param_dtype
        state["model"] = model

        message = f"[{name}] merged LoRA into {merged} base weights and unloaded the adapter"
        device_module = ops.get_device_module()
        if device_module is not None:
            message += f"; {ops.get_device_type().upper()} memory allocated {device_module.memory_allocated() / 2**30:.1f} GiB"
        logger.info(message)
        return state


class CastLoraAfterPlacement(PlacementPlugin):
    """Cast every active LoRA adapter to the dtype of the weight it adapts.

    ``dtype`` (optional) overrides the target, e.g. ``float16``. peft's forward
    casts its input to ``lora_A``'s dtype, so after the cast a bf16 activation
    goes through the adapter with no conversion at all.
    """

    def after_placement(self, state: dict[str, Any]) -> dict[str, Any]:
        name = state["name"]
        if not self.config.get("enabled", True):
            logger.info("[%s] LoRA adapter cast disabled; the adapter keeps its own dtype", name)
            return state

        from peft.tuners.lora.layer import LoraLayer
        from torch.distributed.tensor import DTensor

        model = state["model"]
        assert hasattr(model, "peft_config"), (
            f"[{name}] CastLoraAfterPlacement needs a peft-wrapped model; configure "
            "an adapter or drop the plugin"
        )
        assert not model.training, (
            f"[{name}] casting a LoRA to half precision is for inference only "
            "(runtime.training: false)"
        )
        assert not any(isinstance(p, DTensor) for p in model.parameters()), (
            f"[{name}] parameters are sharded; cast the adapter only with placement.fsdp: null"
        )
        override = getattr(torch, self.config.dtype) if "dtype" in self.config else None

        cast = 0
        dtypes: set[torch.dtype] = set()
        for module in model.modules():
            if not isinstance(module, LoraLayer):
                continue
            target = override or module.get_base_layer().weight.dtype
            for adapter in module.active_adapters:
                if adapter not in module.lora_A:
                    continue
                module.lora_A[adapter].to(target)
                module.lora_B[adapter].to(target)
                dtypes.add(target)
                cast += 1
        assert cast > 0, f"[{name}] no active LoRA adapter found to cast"

        message = (
            f"[{name}] cast {cast} LoRA adapters to "
            f"{', '.join(sorted(str(d).removeprefix('torch.') for d in dtypes))}; they run unmerged"
        )
        device_module = ops.get_device_module()
        if device_module is not None:
            message += f"; {ops.get_device_type().upper()} memory allocated {device_module.memory_allocated() / 2**30:.1f} GiB"
        logger.info(message)
        return state
