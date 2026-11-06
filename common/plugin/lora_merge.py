"""Fold a LoRA adapter into its base weights after placement (inference only).

Example::

    placement:
      plugins:
        - module: common.plugin.lora_merge
          class_name: MergeLoraAfterPlacement
          enabled: true   # false keeps the adapter unmerged, for A/B runs

An unmerged adapter costs two extra matmuls per wrapped linear, plus the casts
and the add that fold its output back into the base result. peft builds the
adapter in fp32, and on Ascend, where fp32 matmul does not use HF32 by default,
that path took about a quarter of a MiniMax-H3 rollout forward while holding
under 4% of the parameters. With a fixed adapter and no gradients,
``W + scaling * B @ A`` gives the same result with none of that cost.

This runs after placement so the arithmetic happens on the device. On the host,
every rank would make a private copy of every merged weight instead of sharing
the checkpoint's page cache. Sharded (FSDP) parameters are rejected, because the
merge needs whole weights.

The delta is added in fp32 and rounded to the base weight's dtype once. The
merged model therefore differs from the unmerged one only by that rounding.
After the merge the adapter layers are unloaded, so the model is the plain base
architecture again and the adapter's memory is released.
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
