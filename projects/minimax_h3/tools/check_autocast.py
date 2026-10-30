"""Check that the inference path's autocast regions take effect on this device.

A ``torch.autocast`` keyed to the wrong device type is a silent no-op: nothing
errors, the compute dtype is just wrong. "It ran" is therefore no evidence, so
this records the dtype each Linear actually computes in (forward hooks see the
raw output, before any trailing ``.to()``) at every site MiniMax-H3 inference
relies on:

1. ``AutocastNoGradWrapper`` (wraps the video VAE ``decode_base`` in the
   preview trial): a Linear inside the wrapped call computes in the configured
   dtype.
2. ``BaseSampler.autocast()``: same, for sampler math when enabled.
3. Video VAE ViT decoder fp32 islands under an active autocast: ``x_embedder``
   and ``proj_out`` compute in fp32 while the transformer blocks compute in the
   autocast dtype.

Runs on the active accelerator (NPU, CUDA) or CPU. From the repo root, after
sourcing the CANN env on Ascend::

    python -m projects.minimax_h3.tools.check_autocast [--dtype float16]
"""

from __future__ import annotations

import argparse

try:
    import torch_npu  # noqa: F401
except ModuleNotFoundError as exc:
    if exc.name != "torch_npu":
        raise

import torch
from torch import nn

from common.diffusion.sampler.consistency import ConsistencySampler
from common.distributed import ops
from common.plugin.autocast import AutocastNoGradWrapper
from projects.minimax_h3.modeling.video_vae.vae_vit import ViT3DDecoder


def _record_output_dtypes(modules: dict[str, nn.Module]) -> dict[str, torch.dtype]:
    """First-call output dtype per module (the hook returns None: output untouched)."""
    seen: dict[str, torch.dtype] = {}
    for name, module in modules.items():
        def hook(_module, _inputs, output, name=name):
            seen.setdefault(name, output.dtype)
        module.register_forward_hook(hook)
    return seen


def check_wrapper(device: torch.device, dtype: torch.dtype) -> dict[str, torch.dtype]:
    model = nn.Module()
    model.linear = nn.Linear(16, 16).to(device)
    model.decode_base = lambda x: model.linear(x)
    AutocastNoGradWrapper({
        "functions": ["decode_base"],
        "grad_enabled": False,
        "autocast": {"enabled": True, "dtype": str(dtype).removeprefix("torch.")},
    }).after_runtime({"model": model})
    seen = _record_output_dtypes({"wrapper/linear": model.linear})
    model.decode_base(torch.randn(4, 16, device=device))
    return seen


def check_sampler(device: torch.device, dtype: torch.dtype) -> dict[str, torch.dtype]:
    sampler = ConsistencySampler(schedule=None, autocast={"enabled": True, "dtype": str(dtype).removeprefix("torch.")})
    linear = nn.Linear(16, 16).to(device)
    seen = _record_output_dtypes({"sampler/linear": linear})
    with sampler.autocast():
        linear(torch.randn(4, 16, device=device))
    return seen


def check_vae_vit_islands(device: torch.device, dtype: torch.dtype) -> dict[str, torch.dtype]:
    # Same block recipe as configs/video_vae.yaml vit_decoder_kwargs, one tiny layer.
    decoder = ViT3DDecoder(
        patch_size=2,
        patch_size_t=1,
        in_channels=4,
        out_channels=3,
        num_layers=1,
        heads=2,
        dim_head=64,
        norm_type="rms_norm",
        qk_norm_type="rms_norm",
        ffn_activation_fn="silu",
        ffn_use_gated=True,
        rope_dim_ratio=0.75,
    ).to(device).eval()
    block = decoder.transformer_blocks[0]
    seen = _record_output_dtypes({
        "vae_vit/x_embedder": decoder.x_embedder,
        "vae_vit/proj_out": decoder.proj_out,
        "vae_vit/block.to_qkv": block.attn.to_qkv,
        "vae_vit/block.ff.w2": block.ff.w2,
    })
    with torch.no_grad(), torch.autocast(device.type, dtype=dtype):
        decoder(torch.randn(1, 4, 2, 4, 4, device=device))
    return seen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    dtype = getattr(torch, parser.parse_args().dtype)
    device = ops.get_device()

    expected = {
        "wrapper/linear": dtype,
        "sampler/linear": dtype,
        "vae_vit/x_embedder": torch.float32,
        "vae_vit/proj_out": torch.float32,
        "vae_vit/block.to_qkv": dtype,
        "vae_vit/block.ff.w2": dtype,
    }
    seen = {}
    seen.update(check_wrapper(device, dtype))
    seen.update(check_sampler(device, dtype))
    seen.update(check_vae_vit_islands(device, dtype))

    failed = False
    print(f"device={device} autocast_dtype={dtype}")
    for name, want in expected.items():
        got = seen.get(name)
        ok = got == want
        failed |= not ok
        print(f"  {'OK  ' if ok else 'FAIL'} {name:24s} expected={want} got={got}")
    if failed:
        raise SystemExit("autocast check failed: an autocast region is a no-op or an fp32 island leaks")
    print("autocast check passed")


if __name__ == "__main__":
    main()
