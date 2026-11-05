"""Microbenchmark the cached-rollout attention on one accelerator, no model build.

Compares, at the shapes the streaming rollout actually runs, the two backends
``utils.flash_attn.FlashAttention`` can take on Ascend:

    sdpa   ``_sdpa_varlen`` -- torch SDPA per document, materializes the scores
    fused  ``_npu_fused_varlen`` -- ``torch_npu.npu_fusion_attention``, TND layout

and checks that they agree before timing them, so a speedup is never reported
for a kernel that computes something else.

Default shapes are the production trial (768x1376, SP up_size 4) as one rank
sees them after the Ulysses exchange: one chunk of 5 latents x 24 x 43 = 5160
query rows, 56 / 4 = 14 heads of 128, and a key length that grows by one chunk
per step until sink (2) + window (2) + the chunk itself is full::

    python -m projects.minimax_h3.tools.bench_npu_attention

Each line reports latency, effective TFLOP/s and the peak memory one call adds.
Multiply the per-call latency by 50 layers x the forwards per clip (4 NFE + 1
cache write per chunk, 11 chunks at 192 frames) to see what attention alone
costs per video. The 384x672 bring-up shape is ``--q-len 1290``.
"""

from __future__ import annotations

import argparse
import time

import torch

from utils.flash_attn import (
    NPU_FUSED_ATTN_AVAILABLE,
    _npu_fused_varlen,
    _sdpa_varlen,
)


def _device() -> torch.device:
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        pass
    if hasattr(torch, "npu") and torch.npu.is_available():
        return torch.device("npu", 0)
    if torch.cuda.is_available():
        return torch.device("cuda", 0)
    return torch.device("cpu")


def _sync(device: torch.device) -> None:
    if device.type == "npu":
        torch.npu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def _memory(device: torch.device):
    return {"npu": getattr(torch, "npu", None), "cuda": torch.cuda}.get(device.type)


def _bounds(lens: list[int]) -> list[int]:
    out = [0]
    for value in lens:
        out.append(out[-1] + value)
    return out


def _inputs(q_lens, k_lens, heads, head_dim, dtype, device):
    generator = torch.Generator(device="cpu").manual_seed(0)

    def make(rows):
        return torch.randn(rows, heads, head_dim, generator=generator).to(device, dtype)

    return make(sum(q_lens)), make(sum(k_lens)), make(sum(k_lens))


def _run(backend, q, k, v, q_bounds, k_bounds, scale):
    if backend == "sdpa":
        return _sdpa_varlen(
            q, k, v, q_bounds=q_bounds, k_bounds=k_bounds,
            dropout_p=0.0, softmax_scale=scale, causal=False,
        )
    return _npu_fused_varlen(q, k, v, q_bounds=q_bounds, k_bounds=k_bounds, softmax_scale=scale)


def _check(q, k, v, q_bounds, k_bounds, scale, tolerance) -> str:
    reference = _run("sdpa", q, k, v, q_bounds, k_bounds, scale).float()
    candidate = _run("fused", q, k, v, q_bounds, k_bounds, scale).float()
    max_abs = (reference - candidate).abs().max().item()
    cosine = torch.nn.functional.cosine_similarity(
        reference.flatten(), candidate.flatten(), dim=0
    ).item()
    verdict = "OK" if cosine > 1 - tolerance else "MISMATCH"
    return f"{verdict}  max|diff|={max_abs:.3e}  cosine={cosine:.6f}"


def _time(backend, q, k, v, q_bounds, k_bounds, scale, *, warmup, iters, device):
    memory = _memory(device)
    for _ in range(warmup):
        _run(backend, q, k, v, q_bounds, k_bounds, scale)
    _sync(device)
    if memory is not None:
        memory.empty_cache()
        memory.reset_peak_memory_stats()
        base = memory.memory_allocated()
    start = time.perf_counter()
    for _ in range(iters):
        _run(backend, q, k, v, q_bounds, k_bounds, scale)
    _sync(device)
    seconds = (time.perf_counter() - start) / iters
    peak = (memory.max_memory_allocated() - base) / 2**30 if memory is not None else float("nan")
    return seconds, peak


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--q-len", type=int, default=5160, help="query rows per document")
    parser.add_argument(
        "--kv-chunks", type=str, default="1,2,3,4,5",
        help="key length in multiples of --q-len, one benchmark line each",
    )
    parser.add_argument("--docs", type=int, default=1, help="documents packed per call (prompts per SP group)")
    parser.add_argument("--heads", type=int, default=14)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--skip-sdpa", action="store_true", help="time only the fused kernel")
    parser.add_argument("--tolerance", type=float, default=1e-3, help="allowed 1 - cosine")
    args = parser.parse_args()

    device = _device()
    dtype = getattr(torch, args.dtype)
    scale = args.head_dim ** -0.5
    backends = (["sdpa"] if not args.skip_sdpa else []) + (["fused"] if NPU_FUSED_ATTN_AVAILABLE else [])
    if "fused" not in backends:
        print("npu_fusion_attention unavailable (no torch_npu, or RAVEN_NPU_FUSED_ATTN=0): timing SDPA only")
    print(
        f"device={device} dtype={args.dtype} heads={args.heads} head_dim={args.head_dim} "
        f"q_len={args.q_len} docs={args.docs}"
    )

    # Correctness first, on a ragged multi-document packing: the TND bounds are
    # the part most likely to be wrong, and equal-length documents would hide
    # an off-by-one between documents.
    if "fused" in backends:
        q_lens = [args.q_len, max(1, args.q_len // 3)]
        k_lens = [3 * args.q_len, max(1, args.q_len // 2) + args.q_len]
        q, k, v = _inputs(q_lens, k_lens, args.heads, args.head_dim, dtype, device)
        print("check (2 ragged docs):", _check(q, k, v, _bounds(q_lens), _bounds(k_lens), scale, args.tolerance))
        del q, k, v

    rows = []
    for multiple in (int(value) for value in args.kv_chunks.split(",")):
        q_lens = [args.q_len] * args.docs
        k_lens = [args.q_len * multiple] * args.docs
        q, k, v = _inputs(q_lens, k_lens, args.heads, args.head_dim, dtype, device)
        q_bounds, k_bounds = _bounds(q_lens), _bounds(k_lens)
        flops = 4 * sum(a * b for a, b in zip(q_lens, k_lens)) * args.heads * args.head_dim
        line = {"kv": k_lens[0]}
        for backend in backends:
            try:
                seconds, peak = _time(
                    backend, q, k, v, q_bounds, k_bounds, scale,
                    warmup=args.warmup, iters=args.iters, device=device,
                )
                line[backend] = (seconds, flops / seconds / 1e12, peak)
            except RuntimeError as error:  # OOM on the SDPA side is a result too
                line[backend] = error
                memory = _memory(device)
                if memory is not None:
                    memory.empty_cache()
        rows.append(line)
        del q, k, v

    print()
    for line in rows:
        parts = [f"kv={line['kv']:>6}"]
        for backend in backends:
            value = line[backend]
            if isinstance(value, Exception):
                parts.append(f"{backend}: FAILED ({str(value).splitlines()[0][:60]})")
            else:
                seconds, tflops, peak = value
                parts.append(f"{backend}: {seconds * 1e3:8.2f} ms {tflops:6.1f} TFLOP/s +{peak:5.2f} GiB")
        if all(not isinstance(line.get(b), Exception) for b in ("sdpa", "fused")) and len(backends) == 2:
            parts.append(f"speedup x{line['sdpa'][0] / line['fused'][0]:.1f}")
        print("  ".join(parts))


if __name__ == "__main__":
    main()
