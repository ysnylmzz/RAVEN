"""Measure the real-time factor (RTF) of MiniMax-H3 streaming inference.

RTF = generation wall-clock / playback duration of the clip produced. RTF < 1
means the run emits video faster than it plays, which is the bar a streaming
deployment has to clear; RTF 2.0 means every second of video costs two seconds
of compute.

This is a val-only engine, not a standalone script, because the numbers are
only meaningful against the real model build: FSDP wrapping, CPU offload and
placement all land on the critical path, and a hand-rolled harness that skips
``build_models`` would measure a different program. It therefore reuses the
trial configs verbatim and only replaces what ``validate`` does with the
prompts -- time the phases instead of saving mp4s.

Three things it does that a naive ``time.perf_counter()`` around the call would
get wrong:

1. **Device synchronization.** NPU/CUDA kernel launches are asynchronous, so an
   unsynchronized timer measures launch latency, not compute. Every boundary
   here syncs first.
2. **Warmup.** The first pass pays CANN/cuDNN op compilation and allocator
   growth that no later pass repeats. Warmup iterations are run and discarded.
3. **Slowest-rank aggregation.** Under FSDP/SP the wall clock is set by the
   last rank to arrive, so per-phase timings are reduced with ``max`` across
   ranks rather than averaged.

The phase split mirrors the production path:

    encode   ``_validation_inputs``  (Qwen3-VL text encoder, once per prompt)
    rollout  ``_rollout_latents``    (the 4-NFE causal chunk loop -- the DiT)
    decode   ``_decode_latents``     (video VAE + audio VAE; the video tiles are
                                      split across the SP group unless
                                      validation.decode_parallel is false, and
                                      a clip too large for the device decodes
                                      into host memory, see
                                      validation.decode_to_host)

``_chunk_forward`` is additionally wrapped so the rollout's internal structure
falls out: per-chunk latency and, for streaming, time-to-first-chunk -- when
the first chunk is fully denoised and could start playing.

Run it exactly like any other trial::

    ASCEND_RT_VISIBLE_DEVICES=0 N=1 bash tools/multi_run.sh \
        projects/minimax_h3/trials/npu/npu_bench_rtf_950.yaml

Config knobs (all optional, under a top-level ``benchmark:`` block)::

    benchmark:
      warmup: 1        # discarded iterations
      iters: 3         # measured iterations
      decode: true     # include the VAE decode in the timed path
      per_chunk: true  # wrap _chunk_forward for the streaming profile
      profile: false   # one extra untimed rollout with the op profiler on one chunk
      profile_chunk: 5 # which chunk; past sink+window so the KV cache is full
      profile_ranks: [0]

The backbone's parameter dtypes (LoRA adapter vs base weights) are logged and
stored in the report, since an fp32 adapter is a silent slowdown on Ascend.

Reports a table to stdout and writes ``benchmark/rtf.{md,json}`` under the run
directory; with ``profile`` on, traces land in ``benchmark/profile/rank{N}/``.
"""

from __future__ import annotations

import json
import platform
import statistics
import time
from contextlib import contextmanager
from typing import Any, Iterator

import torch
from tabulate import tabulate

from common.distributed import ops
from common.engine import BaseEngine
from common.logging import get_logger
from common.phase import ExecutionPhase, execution_phase
from common.seed import RandomState, combine_seed
from projects.minimax_h3.modeling.constants import MINIMAX_H3_SUPPORTED_FPS

logger = get_logger()

# Phase keys in report order. ``decode`` is optional (see benchmark.decode), so
# the report filters against what was actually recorded rather than assuming.
_PHASES = ("encode", "rollout", "decode")


class _SyncTimer:
    """Wall-clock timing with the accelerator drained at both boundaries."""

    def __init__(self) -> None:
        # None on CPU; ops owns the cuda/npu/cpu branch (Faz 2's device layer),
        # so this harness runs unchanged on the CUDA golden reference.
        self._device_module = ops.get_device_module()

    def sync(self) -> None:
        if self._device_module is not None:
            self._device_module.synchronize()

    @contextmanager
    def measure(self, store: dict[str, float], key: str) -> Iterator[None]:
        self.sync()
        start = time.perf_counter()
        try:
            yield
        finally:
            self.sync()
            store[key] = time.perf_counter() - start


@contextmanager
def _record_chunk_forwards(
    meta: Any, timer: _SyncTimer, sink: list[dict[str, Any]]
) -> Iterator[None]:
    """Time every ``_chunk_forward`` call for the duration of one rollout.

    Shadows the bound method with an instance attribute and pops it on exit, so
    nothing survives the context even if the rollout raises.

    The per-call sync this adds serializes launch against compute, which a
    clean rollout would overlap. At 33B parameters per forward that overlap is
    noise, but it is why the chunk timings are reported as a *profile* and the
    headline rollout number comes from the outer, uninstrumented timer.
    """
    original = meta._chunk_forward
    origin = time.perf_counter()

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        timer.sync()
        start = time.perf_counter()
        out = original(*args, **kwargs)
        timer.sync()
        end = time.perf_counter()
        sink.append(
            {
                "chunk_index": kwargs.get("chunk_index"),
                "role": kwargs.get("role"),
                "seconds": end - start,
                "done_at": end - origin,
            }
        )
        return out

    meta._chunk_forward = wrapped
    try:
        yield
    finally:
        meta.__dict__.pop("_chunk_forward", None)


def _param_dtype_report(model: torch.nn.Module) -> dict[str, dict[str, int]]:
    """Parameter count per dtype, LoRA adapter and base weights separately.

    A trial's ``placement.weight_dtype`` casts the *trainable* parameters, which
    can include the LoRA adapter. An fp32 adapter makes ``lora.py`` cast every
    adapter input to fp32 and run the adapter matmuls in fp32 -- cheap on CUDA
    with TF32, slow on Ascend, where fp32 matmul does not use HF32 by default.
    """
    report: dict[str, dict[str, int]] = {"lora": {}, "base": {}}
    for name, param in model.named_parameters():
        group = "lora" if "lora_" in name else "base"
        key = str(param.dtype).removeprefix("torch.")
        report[group][key] = report[group].get(key, 0) + param.numel()
    return report


def _make_profiler(out_dir: Any) -> Any:
    """Op-level profiler for the active device, writing a trace under ``out_dir``.

    On NPU this is torch_npu's profiler; its output (``ASCEND_PROFILER_OUTPUT/``
    with ``op_statistic.csv``, ``kernel_details.csv`` and the timeline) opens in
    MindStudio Insight. Level1 records CANN-level ops including HCCL and AI CPU
    tasks, which is what separates "slow kernel" from "idle card".
    """
    if ops.get_device_type() == "npu":
        import torch_npu

        experimental = torch_npu.profiler._ExperimentalConfig(
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        )
        return torch_npu.profiler.profile(
            activities=[
                torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU,
            ],
            record_shapes=True,
            on_trace_ready=torch_npu.profiler.tensorboard_ascend_trace_handler(str(out_dir)),
            experimental_config=experimental,
        )
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    return torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(out_dir)),
    )


@contextmanager
def _profile_chunk(
    meta: Any, timer: _SyncTimer, target_chunk: int, out_dir: Any
) -> Iterator[list[int]]:
    """Profile every forward of one chunk: its NFE denoising steps and its cache write.

    Starts at the first ``_chunk_forward`` for ``target_chunk`` and stops at the
    first call for any other chunk (or on exit). Pick a chunk past sink+window
    so the KV cache is at its steady-state length. Yields a one-element list
    holding the number of forwards captured.
    """
    original = meta._chunk_forward
    profiler = _make_profiler(out_dir)
    state = {"started": False, "stopped": False}
    captured = [0]

    def stop() -> None:
        if state["started"] and not state["stopped"]:
            timer.sync()
            profiler.stop()
            state["stopped"] = True

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        chunk_index = kwargs.get("chunk_index")
        if chunk_index == target_chunk and not state["started"]:
            timer.sync()
            profiler.start()
            state["started"] = True
        elif chunk_index != target_chunk:
            stop()
        out = original(*args, **kwargs)
        if state["started"] and not state["stopped"]:
            captured[0] += 1
        return out

    meta._chunk_forward = wrapped
    try:
        yield captured
    finally:
        stop()
        meta.__dict__.pop("_chunk_forward", None)


def _summarize(samples: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(samples),
        "min": min(samples),
        "max": max(samples),
        "stdev": statistics.stdev(samples) if len(samples) > 1 else 0.0,
    }


class RTFBenchmarkEngine(BaseEngine):
    """Times the inference path instead of writing media. See module docstring."""

    def run(self) -> None:
        config = self.config
        validation = config.validation
        bench = config.get("benchmark", {})
        warmup = int(bench.get("warmup", 1))
        iters = int(bench.get("iters", 3))
        want_decode = bool(bench.get("decode", True))
        per_chunk = bool(bench.get("per_chunk", True))
        # After the measured iterations, one extra untimed rollout with the
        # op-level profiler around a single chunk. Off by default: the trace is
        # large and the profiler perturbs the timings of whatever it wraps.
        profile = bool(bench.get("profile", False))
        profile_chunk = int(bench.get("profile_chunk", 5))
        profile_ranks = set(bench.get("profile_ranks", [0]))
        assert iters >= 1, f"benchmark.iters must be >= 1, got {iters}"

        timer = _SyncTimer()
        device_module = ops.get_device_module()
        meta = self.meta
        models = self.models
        backbone = models["backbone"]

        dtype_report = _param_dtype_report(backbone)
        logger.info("[benchmark] backbone parameter dtypes (numel): %s", dtype_report)

        prompts = meta._validation_prompts(validation)
        assert prompts, "benchmark requires at least one validation prompt"
        # One prompt for every iteration: the point is run-to-run variance of
        # the same work, not prompt-to-prompt variance of different work.
        prompt = prompts[0]

        # Decode exactly as validate does: with validation.decode_parallel the
        # whole SP group shares the video decode tiles; without it only the
        # group's main rank decodes, since a full decode on every rank would
        # be duplicated work that distorts the slowest-rank reduction.
        sp_size = meta._sp_size()
        is_group_main = ops.get_rank() % sp_size == 0
        decode_group = meta._decode_group(validation)
        decode_to_host = validation.get("decode_to_host", None)
        logger.info(
            "[benchmark] video decode: %s",
            f"tiles split across the SP group ({sp_size} ranks)"
            if decode_group is not None else "main rank of each SP group only",
        )

        fps = int(validation.get("fps", MINIMAX_H3_SUPPORTED_FPS))
        num_frames = int(validation.num_frames)

        iterations: list[dict[str, float]] = []
        chunk_profile: list[dict[str, Any]] = []
        decoded_frames: int | None = None

        was_training = [(model, model.training) for model in models.values()]
        try:
            for model in models.values():
                model.eval()

            # The phase the production path runs in: model internals branch on
            # it, so timing outside it would time a different program.
            with execution_phase(ExecutionPhase.VALIDATION):
                total = warmup + iters
                for index in range(total):
                    measured = index >= warmup
                    label = "measure" if measured else "warmup"
                    logger.info(
                        "[benchmark] %s iteration %d/%d", label,
                        (index - warmup + 1) if measured else (index + 1),
                        iters if measured else warmup,
                    )

                    # Reset the peak at the start of the first measured pass so
                    # the reported peak reflects steady state, not warmup's
                    # allocator growth.
                    if measured and index == warmup and device_module is not None:
                        device_module.reset_peak_memory_stats()

                    # When decode runs on the SP main ranks only, the other
                    # ranks would enter the next encode early and its timer
                    # would absorb the wait for that decode: encode, and the
                    # max-reduced total, came out ~one decode too long.
                    timer.sync()
                    ops.barrier()

                    phases: dict[str, float] = {}
                    rng = RandomState(
                        combine_seed(int(validation.seed), "benchmark", index, "backbone")
                    )

                    with timer.measure(phases, "encode"):
                        inputs = meta._validation_inputs(config, models, [prompt])

                    chunks: list[dict[str, Any]] = []
                    if per_chunk and measured:
                        with _record_chunk_forwards(meta, timer, chunks):
                            with timer.measure(phases, "rollout"):
                                latents, _ = meta._rollout_latents(
                                    backbone, inputs, rng, keep_trajectory=False, rngs=[rng]
                                )
                    else:
                        with timer.measure(phases, "rollout"):
                            latents, _ = meta._rollout_latents(
                                backbone, inputs, rng, keep_trajectory=False, rngs=[rng]
                            )

                    if want_decode and (is_group_main or decode_group is not None):
                        with timer.measure(phases, "decode"):
                            frames, waveform = meta._decode_latents(
                                models,
                                latents,
                                index=0,
                                decode_group=decode_group,
                                keep=is_group_main,
                                to_host=decode_to_host,
                            )
                        # Trust the decoder over the config for the clip length
                        # the RTF divides by. Only the main ranks keep frames.
                        if frames is not None:
                            decoded_frames = int(frames.shape[2])
                        del frames, waveform

                    del latents, inputs

                    if measured:
                        phases["total"] = sum(
                            phases[key] for key in _PHASES if key in phases
                        )
                        iterations.append(phases)
                        if chunks:
                            chunk_profile = chunks

                if profile:
                    # Every rank runs the rollout (the collectives need them
                    # all); only the selected ranks record.
                    rng = RandomState(
                        combine_seed(int(validation.seed), "benchmark", "profile", "backbone")
                    )
                    inputs = meta._validation_inputs(config, models, [prompt])
                    if ops.get_rank() in profile_ranks:
                        out_dir = logger.iter_dir("benchmark") / "profile" / f"rank{ops.get_rank()}"
                        out_dir.mkdir(parents=True, exist_ok=True)
                        with _profile_chunk(meta, timer, profile_chunk, out_dir) as captured:
                            latents, _ = meta._rollout_latents(
                                backbone, inputs, rng, keep_trajectory=False, rngs=[rng]
                            )
                        if captured[0] == 0:
                            logger.warning(
                                "[benchmark] profile_chunk %d was never run; no trace written",
                                profile_chunk,
                            )
                        else:
                            logger.info(
                                "[benchmark] profiled %d forwards of chunk %d -> %s",
                                captured[0], profile_chunk, out_dir,
                            )
                    else:
                        latents, _ = meta._rollout_latents(
                            backbone, inputs, rng, keep_trajectory=False, rngs=[rng]
                        )
                    del latents, inputs
        finally:
            for model, training in was_training:
                model.train(training)

        peak_gb = None
        if device_module is not None:
            peak_gb = device_module.max_memory_allocated() / (1024**3)

        self._report(
            iterations=iterations,
            chunk_profile=chunk_profile,
            num_frames=decoded_frames or num_frames,
            fps=fps,
            peak_gb=peak_gb,
            warmup=warmup,
            want_decode=want_decode,
            dtype_report=dtype_report,
        )

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def _report(
        self,
        *,
        iterations: list[dict[str, float]],
        chunk_profile: list[dict[str, Any]],
        num_frames: int,
        fps: int,
        peak_gb: float | None,
        warmup: int,
        want_decode: bool,
        dtype_report: dict[str, dict[str, int]],
    ) -> None:
        """Reduce across ranks, then log and persist the result from rank 0."""
        # Slowest rank sets the wall clock. Without a shared decode, decode is
        # absent on non-main ranks, so the default keeps max() from inventing
        # a faster run.
        gathered = ops.all_gather_object(iterations)
        merged: list[dict[str, float]] = []
        for index in range(len(iterations)):
            keys = {key for rank in gathered for key in rank[index]}
            merged.append(
                {key: max(rank[index].get(key, 0.0) for rank in gathered) for key in keys}
            )
        # total is re-derived after the reduction: the per-rank sums would
        # otherwise credit one rank's encode to another's rollout.
        for entry in merged:
            entry["total"] = sum(entry.get(key, 0.0) for key in _PHASES)

        peaks = [value for value in ops.all_gather_object(peak_gb) if value is not None]

        if ops.get_rank() != 0:
            ops.barrier()
            return

        present = [key for key in _PHASES if any(key in entry for entry in merged)]
        stats = {key: _summarize([entry[key] for entry in merged if key in entry]) for key in present}
        stats["total"] = _summarize([entry["total"] for entry in merged])

        video_seconds = num_frames / fps
        total_mean = stats["total"]["mean"]
        rollout_mean = stats["rollout"]["mean"]

        headline = {
            "rtf_end_to_end": total_mean / video_seconds,
            "rtf_rollout_only": rollout_mean / video_seconds,
            "generated_fps": num_frames / total_mean,
            "clip_seconds": video_seconds,
            "num_frames": num_frames,
            "fps": fps,
            "seconds_per_clip": total_mean,
            "peak_memory_gb": max(peaks) if peaks else None,
            "world_size": ops.get_world_size(),
            "device_type": ops.get_device_type(),
            "iters": len(merged),
            "warmup": warmup,
            "decode_included": want_decode,
            "backbone_param_dtypes": dtype_report,
            "torch": torch.__version__,
            "host": platform.node(),
        }

        phase_rows = [
            [
                key,
                f"{stats[key]['mean']:.3f}",
                f"{stats[key]['min']:.3f}",
                f"{stats[key]['max']:.3f}",
                f"{stats[key]['stdev']:.3f}",
                f"{100 * stats[key]['mean'] / total_mean:.1f}%",
            ]
            for key in present + ["total"]
        ]
        phase_table = tabulate(
            phase_rows,
            headers=["phase", "mean s", "min s", "max s", "stdev", "share"],
            tablefmt="heavy_outline",
            numalign="left",
            stralign="left",
        )

        summary_rows = [
            ["RTF (end to end)", f"{headline['rtf_end_to_end']:.3f}"],
            ["RTF (rollout only)", f"{headline['rtf_rollout_only']:.3f}"],
            ["seconds per clip", f"{total_mean:.2f}"],
            ["generated fps", f"{headline['generated_fps']:.2f}"],
            ["clip length", f"{num_frames} frames @ {fps} fps = {video_seconds:.2f} s"],
            ["peak memory", f"{headline['peak_memory_gb']:.1f} GiB" if peaks else "n/a"],
            ["world size", str(headline["world_size"])],
        ]

        chunk_table = ""
        if chunk_profile:
            # "Chunk k is ready" = its last forward returned; with 4 NFE steps
            # a chunk is touched several times before it is done.
            by_chunk: dict[Any, dict[str, Any]] = {}
            for record in chunk_profile:
                entry = by_chunk.setdefault(
                    record["chunk_index"], {"calls": 0, "seconds": 0.0, "ready_at": 0.0}
                )
                entry["calls"] += 1
                entry["seconds"] += record["seconds"]
                entry["ready_at"] = max(entry["ready_at"], record["done_at"])
            ordered = sorted(by_chunk.items(), key=lambda item: item[1]["ready_at"])
            first_ready = ordered[0][1]["ready_at"]
            headline["time_to_first_chunk_s"] = first_ready
            headline["num_chunks"] = len(ordered)
            chunk_rows = [
                [
                    str(chunk_index),
                    str(entry["calls"]),
                    f"{entry['seconds']:.3f}",
                    f"{entry['ready_at']:.3f}",
                ]
                for chunk_index, entry in ordered
            ]
            chunk_table = tabulate(
                chunk_rows,
                headers=["chunk", "forwards", "compute s", "ready at s"],
                tablefmt="heavy_outline",
                numalign="left",
                stralign="left",
            )
            summary_rows.insert(
                2, ["time to first chunk", f"{first_ready:.2f} s"]
            )

        summary_table = tabulate(
            summary_rows, headers=["metric", "value"],
            tablefmt="heavy_outline", numalign="left", stralign="left",
        )

        logger.info(
            "[benchmark] RTF result\n%s\n\nPhase breakdown\n%s%s",
            summary_table, phase_table,
            f"\n\nStreaming chunk profile\n{chunk_table}" if chunk_table else "",
        )

        payload = {
            "headline": headline,
            "phases": stats,
            "iterations": merged,
            "chunk_profile": chunk_profile,
        }
        out_dir = logger.iter_dir("benchmark")
        (out_dir / "rtf.json").write_text(json.dumps(payload, indent=2))
        markdown = [
            "# MiniMax-H3 inference RTF",
            "",
            f"- device: `{headline['device_type']}` x{headline['world_size']}, torch {headline['torch']}",
            f"- clip: {num_frames} frames @ {fps} fps = {video_seconds:.2f} s",
            f"- {len(merged)} measured iterations after {warmup} warmup",
            "",
            "## Summary",
            "",
            summary_table,
            "",
            "## Phase breakdown",
            "",
            phase_table,
        ]
        if chunk_table:
            markdown += ["", "## Streaming chunk profile", "", chunk_table]
        (out_dir / "rtf.md").write_text("\n".join(markdown) + "\n")
        logger.info("[benchmark] wrote %s", out_dir / "rtf.md")
        ops.barrier()
