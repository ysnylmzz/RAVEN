# MiniMax-H3 streaming inference on Ascend 950: what made it fast

This is the record of how one 8-second, 768×1376 clip went from **261 s to
21 s** on 8× Ascend 950: 12.6× faster end to end. It covers each change, what
it fixed, how it was measured, and how to switch it off to check. For how to
run things, see [README-NPU.md](README-NPU.md).

## Result

RTF is generation wall clock divided by video length (8.0 s). An RTF of 1
would be real time. Every row adds one change to the row before it.

| # | Change | RTF | Seconds per clip | Rollout | Decode | Time to first chunk | Peak memory | Speedup over previous row |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | Starting point: FSDP2, SDPA attention | 32.6 | 260.8 | 232.4 s | 27.9 s | 8.98 s | 48.0 GiB | — |
| 1 | Fused NPU attention | 11.6 | 93.0 | 63.8 s | 28.5 s | 5.63 s | 36.2 GiB | 2.8× |
| 2 | DiT replicated instead of FSDP-sharded | 9.0 | 72.0 | 42.8 s | 28.1 s | 2.66 s | 92.0 GiB | 1.3× |
| 3 | LoRA merged into the weights, contiguous KV merge | 7.8 | 62.1 | 32.4 s | 29.1 s | 1.74 s | 87.3 GiB | 1.16× |
| 4 | Video decode split across the SP group | 4.8 | 38.1 | 29.4 s | 8.0 s | 1.76 s | 87.3 GiB | 1.63× |
| 5 | One SP group of 8 cards instead of two of 4 | **2.6** | **20.6** | **15.4 s** | **4.6 s** | **1.16 s** | 87.3 GiB | 1.85× |

All rows: RTF benchmark, one prompt, 192 frames, 1 warmup and 2–4 measured
iterations, means across iterations. Encode (the text encoder, about 0.65 s)
is included in seconds per clip but not shown. Peak memory is the allocator's
`max_memory_allocated` on the busiest card.

Rows 0–4 ran two SP groups of 4 cards, and those runs slowed down iteration
by iteration (see [Measuring it](#measuring-it)). Their means therefore
include some slowdown that row 5 does not have. On the first iteration alone,
row 3 → 4 → 5 went 48.5 s → 32.7 s → 20.2 s per clip.

Where the time goes now: rollout 75%, decode 22%, encode 3%.

## What a clip costs

Generating a clip has three phases:

- **encode**: the Qwen3-VL text encoder reads the prompt, once.
- **rollout**: the 33B-parameter DiT generates the clip's latents causally, in
  12 chunks. Each chunk is denoised in 4 steps against a KV cache of the
  earlier chunks, which is 5 forwards per chunk and about 59 forwards per clip.
- **decode**: the video VAE turns latents into 192 frames, and the audio VAE
  produces the waveform.

Almost every change below makes one forward cheaper, or stops cards from
waiting on each other.

## Step 1: fused attention (rollout 232 s → 64 s)

**Problem.** On NPU the varlen attention fell back to PyTorch SDPA, which
materialises the full score matrix for every packed sequence. At 768×1376 that
dominated each forward and added about 55 GiB of transient memory.

**Change.** `utils/flash_attn.py` routes the cached rollout's non-causal varlen
attention to `torch_npu.npu_fusion_attention` (TND layout), which never builds
the score matrix (commit `2230d46`).

**Effect.** Rollout went from 232.4 s to 63.8 s (3.6×) and RTF from 32.6 to
11.6. A microbenchmark against SDPA on a ragged packing gave cosine 0.999997,
and the generated videos look the same.

**Check it.** `RAVEN_NPU_FUSED_ATTN=0` in the environment restores the SDPA
path. `python -m projects.minimax_h3.tools.bench_npu_attention` compares the
two kernels at production shapes.

## Step 2: replicate the DiT instead of sharding it (rollout 64 s → 43 s)

**Problem.** The base trial shards the DiT with FSDP2 across 8 cards and
reshards it after every forward. Every forward therefore all-gathered the full
~66 GB bf16 trunk again, which put a floor of about 0.9 s under each forward.
Host-side launch overhead was ruled out first: `TASK_QUEUE_ENABLE=2` changed
nothing.

**Change.** A 950 card has 128 GiB, enough to hold the whole trunk, so the 950
production trial (`trials/npu/npu_8card_950.yaml`) places the DiT whole on
every card (`fsdp: null`). The text encoder, which runs once per prompt, stays
sharded.

**Effect.** Rollout went from 63.8 s to 42.8 s and time to first chunk from
5.6 s to 2.7 s. The cost is memory: the peak rose from 36 to 92 GiB per card.

**Check it.** `npu_bench_rtf_8card_950_fsdp.yaml` is the sharded variant.

## Step 3: merge the LoRA and fix the KV cache merge (rollout 43 s → 32 s)

An op-level profile of one steady-state forward split roughly as follows:

| Share | What |
| --- | --- |
| 29% | base matmuls |
| 26% | the LoRA adapter: fp32 matmuls, casts and adds |
| 17% | FlashAttention |
| 11% | merging new keys and values into the KV cache |
| 6% | Ulysses all-to-all |

That profile led to two fixes.

**LoRA merge.** peft keeps the adapter in fp32 as separate low-rank matmuls on
every projection, so the 1.26 B adapter parameters cost a quarter of each
forward. A placement plugin (`common/plugin/lora_merge.py`, commit `734534a`)
folds `W + scaling·B·A` into the base weights once, in fp32, rounds to bf16 and
removes the adapter layers. The result differs from the unmerged path only by
bf16 rounding: on a test layer, merged and unmerged outputs agree to cosine
0.99999, and the merged one is as close to an fp32 reference as the unmerged
one. It refuses FSDP-sharded weights and models in training mode.

**KV merge.** After the Ulysses exchange, the new keys and values are strided
views into one fused buffer. `index_put` with a strided source takes a slow
path on Ascend: about 560 µs against 100 µs. One `.contiguous()` before the
merge fixes it (`causal_model_sp.py` and `causal_model.py`, commit `de303e0`).

**Effect.** Rollout went from 42.8 s to 32.4 s and time to first chunk to 1.74 s.
Peak memory dropped 4.7 GiB, which is the fp32 adapter. With the merge alone
switched off in the same build, RTF was 8.6 against 7.8, about 7 s more per
clip on every iteration. The KV fix is worth about 2 s of rollout, but that is
an estimate across two sessions, within the drift.

**Check it.** `models.backbone.placement.plugins.0.enabled false` runs the
adapter unmerged. The benchmark logs the backbone's parameter dtypes, and
after the merge the adapter entry is empty.

## Step 4: decode on every card of the group (decode 29 s → 8 s)

**Problem.** After step 3, decode was 47% of the wall clock. Every card of an
SP group holds the same latents after the rollout, but only the group's first
card decoded them while the other three waited.

**Change.** The released VAE can already split each temporal clip's spatial
decoder tiles across a group of cards, but it depended on sglang's process
group state, which this repo does not vendor. `tile_parallel_group(group)` in
`klvae.py` supplies a plain torch process group in its place (commit `d880595`).
Validation and the benchmark decode on the whole SP group; the first card still
writes the video. At 768×1376 a clip is 4 × 7 = 28 tiles, and one all-gather
per clip hands every card all of them. The output is the same as a one-card
decode: it was bit-identical in a 4-process CPU test.

**Effect.** Decode went from 28.9 s to 8.0 s (22.6 s → 6.8 s on the first
iteration), and RTF from 7.8 to 4.8. Peak memory did not change. Four cards
gave 3.3× rather than 4× because the audio decode is not split and the
per-clip gathers cost a little.

**Check it.** `validation.decode_parallel false` decodes on one card per group.

## Step 5: one SP group of 8 cards (rollout 29 s → 15 s)

**Problem.** The base trial uses Ulysses SP across groups of 4, so 8 cards form
two groups that each generate their own clip. For a single prompt, validation
pads the second group with an unsaved duplicate. One clip therefore only ever
used 4 cards, and the other 4 repeated the same work.

**Change.** `npu_8card_950.yaml` now uses `up_size: 8`. Each card holds 7 of
the 56 attention heads and an eighth of the sequence rows, and the decode
splits each clip's 28 tiles at most 4 per card. The model already supported
it: 56 heads and the 64-row packing alignment both divide by 8. No code
changed.

**Effect.** Measured in one session on the first iteration: rollout went from
25.5 s to 15.1 s (1.69×), decode from 6.9 s to 4.5 s, and time to first chunk
from 1.76 s to 1.16 s. A steady chunk now takes 1.41 s, or 0.28 s per forward.
RTF went from 4.8 to 2.6, and the wider all-to-all costs far less than it
saves.

**Trade-off.** Two groups of 4 generate two clips at a time. For a batch of
prompts the two layouts are about even: two groups took 33–47 s per pair of
clips (16.5–23.5 s each), and one group takes 20.6 s per clip. One group wins
on latency and is steadier, so it is the default. `npu_8card_950_sp4.yaml`
keeps two groups for batch work.

## Measuring it

**The benchmark.** `trials/npu/npu_bench_rtf_8card_950.yaml` swaps in an RTF
engine (`projects/minimax_h3/tools/benchmark_rtf.py`, commit `0b05cba`). It
times encode, rollout and decode separately, discards a warmup pass, records
when each chunk is ready, and takes the slowest card's time for each phase.
`benchmark.profile true` adds an op-level trace of one steady-state chunk.

**A timing bug fixed along the way.** Decode ran only on each group's first
card, so the other cards started the next iteration early and their encode
timer absorbed the wait. Encode, and the total, were overstated by about one
decode (encode read 25–37 s instead of 0.65 s). A barrier before each iteration
fixed it (commit `d376264`). Row 0's total was corrected from log timestamps;
its rollout was never affected.

**Iterations slow down with two SP groups.** Every two-group run got slower
iteration by iteration, by up to ~45% by the fourth. During those runs the
cards reached 98 °C, and `npu-smi` reported cards 0 and 3 as Critical. Yet the
one-group run stayed flat (20.2–21.1 s across four iterations), and a two-group
run straight after it drifted again. Both layouts keep all 8 cards busy, so
temperature alone may not explain it. Until that is settled, compare runs on
the same iteration index, measure A/B pairs in one session, and check
`npu-smi info` before starting.

## What did not help

- **`TASK_QUEUE_ENABLE=2` / `CPU_AFFINITY_CONF=1`.** No change to rollout or
  its variance. The host is not the bottleneck: the device is busy for about
  0.66 s of each ~0.7 s forward.
- **Overlapping decode with rollout, for total time.** Considered and not
  built. The cards are already close to fully busy during rollout, so decoding
  on them between chunks could hide at most 1–2 s of the decode. It would
  still help streaming latency (see below).

## What is left

Rollout is 75% of a clip, at about 0.28 s per forward on 8 cards.

1. **Profile the one-group build.** The breakdown above is from the two-group
   build before the LoRA merge. A fresh `benchmark.profile true` run shows
   what one forward is made of now.
2. **Find the cause of the two-group slowdown.** Run the one-group benchmark
   with `benchmark.iters 10`. If it stays flat, the two-group layout causes the
   drift; if it slows too, it is heat. Either way, the card temperatures need
   to go to the machine's administrators.
3. **Stream the decode.** Decoding each VAE clip as soon as its latents exist
   would let playback start a few seconds into a clip instead of after it. It
   would barely change total time.
4. **Split the audio decode.** It still runs whole on every card.
5. **Causal varlen attention** still uses the SDPA fallback. The production
   rollout does not use it.

## Where the changes live

| Branch | Commits | Steps |
| --- | --- | --- |
| `npu-perf-diagnostics` | `0b05cba` RTF benchmark, `2230d46` fused attention, `d376264` benchmark barrier | 1, measuring |
| `npu-rollout-opt` | `de303e0` KV merge, `734534a` LoRA merge plugin, `3ce8ca2` replicated 950 trial | 2, 3 |
| `npu-parallel-decode` | `d880595` group decode, `a0fc46d` and `89bae55` measurements | 4 |
| `npu-sp8` | `352a523` SP 8 trials, `5e88bef` SP 8 as the default | 5 |

Each branch builds on the one above it, so `npu-sp8` contains everything.
