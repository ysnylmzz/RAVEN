# Running on Huawei Ascend NPU

This document covers streaming inference for `projects/minimax_h3` on Ascend
NPUs. The main [README](README.md) describes the framework itself and the CUDA
path; everything here is additive and nothing in it changes CUDA behaviour.

Training on NPU is not covered. The validated path is validation-only
inference through `common.engine.BaseEngine`.

## Validated environment

| Component | Version |
| --- | --- |
| Hardware | Ascend 950 (128 GiB HBM per card), 8 cards |
| CANN | 9.1.0 |
| Container | `quay.io/ascend/vllm-omni:v0.31.0rc1-a5` |
| Python | 3.12.13 (from the image) |
| torch | 2.11.0 |
| torch_npu | 2.11.0.post3 |

Ascend 910B (64 GiB per card) is also supported, but the 33.1B DiT no longer
fits a single card without FSDP CPU offload. See
[Choosing a trial](#choosing-a-trial).

FlashAttention is not available on NPU. For the cached streaming rollout
(non-causal varlen attention), `utils/flash_attn.py` routes to
`torch_npu.npu_fusion_attention` in TND layout. Everything else falls back to
a per-document `scaled_dot_product_attention` path that materializes the score
matrix. `RAVEN_NPU_FUSED_ATTN=0` forces that fallback for the rollout too.
[Performance](#performance) has figures for both.

## Container

```sh
docker pull quay.io/ascend/vllm-omni:v0.31.0rc1-a5
```

```sh
docker run -it --name raven-npu \
  --privileged \
  --network host \
  --shm-size 64g \
  --ulimit memlock=-1 \
  --device /dev/davinci0 --device /dev/davinci1 \
  --device /dev/davinci2 --device /dev/davinci3 \
  --device /dev/davinci4 --device /dev/davinci5 \
  --device /dev/davinci6 --device /dev/davinci7 \
  --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /usr/local/dcmi:/usr/local/dcmi:ro \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  -v /path/to/weights:/weights \
  -v /path/to/RAVEN:/workspace/RAVEN \
  quay.io/ascend/vllm-omni:v0.31.0rc1-a5 bash
```

Two flags matter more than they look:

- **Every `--device /dev/davinciN` you intend to use.** Passing only
  `davinci0` leaves the other cards invisible to the container even though
  `npu-smi` on the host lists them. `--privileged` happens to expose them all,
  which masks the mistake until you try to run on 8 ranks.
- **`--ulimit memlock=-1`** is required if you ever enable FSDP
  `cpu_offload`, which pins host memory. Without it an 8-rank run dies in
  `pin_memory()`. See [Troubleshooting](#troubleshooting).

Verify the cards are visible inside the container before going further:

```sh
npu-smi info
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python -c "import torch, torch_npu; print(torch.npu.device_count())"
```

## Python dependencies

The image already ships a working `torch` + `torch_npu` pair. **Do not install
the pinned `torch==2.11.0+cpu` / `torch_npu` lines over it** — that replaces the
image's build and NPU support stops working. Install only the rest:

```sh
cd /workspace/RAVEN
grep -vE '^(torch|torchvision|torch_npu)==' tools/requirements-npu-inference.txt \
  > /tmp/raven-npu-deps.txt
pip install -r /tmp/raven-npu-deps.txt
```

On a bare-metal host without the container, build the environment from
scratch instead:

```sh
conda env create -f tools/environment-npu.yaml   # optional
bash tools/prepare_venv_npu.sh
```

`tools/prepare_venv_npu.sh` sources CANN's `set_env.sh`, creates `venv/`, and
installs `tools/requirements-npu-inference.txt` including the torch pins. Both
conda and the venv are optional: the launcher uses whichever Python is on
`PATH` when neither is present, which is what the container route relies on.

## Weights

Download the four MiniMax-H3 components plus the RAVEN LoRA, then point the
trial at them. The shipped paths are absolute and must be edited — there are
six of them in
`projects/minimax_h3/trials/base/causal_minimax_h3_base/minimax_h3_raven_streaming_lora_4nfe_preview.yaml`:

| Line | Key | Component |
| --- | --- | --- |
| 74 | `tokenizer_path` | FL2VA tokenizer |
| 102 | `path` | DiT shard index (`model.safetensors.index.json`) |
| 119 | `weight` | RAVEN streaming LoRA |
| 148 | `pretrained_model_name_or_path` | Qwen3-VL-32B text encoder |
| 170 | `path` | Video VAE |
| 195 | `path` | Audio VAE |

Alternatively override them on the command line without editing the file, as
`KEY VALUE` pairs — see [Running a trial](README.md#running-a-trial).

**Put the weights on local disk if you can.** Eight ranks reading ~66 GB of
DiT safetensors off one shared mount is the single largest source of startup
skew, and that skew is what trips the HCCL barrier timeout described in
[Troubleshooting](#troubleshooting).

## Sanity check

Before a full run, confirm that autocast is keyed to the right device type:

```sh
python -m projects.minimax_h3.tools.check_autocast --dtype float16
```

All checks should report OK. This exercises the fp32 "islands" that the model
creates with `autocast(enabled=False)`; if they were still hardcoded to
`cuda`, they would silently do nothing on NPU.

## Running

The standard launcher works unchanged. It detects `npu-smi` and switches to
the NPU path on its own:

```sh
N=<ranks> bash tools/multi_run.sh <trial.yaml> [KEY VALUE ...]
```

On the NPU path it additionally sources CANN's `set_env.sh`, asserts that
`torch_npu` imports, derives `N` from `torch.npu.device_count()`, and sets
`HCCL_CONNECT_TIMEOUT=1800` (the 120 s default is far too short while 30B+
models load) and `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`.

### Choosing a trial

| Trial | Cards | What it is for |
| --- | --- | --- |
| `trials/npu/npu_4a_single_card.yaml` | 1 | Smoke test on a 64 GiB card. DiT and text encoder keep FSDP on a 1-rank mesh purely for CPU offload. 384×672. |
| `trials/npu/npu_4a_single_card_950.yaml` | 1 | Same, for a 128 GiB card: the DiT is placed whole, with no FSDP and no collectives at all. |
| `trials/npu/npu_4b_fsdp8_no_sp.yaml` | 8 | FSDP2 across 8 cards with Ulysses SP off. 384×672 — keep it there, see the note below. |
| `trials/npu/npu_8card_950.yaml` | 8 | **Production on Ascend 950.** The base production trial with the DiT replicated on every card instead of FSDP-sharded, the LoRA merged into the base weights, and no CPU offload. 92 GiB peak per card. |
| `trials/base/.../minimax_h3_raven_streaming_lora_4nfe_preview.yaml` | 8 | Base production trial: 768×1376, FSDP2 `shard_size: 8` and SP `up_size: 4`. On a 950 it is slower than the trial above; use it where the DiT does not fit a card. |

Single card:

```sh
ASCEND_RT_VISIBLE_DEVICES=0 N=1 bash tools/multi_run.sh \
  projects/minimax_h3/trials/npu/npu_4a_single_card_950.yaml
```

Eight cards, production settings on Ascend 950:

```sh
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
HCCL_NPU_SOCKET_PORT_RANGE=auto \
N=8 bash tools/multi_run.sh projects/minimax_h3/trials/npu/npu_8card_950.yaml
```

Under FSDP2 every chunk forward all-gathers the whole ~66 GB bf16 DiT trunk
again, which puts a floor of about 0.9 s under each forward. A 128 GiB card
holds the trunk outright, so this trial replicates it and keeps only Ulysses SP.
Do not add `models.backbone.placement.fsdp.*` overrides to it: they recreate an
fsdp node, and the LoRA merge refuses sharded weights.

The LoRA merge (`common.plugin.lora_merge`) folds the adapter into the base
weights once, after placement. peft keeps the adapter in fp32 as separate
matmuls, and on Ascend that path took about a quarter of every forward. The
merged weights differ from the unmerged path only by bf16 rounding. Append
`models.backbone.placement.plugins.0.enabled false` to run the adapter unmerged.

The FSDP-sharded build, for comparison or where the DiT does not fit a card:

```sh
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
HCCL_NPU_SOCKET_PORT_RANGE=auto \
N=8 bash tools/multi_run.sh \
  projects/minimax_h3/trials/base/causal_minimax_h3_base/minimax_h3_raven_streaming_lora_4nfe_preview.yaml \
  models.backbone.placement.fsdp.cpu_offload false \
  models.text_encoder.placement.fsdp.cpu_offload false
```

On a 128 GiB card, turning `cpu_offload` off is both faster and necessary
unless the container was started with `--ulimit memlock=-1`. Static footprint
with it off is 24.7 GiB per card, so there is ample room.

> **Do not raise the resolution on `npu_4b_fsdp8_no_sp.yaml`.** With SP off
> every rank carries the whole sequence, and the SDPA fallback's score matrix
> grows with the square of the pixel count: 384×672 costs about 0.8 GiB,
> 768×1376 about 14.9 GiB plus fp32 softmax intermediates, which exceeds
> 120 GiB per card. Use the production trial for full resolution — it has SP
> enabled.

### Output

Per-rank logs are written to `<run_dir>/logs/log_rank{RANK}.txt` in append
mode, so they accumulate across runs. Only local rank 0 writes to the console.
Generated media lands under
`<run_dir>/<namespace>/<step:07d>/validation/backbone/`.

## Picking frames, resolution, and `max_seqlen`

Three constraints:

- `validation.num_frames` must be `17n + 5` (192, 379, 549, 719, …).
- `validation.height` and `validation.width` must be multiples of 32.
- `data.args.max_seqlen` must be large enough for the packed sequence. It is
  **not** derived from the validation size — the validation packer overrides
  height, width, and frame count but keeps `max_seqlen` from `data.args`.

Compute it as:

```
latent_t   = ((num_frames - 5) // 17) * 5 + 2
video_rows = latent_t * (height // 32) * (width // 32)
max_seqlen = video_rows + audio_rows + text_tokens
```

The shipped `max_seqlen: 59464` is exactly sized for 768×1376×192: 58,824
video rows plus 640 for audio and text. Keep comparable headroom when you
change the geometry. If you get it wrong, `_validation_inputs` asserts and
prints the row count it actually needed, so one failed run tells you the
answer.

Long video is cheap. Streaming fixes each chunk at 5 video latents and bounds
the KV cache by `sink` + `window_size`, so **memory does not grow with video
length** — only time does, and close to linearly. Doubling 192 frames to 379
doubled the chunk count from 11 to 22, cost 2.11× the wall time, and left the
static footprint at 24.7 GiB unchanged.

```sh
# 379 frames (15.8 s at 24 fps) on 8 cards
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 HCCL_NPU_SOCKET_PORT_RANGE=auto \
N=8 bash tools/multi_run.sh \
  projects/minimax_h3/trials/base/causal_minimax_h3_base/minimax_h3_raven_streaming_lora_4nfe_preview.yaml \
  models.backbone.placement.fsdp.cpu_offload false \
  models.text_encoder.placement.fsdp.cpu_offload false \
  validation.num_frames 379 validation.num_prompts 1 \
  data.args.max_seqlen 118000 data.args.max_seqlen_per_sample 118000
```

## Performance

Measured on 8× Ascend 950, CANN 9.1.0, 768×1376, `cpu_offload false`.

| Run | Frames | Video length | Rollout + decode | Static memory |
| --- | --- | --- | --- | --- |
| 1 prompt | 192 | 8.0 s | 4 min 25 s | 24.7 GiB |
| 1 prompt | 379 | 15.8 s | 9 min 19 s | 24.7 GiB |
| 4 prompts, 2 SP groups | 192 | 8.0 s each | 9 min 35 s | 24.7 GiB |

Model build adds roughly 110–130 s on top, dominated by reading DiT weights.

Static memory by stage, from `log_rank0.txt`: backbone 8.3 GiB → text encoder
14.5 → video VAE 24.2 → audio VAE 24.7. Transient peaks add roughly 55 GiB of
SDPA score matrices on top, and `npu-smi` reports around 80 GB because the
caching allocator's reserved figure is a high-water mark that is never
returned to the driver.

The figures above are from the SDPA fallback, before the fused attention path
existed; its score matrices were the bottleneck.

### Optimization steps

RTF benchmark, 1 prompt, 192 frames (8.0 s of video), 1 warmup and 2–4
measured iterations. RTF is generation wall clock divided by video length.
Peak memory is the allocator's `max_memory_allocated`, not `npu-smi`.

| Build | RTF | Rollout | Decode | Time to first chunk | Peak memory |
| --- | --- | --- | --- | --- | --- |
| FSDP, SDPA attention | 32.6 | 232.4 s | 27.9 s | 8.98 s | 48.0 GiB |
| FSDP, fused attention | 11.6 | 63.8 s | 28.5 s | 5.63 s | 36.2 GiB |
| Replicated DiT, fused attention | 9.0 | 42.8 s | 28.1 s | 2.66 s | 92.0 GiB |
| + LoRA merged, contiguous KV merge (`npu_8card_950.yaml`) | not yet measured | | | | |

The figures drift with card temperature. Under sustained load the cards in the
validated machine reached 98 °C and every iteration after the first ran
slower. The replicated build's first measured iteration took 35.0 s of
rollout (RTF 7.2), and the later ones 43–47 s. Compare runs on the same
iteration index, and watch `npu-smi info` while measuring.

An op-level profile of one steady-state forward on the replicated build,
before the last row's changes, split roughly as follows. Base matmuls took 29%
and the fp32 LoRA path (matmuls, casts and adds) 26%. FlashAttention took 17%,
the KV cache merge 11%, and the Ulysses all-to-all 6%. Decode runs the
unsharded VAE on one card per SP group and is now about a third of the
wall clock.

### Finding where the time goes

All three of these tools run on the real model build or at its real shapes:

- **Attention microbenchmark.** `python -m projects.minimax_h3.tools.bench_npu_attention`
  checks the fused kernel against SDPA on a ragged packing, then times both
  at the production per-rank shapes as the KV cache fills.
- **Phase and chunk breakdown.**
  `trials/npu/npu_bench_rtf_8card_950.yaml` is `npu_8card_950.yaml` with the
  RTF engine. It splits encode / rollout / decode, discards a warmup pass,
  and logs the backbone's parameter dtypes, LoRA adapter separately (empty
  once merged). Its header lists the one-variable A/B runs.
  `npu_bench_rtf_8card_950_fsdp.yaml` is the FSDP-sharded variant.
- **Op-level trace.** Add `benchmark.profile true` to wrap one steady-state
  chunk in the `torch_npu` profiler. The trace is written under
  `benchmark/profile/rank0/` and opens in MindStudio Insight.

## What the port changes

| File | Change |
| --- | --- |
| `common/distributed/ops.py` | `get_device_type()` / `get_device_module()` / `get_device()` / `get_process_group_backend()`. The process group backend becomes the composite `cpu:gloo,npu:hccl`. |
| `projects/minimax_h3/**` | `torch.autocast` regions keyed to the active device type instead of a hardcoded `cuda`, so the fp32 islands survive. |
| `projects/minimax_h3/modeling/audio_vae/alias_free.py` | Convolutions chunked when their input exceeds CANN's 1,000,000-element width limit. |
| `utils/__init__.py` | Makes `utils` a regular package so `utils.flex_attn` resolves. |
| `tools/multi_run.sh` | NPU branch: CANN env, `torch_npu` assertion, HCCL timeout, allocator config. conda and `venv/` made optional. |
| `tools/prepare_venv_npu.sh`, `tools/environment-npu.yaml`, `tools/requirements-npu-inference.txt` | NPU environment build. |
| `projects/minimax_h3/tools/check_autocast.py` | Verifies autocast device keying. |
| `utils/flash_attn.py` | Non-causal varlen attention routed to `torch_npu.npu_fusion_attention` (TND layout) on NPU. |
| `projects/minimax_h3/modeling/transformer/causal_model*.py` | The cached rollout makes the new K/V contiguous before merging them with the cache; strided sources sent `index_put` down a slow path. |
| `common/plugin/lora_merge.py` | Placement plugin that merges a LoRA adapter into the base weights for inference and unloads it. |
| `projects/minimax_h3/tools/` | `bench_npu_attention` (attention microbenchmark) and `benchmark_rtf` (RTF engine). |
| `projects/minimax_h3/trials/npu/` | Bring-up trials, the 950 production trial and the RTF benchmark trials. |

### The audio VAE width limit

Worth knowing about because it only appears on long video. CANN rejects a
convolution whose input is wider than 1,000,000 elements. The final
`Activation1d` in the BigVGAN decoder upsamples the 32 kHz waveform by 2
before filtering it back down, so the low-pass filter crosses that limit at
**375 frames** (15.6 s at 24 fps) — a 379-frame clip misses it by 1%.

`alias_free.py` now splits both the low-pass convolution and the upsample's
transposed convolution into chunks when needed. The chunking is exact, not an
approximation: the forward convolution's chunks are aligned to output indices
and concatenated, while the transposed convolution's chunks overlap by
`kernel_size - stride` and are summed.

## Troubleshooting

**`ModuleNotFoundError: No module named 'utils.flex_attn'; 'utils' is not a package`**
A stray `utils.py` somewhere on `sys.path` shadows the repository's `utils`
package. Confirm with `python -c "import utils; print(utils.__file__)"` and
remove or rename the offending file.

**`torch.OutOfMemoryError` in `pin_memory()`, `aclrtMallocHostWithCfg` (207001), `rtsMallocHost ... out of memory`**
This is **host** pinned memory, not device memory. FSDP `cpu_offload` needs
roughly 116 GiB pinned across 8 ranks. Either disable it
(`models.backbone.placement.fsdp.cpu_offload false`, likewise for
`text_encoder`) or restart the container with `--ulimit memlock=-1`. On a
128 GiB card, disabling it is the right answer.

**`Execution_Error_Stream_Synchronize_Timeout(EE1002)` / `Stream synchronize timeout ... timeout=65000ms` at the first barrier**
The first collective after model building is
`dist.barrier()` in `common/distributed/unified_parallel.py`. If one rank is
more than 65 s behind the others in loading weights, CANN's device-side stream
sync timeout fires there. This has nothing to do with the generation settings —
it happens before any frame is produced. Re-run first, since the page cache may
now be warm; if it recurs, stage the weights on local disk, or pre-warm them
with `cat /path/to/weights/**/*.safetensors > /dev/null`. Note that
`HCCL_CONNECT_TIMEOUT` (which the launcher already raises) and the 30-minute
process group timeout govern different things and will not help.

**Only one card is visible, or ranks fail to start**
Check that every `/dev/davinciN` was passed to `docker run`, and set
`ASCEND_RT_VISIBLE_DEVICES` explicitly. `N` is derived from
`torch.npu.device_count()`, which honours that variable and counts both dies
of a dual-die card.

**`npu-smi` shows far more memory in use than the logs report**
Expected. The logs report the allocator's *allocated* figure; `npu-smi` sees
*reserved*, a high-water mark the caching allocator never returns.

**Out of memory at higher resolution on an 8-card run**
Check whether SP is on. With `up_size: null` every rank holds the whole
sequence and the SDPA score matrix scales with the square of the pixel count.
Use the production trial rather than raising the resolution on
`npu_4b_fsdp8_no_sp.yaml`.

## Known gaps

- `npu_fusion_attention` covers only the non-causal cached rollout. Causal
  varlen attention still uses the SDPA fallback.
- The LoRA merge and the contiguous KV merge in `npu_8card_950.yaml` have not
  yet been measured on hardware, nor has its video been compared with the
  unmerged build.
- Decode runs on one card per SP group while the others idle; it is about a
  third of the wall clock on the 950 build.
- FSDP `CPUOffloadPolicy` across multiple ranks is unverified — it needs
  `--ulimit memlock=-1` and has not been run to completion.
- Training on NPU is untested.
