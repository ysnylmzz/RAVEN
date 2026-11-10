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

`tools/download_minimax_h3_weights.sh` fetches everything the streaming trials
need, about 139 GiB, from two public Hugging Face repos: the FL2VA DiT, text
encoder, video VAE and tokenizer plus the audio VAE from
[MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) (MiniMax-H3
Community License), and the preview LoRA from
[mvp-lab/MiniMax-H3-RAVEN-Streaming-LoRA](https://huggingface.co/mvp-lab/MiniMax-H3-RAVEN-Streaming-LoRA).
It checks the free space first, resumes when re-run, and verifies every shard
at the end.

```sh
bash tools/download_minimax_h3_weights.sh                 # into /root/models
bash tools/download_minimax_h3_weights.sh /data/weights   # anywhere else
HF_ENDPOINT=https://hf-mirror.com bash tools/download_minimax_h3_weights.sh   # through a mirror
```

Under `/root/models` the shipped trials find the weights without changes.
Anywhere else, the script writes the six path overrides to
`<dir>/raven_minimax_h3_overrides.txt`; append them to the launch command:

```sh
N=8 bash tools/multi_run.sh projects/minimax_h3/trials/npu/npu_8card_950.yaml \
  $(cat /data/weights/raven_minimax_h3_overrides.txt)
```

The paths are absolute, and these are the six of them in
`projects/minimax_h3/trials/base/causal_minimax_h3_base/minimax_h3_raven_streaming_lora_4nfe_preview.yaml`:

| Line | Key | Component |
| --- | --- | --- |
| 74 | `tokenizer_path` | FL2VA tokenizer |
| 102 | `path` | DiT shard index (`model.safetensors.index.json`) |
| 119 | `weight` | RAVEN streaming LoRA |
| 148 | `pretrained_model_name_or_path` | Qwen3-VL-32B text encoder |
| 170 | `path` | Video VAE |
| 195 | `path` | Audio VAE |

Edit them there, or override them on the command line as `KEY VALUE` pairs
(see [Running a trial](README.md#running-a-trial)), which is what the
overrides file does.

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
| `trials/npu/npu_8card_950.yaml` | 8 | **Production on Ascend 950.** The base production trial with one SP group across all 8 cards, the DiT replicated on every card instead of FSDP-sharded, the LoRA adapter unmerged in bf16, and no CPU offload. RTF 2.7: an 8 s clip in 21.5 s. 90 GiB peak per card. |
| `trials/npu/npu_4card_950.yaml`, `npu_2card_950.yaml`, `npu_1card_950.yaml` | 4, 2, 1 | The production trial on fewer 950 cards. Not yet run; see [Other card counts and card types](#other-card-counts-and-card-types). |
| `trials/npu/npu_8card_64g.yaml`, `npu_4card_64g.yaml` | 8, 4 | For 64 GiB cards (910B): DiT FSDP-sharded, one SP group. Not yet run on 910B. |
| `trials/npu/npu_8card_950_sp4.yaml` | 8 | The same with two SP groups of 4, which run two prompts at a time. A single clip takes about twice as long (RTF 4.8). |
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
holds the trunk outright, so this trial replicates it and keeps only Ulysses SP,
across all 8 cards.

Measured at SP 8 in one session (`npu_bench_rtf_8card_950_fsdp.yaml` against
the replicated baseline, adapter in fp32 on both sides):

| DiT placement | RTF | Seconds per clip | Rollout | Per forward | Time to first chunk | Peak memory |
| --- | --- | --- | --- | --- | --- | --- |
| Replicated | 3.0 | 24.3 | 18.9 s | 0.34 s | 1.49 s | 92.0 GiB |
| FSDP2 across 8 cards | 7.8 | 62.1 | 57.0 s | 0.95 s | 5.58 s | 33.9 GiB |

FSDP saves 58 GiB per card and costs 2.6× the wall clock. Its per-forward time
is the all-gather, not compute: each card receives about 58 GiB of weights per
forward (7/8 of the bf16 trunk and the fp32 adapter), and the forward's compute
takes a third of that time. Each card processes only ~680 rows of a chunk
under SP 8, too few to hide the gather; the same holds for LLM decoding, which
is why LLM inference engines shard with tensor parallelism instead. Use the
replicated trial whenever the DiT fits.
Do not add `models.backbone.placement.fsdp.*` overrides to it: they recreate an
fsdp node, and the LoRA cast plugin refuses sharded weights.

peft keeps the LoRA adapter in fp32 as separate matmuls, and on Ascend that
path took about a quarter of every forward. `common.plugin.lora_merge` has two
ways to cut it:

- `MergeLoraAfterPlacement` folds the adapter into the bf16 base weights. It is
  the fastest, but it breaks long clips: with it, 15.8 s and 40.6 s clips
  degraded from about 8 s on (the trained length), while the same 15.8 s clip
  with the adapter unmerged stayed clean. Rounding `W + B·A` to bf16 drops every
  delta element smaller than half a bf16 step of `W`, so the merged model is a
  fixed perturbation of the trained one. Use it only for clips of 8 s or less,
  if at all.
- `CastLoraAfterPlacement` keeps the adapter unmerged and casts it to bf16, so
  the weights stay exact and only the adapter branch's arithmetic is rounded.
  The production trial uses it. 15.8 s and 40.6 s clips with it stayed clean,
  and it recovers about three quarters of the merge's speed. Append
  `models.backbone.placement.plugins.0.enabled false` for the fp32 adapter.

Measured in one session with the RTF benchmark (192 frames, 4 iterations):

| LoRA adapter | RTF | Seconds per clip | Rollout | Steady chunk | Time to first chunk | Peak memory | Clips past 8 s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fp32, unmerged (peft default) | 3.0 | 24.2 | 18.8 s | 1.69 s | 1.49 s | 92.0 GiB | clean |
| **bf16, unmerged (production)** | **2.7** | **21.5** | **16.3 s** | **1.48 s** | **1.31 s** | **89.7 GiB** | **clean** |
| merged into bf16 weights (earlier session) | 2.6 | 20.6 | 15.4 s | 1.41 s | 1.16 s | 87.3 GiB | degraded from 8 s |

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

### Other card counts and card types

Every trial is written for a fixed number of cards. To run on another server,
pick the trial for its card memory and count from the table below, or adapt
one using the rules after it. Only the 8-card Ascend 950 rows have been run.

| Card memory | Cards | Trial | DiT placement | Status |
| --- | --- | --- | --- | --- |
| 128 GiB (950) | 8 | `npu_8card_950.yaml` | replicated, SP 8 | measured: RTF 2.7, 89.7 GiB peak |
| 128 GiB | 16 (2 nodes × 8) | `npu_8card_950.yaml`, unchanged | replicated, two SP groups of 8 | not run |
| 128 GiB | 4 | `npu_4card_950.yaml` | replicated, SP 4 | not run; ~87 GiB static |
| 128 GiB | 2 | `npu_2card_950.yaml` | replicated, SP 2 | not run; ~99 GiB static, tight |
| 128 GiB | 1 | `npu_1card_950.yaml` | whole, no SP; text encoder offloaded to host | not run; ~75 GiB static |
| 64 GiB (910B) | 8 | `npu_8card_64g.yaml` | FSDP across 8, SP 8 | same placement measured on 950: RTF 7.8, 33.9 GiB peak |
| 64 GiB | 4 | `npu_4card_64g.yaml` | FSDP across 4, SP 4 | not run; ~39 GiB static |
| 64 GiB | 1 | `npu_4a_single_card.yaml` | FSDP on one card with CPU offload, 384×672 | smoke test only; no full-resolution trial for 1–2 cards |

Each trial's header gives its memory estimate and the command to run it.

**The rules a trial has to satisfy:**

- **SP size is 1, 2, 4 or 8.** Ulysses SP splits the 56 attention heads, and
  the packed sequence is aligned to 64 rows without padding, so the size must
  divide both. 16 is not possible; on 16 cards use two groups of 8.
- **Set the SP size in three places at once:** `distributed.up_size` (a list),
  `engine.up_size.train` and `engine.up_size.validation`. `null` everywhere
  turns SP off.
- **The world size is a multiple of the SP size.** Consecutive ranks form a
  group, and each group generates its own clip, so 16 ranks with SP 8 make two
  clips at a time. Keep a group inside one node: SP exchanges data in every
  attention layer.
- **FSDP `shard_size` divides the world size.** The production trials shard
  the text encoder with `shard_size: 8`, so they fail on fewer cards with
  `shard_size 8 must divide world size 4`. The 4-, 2- and 1-card trials set it
  to the card count. A `shard_size` smaller than the world shards within
  groups of that size and replicates across them.
- **Replicate the DiT only where it fits.** The DiT and its bf16 adapter take
  64.1 GiB on every card, the two VAEs another 10.3 GiB. That needs a 128 GiB
  card. On 64 GiB cards the DiT has to be FSDP-sharded, which costs about 2.6×
  the time (see the FSDP comparison above).

**Estimating memory per card.** Static memory is what the models take:

| Component | Replicated | FSDP-sharded across *n* cards |
| --- | --- | --- |
| DiT, bf16 | 61.7 GiB | 61.7 / *n* GiB |
| LoRA adapter | 2.4 GiB (bf16) | 4.7 / *n* GiB (fp32) |
| Text encoder, bf16 | — | 49.4 / *n* GiB, or ~0 offloaded to host |
| Video VAE, fp32 | 9.7 GiB | 9.7 GiB |
| Audio VAE, fp32 | 0.6 GiB | 0.6 GiB |

During the rollout, add the KV cache: about 25 GiB divided by the SP size
(3.2 GiB at SP 8, 25 GiB without SP), constant at any video length. Decoding
adds a few GiB more; clips too long to decode on the card go to host memory
on their own. Leave a few GiB for HCCL and CANN as well: a 128 GiB card shows
about 123 GiB to PyTorch.

**16 cards on two nodes.** The launcher takes the usual torchrun variables.
Run the same command on both nodes, with the weights and the repository at the
same paths on each:

```sh
# on node 0 (node 1: NODE_RANK=1, its own LOCAL_ADDR)
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 HCCL_NPU_SOCKET_PORT_RANGE=auto \
NNODES=2 NODE_RANK=0 MASTER_ADDR=<node 0 IP> MASTER_PORT=7890 \
LOCAL_ADDR=<this node's IP> \
N=8 bash tools/multi_run.sh projects/minimax_h3/trials/npu/npu_8card_950.yaml
```

Set `LOCAL_ADDR` explicitly: inside a container the hostname often resolves to
127.0.0.1, which the other node cannot reach. If HCCL picks the wrong network
interface, name it with `HCCL_SOCKET_IFNAME`. This setup has not been run.

**Other Ascend cards.** Check three things before the first run: the memory
`npu-smi info` reports per card, the count `torch.npu.device_count()` returns
(it counts both dies of a dual-die card, and each die is a rank), and whether
`torch_npu.npu_fusion_attention` exists in the installed `torch_npu`. Without
it the rollout falls back to SDPA, which is much slower and needs far more
memory; `RAVEN_NPU_FUSED_ATTN=0` reproduces that path on a card that has it.

**Bringing up a new setup.** Go from cheap to expensive, so a failure is easy
to place:

1. `npu-smi info` shows the cards idle, and
   `python -m projects.minimax_h3.tools.check_autocast --dtype float16` passes.
2. The chosen trial with `validation.num_prompts 1`: one 8 s clip. Check it
   finishes and the video looks right.
3. The RTF benchmark for speed. Any trial becomes a benchmark by swapping its
   engine; the `benchmark.*` settings are optional (1 warmup, 3 iterations):

   ```sh
   N=4 bash tools/multi_run.sh projects/minimax_h3/trials/npu/npu_4card_950.yaml \
     entry.module projects.minimax_h3.tools.benchmark_rtf \
     entry.class_name RTFBenchmarkEngine validation.num_prompts 1
   ```
4. A clip longer than 8 s (see
   [Picking frames, resolution, and `max_seqlen`](#picking-frames-resolution-and-max_seqlen)).

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
the KV cache by `sink` + `window_size`, so **the rollout's memory does not grow
with video length** — only time does, and close to linearly. Doubling 192
frames to 379 doubled the chunk count from 11 to 22, cost 2.11× the wall time,
and left the static footprint at 24.7 GiB unchanged.

The decoded video does grow with length: 3 × frames × 768 × 1376 fp32 values,
17 GiB for 60 s. Kept on the card it is copied several times over on the way
to the mp4, and a 60 s clip ran out of memory on top of the replicated DiT's
78 GiB. So when the whole clip would not fit, the decoder writes each finished
temporal clip straight into host memory, and only the rank that saves the
video keeps it (`validation.decode_to_host`: unset decides by size, `true` or
`false` forces it). Device memory then stays flat at any length. The cost
moves to host RAM on the saving rank: about 4–5 copies of the clip while the
mp4 is written, so roughly 50 GB for 40 s and 80 GB for 60 s.

With `npu_8card_950.yaml`, a 40.6 s clip (974 frames) came out clean end to
end. Rollout and decode took 1 min 59 s after the models were built, and the
video was saved 25 s later, 2 min 24 s in all. The 974 frames went to host
memory (11.5 GiB). 60.4 s has not been run since the host decode was added.

| Length | `validation.num_frames` | `max_seqlen` |
| --- | --- | --- |
| 8.0 s | 192 | 59464 (shipped) |
| 15.8 s | 379 | 118000 |
| 40.6 s | 974 | 302000 |
| 60.4 s | 1450 | 448000 |

```sh
# 974 frames (40.6 s at 24 fps) on 8 cards
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 HCCL_NPU_SOCKET_PORT_RANGE=auto \
N=8 bash tools/multi_run.sh projects/minimax_h3/trials/npu/npu_8card_950.yaml \
  validation.num_frames 974 validation.num_prompts 1 \
  data.args.max_seqlen 302000 data.args.max_seqlen_per_sample 302000
```

```sh
# 379 frames (15.8 s at 24 fps) on 8 cards
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 HCCL_NPU_SOCKET_PORT_RANGE=auto \
N=8 bash tools/multi_run.sh projects/minimax_h3/trials/npu/npu_8card_950.yaml \
  validation.num_frames 379 validation.num_prompts 1 \
  data.args.max_seqlen 118000 data.args.max_seqlen_per_sample 118000
```

## Performance

Measured on 8× Ascend 950, CANN 9.1.0, 768×1376, `cpu_offload false`.

Base trial (FSDP2, two SP groups of 4, SDPA attention, before the
optimizations below):

| Run | Frames | Video length | Rollout + decode | Static memory |
| --- | --- | --- | --- | --- |
| 1 prompt | 192 | 8.0 s | 4 min 25 s | 24.7 GiB |
| 1 prompt | 379 | 15.8 s | 9 min 19 s | 24.7 GiB |
| 4 prompts, 2 SP groups | 192 | 8.0 s each | 9 min 35 s | 24.7 GiB |

Production trial (`npu_8card_950.yaml`):

| Run | Frames | Video length | Rollout + decode | Peak memory |
| --- | --- | --- | --- | --- |
| 1 prompt, RTF benchmark | 192 | 8.0 s | 20.8 s | 89.7 GiB |
| 1 prompt | 974 | 40.6 s | 1 min 59 s | decoded to host memory |

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
| + LoRA merged, contiguous KV merge | 7.8 | 32.4 s | 29.1 s | 1.74 s | 87.3 GiB |
| + decode split across the SP group | 4.8 | 29.4 s | 8.0 s | 1.76 s | 87.3 GiB |
| + one SP group of 8 cards instead of two of 4 | 2.6 | 15.4 s | 4.6 s | 1.16 s | 87.3 GiB |
| LoRA unmerged in bf16 instead of merged (`npu_8card_950.yaml`) | 2.7 | 16.3 s | 4.5 s | 1.31 s | 89.7 GiB |

Every row above the last two ran two SP groups of 4. Each row adds one change to
the row before it. [NPU-OPTIMIZATION.md](NPU-OPTIMIZATION.md) walks through the
changes and how each was measured.

With two SP groups the figures drift. Under sustained load the cards in the
validated machine reached 98 °C, and every iteration after the first ran
slower. The replicated build's first measured iteration took 35.0 s of
rollout (RTF 7.2), and the later ones 43–47 s. With the LoRA merged, rollout
went from 25.3 s on the first iteration to 37.6 s on the fourth, and decode
from 22.5 s to 31.7 s. One SP group of 8 did not drift: its four iterations
took 20.2–21.1 s, and the two-group run straight after it drifted from 33.0 s
to 47.1 s again. Both keep all 8 cards busy, so temperature alone may not
explain the drift. Compare runs on the same iteration index, and watch
`npu-smi info` while measuring.

The same build with the merge turned off measured RTF 8.6,
rollout 39.8 s and peak memory 92.0 GiB. Its total was about 7 s higher on
every iteration, so the merge alone saves about 18% of rollout and the
4.7 GiB of the fp32 adapter. The merge was later found to degrade clips
longer than 8 s. The last row replaces it with the bf16 adapter, which costs
about 0.9 s per clip against it; see the LoRA notes under
[Choosing a trial](#choosing-a-trial).

An op-level profile of one steady-state forward on the replicated build,
before the last row's changes, split roughly as follows. Base matmuls took 29%
and the fp32 LoRA path (matmuls, casts and adds) 26%. FlashAttention took 17%,
the KV cache merge 11%, and the Ulysses all-to-all 6%.

After the merge, decode was 47% of the wall clock: it ran the unsharded VAE on
one card per SP group while the other three waited. The last row decodes on
all four. At 768x1376 each temporal clip is 4 x 7 = 28 decoder tiles, so each
card decodes 7 and one all-gather per clip hands every card all 28. The
output is the same as decoding on one card. `validation.decode_parallel false`
restores the one-card decode.

With `up_size: 4` the 8 cards form two SP groups, and each generates its own
clip. For a single prompt the second group generates an unsaved duplicate
(validation pads every group to the same number of prompts), so one clip only
ever uses 4 cards. `npu_8card_950.yaml` now puts all 8 in one group: each card
holds 7 of the 56 heads and an eighth of the sequence rows, and the decode
splits each clip's 28 tiles 4/4/4/4/3/3/3/3. It costs a wider all-to-all per
attention layer.

Measured in one session against `npu_bench_rtf_8card_950_sp4.yaml`, on the
first iteration, rollout went from 25.5 s to 15.1 s (1.69x), decode from 6.9 s
to 4.5 s, and time to first chunk from 1.76 s to 1.16 s. A steady-state chunk
takes 1.41 s, or 0.28 s per forward. RTF went from 4.8 to 2.6. For a batch of
prompts the two layouts are about even: two groups finished two clips in
33–47 s (16.5–23.5 s each), and one group finishes one in 20.6 s. Use
`npu_8card_950_sp4.yaml` for batch throughput.

Measured in one session against that switch, decode went from 28.9 s to
8.0 s on average and from 22.6 s to 6.8 s on the first iteration (3.3x on
four cards). The remainder is the audio decode, which is not split, and the
per-clip all-gathers. RTF went from 7.8 to 4.8, the first iteration's clip
from 48.7 s to 32.7 s (RTF 4.1), and peak memory stayed at 87.3 GiB. Rollout
is unchanged: 25.3 s against 25.4 s on the first iteration. Its lower mean
in the last row only reflects the cards heating less between the shorter
iterations.

### Finding where the time goes

All three of these tools run on the real model build or at its real shapes:

- **Attention microbenchmark.** `python -m projects.minimax_h3.tools.bench_npu_attention`
  checks the fused kernel against SDPA on a ragged packing, then times both
  at the production per-rank shapes as the KV cache fills.
- **Phase and chunk breakdown.**
  `trials/npu/npu_bench_rtf_8card_950.yaml` is `npu_8card_950.yaml` with the
  RTF engine. It splits encode / rollout / decode, discards a warmup pass,
  and logs the backbone's parameter dtypes, LoRA adapter separately. Its header lists the one-variable A/B runs.
  `npu_bench_rtf_8card_950_fsdp.yaml` is the FSDP-sharded variant, and
  `npu_bench_rtf_8card_950_sp4.yaml` the two-group variant.
- **Op-level trace.** Add `benchmark.profile true` to wrap one steady-state
  chunk in the `torch_npu` profiler. The trace is written under
  `benchmark/profile/rank0/` and opens in MindStudio Insight.

## What the port changes

| File | Change |
| --- | --- |
| `common/distributed/ops.py` | `get_device_type()` / `get_device_module()` / `get_device()` / `get_process_group_backend()`. The process group backend becomes the composite `cpu:gloo,npu:hccl`. |
| `projects/minimax_h3/**` | `torch.autocast` regions keyed to the active device type instead of a hardcoded `cuda`, so the fp32 islands survive. |
| `projects/minimax_h3/modeling/audio_vae/alias_free.py`, `bigvgan.py` | Convolutions chunked when their input exceeds CANN's 1,000,000-element width limit. |
| `utils/__init__.py` | Makes `utils` a regular package so `utils.flex_attn` resolves. |
| `tools/multi_run.sh` | NPU branch: CANN env, `torch_npu` assertion, HCCL timeout, allocator config. conda and `venv/` made optional. |
| `tools/prepare_venv_npu.sh`, `tools/environment-npu.yaml`, `tools/requirements-npu-inference.txt` | NPU environment build. |
| `projects/minimax_h3/tools/check_autocast.py` | Verifies autocast device keying. |
| `utils/flash_attn.py` | Non-causal varlen attention routed to `torch_npu.npu_fusion_attention` (TND layout) on NPU. |
| `projects/minimax_h3/modeling/transformer/causal_model*.py` | The cached rollout makes the new K/V contiguous before merging them with the cache; strided sources sent `index_put` down a slow path. |
| `common/plugin/lora_merge.py` | Placement plugins for inference: `CastLoraAfterPlacement` runs the adapter unmerged in bf16, `MergeLoraAfterPlacement` folds it into the base weights (degrades clips past 8 s). |
| `projects/minimax_h3/modeling/video_vae/klvae.py` | `tile_parallel_group(group)` replaces the unvendored sglang decode-parallel state, so the released tile-parallel decode runs over a torch process group. |
| `projects/minimax_h3/meta_models/causal_minimax_h3_base.py` | Validation decodes on every rank of the SP group with the tiles split between them (`validation.decode_parallel`, on by default); the group's main rank still writes the video. |
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

The next limit is the decoder's last stage, which runs at the full 32 kHz
rate: its residual-block convolutions and `conv_post` cross 1,000,000 samples
at about 31 s, and the last upsampling layer at about 62 s. A 40.6 s clip
failed there (`Conv2DV2 ... [2,8,1,1298400]`). `bigvgan.py` now calls those
layers through `npu_conv1d` / `npu_conv_transpose1d`, the same chunking with
dilation, padding and bias handled, and only when the input is too wide. With
them the 40.6 s clip's audio decodes.

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
- The SP-group decode has been checked for identical output on CPU only. On
  hardware the video has been generated but not compared frame by frame.
- The audio VAE decode is not split; it runs on the rank that saves the video.
- Why the merge breaks clips past 8 s is not proven. The bisect isolated it:
  merge off was the only change between a degraded and a clean 15.8 s clip.
- Decoding into host memory has been checked for identical output on CPU
  only. On hardware a 40.6 s clip decoded this way looks clean but has not been
  compared frame by frame.
- 60.4 s has not been generated since the host decode and the BigVGAN
  chunking were added; at that length the last upsampling layer also needs
  chunking, which is in place but unexercised.
- The 4-, 2- and 1-card 950 trials, the 64 GiB trials and the two-node setup
  have not been run. Their memory figures are estimates from the measured
  8-card build.
- Batching several prompts into one forward would share FSDP's all-gather
  between them and might make FSDP competitive for throughput at a third of
  the memory. Not measured; validation does not batch prompts that way today.
- The contiguous KV merge has not been measured on its own. Comparing
  sessions puts it at about 2 s of rollout, which is within thermal drift.
- FSDP `CPUOffloadPolicy` across multiple ranks is unverified — it needs
  `--ulimit memlock=-1` and has not been run to completion.
- Training on NPU is untested.
