#!/bin/bash
# Download the weights the MiniMax-H3 RAVEN streaming trials need.
#
#   bash tools/download_minimax_h3_weights.sh [WEIGHTS_DIR]
#
# WEIGHTS_DIR defaults to /root/models, the root the shipped trials point at;
# with it the trials run without path overrides. Any other directory works too:
# the script then writes the six path overrides to
# WEIGHTS_DIR/raven_minimax_h3_overrides.txt, to append to the launch command:
#
#   N=8 bash tools/multi_run.sh projects/minimax_h3/trials/npu/npu_8card_950.yaml \
#     $(cat WEIGHTS_DIR/raven_minimax_h3_overrides.txt)
#
# What it fetches, about 139 GiB in all, from two public Hugging Face repos:
#
#   MiniMaxAI/MiniMax-H3 (MiniMax-H3 Community License)
#     FL2VA/transformer         DiT, 33.1B parameters        61.7 GiB
#     FL2VA/text_encoder        Qwen3-VL text encoder        62.1 GiB
#     FL2VA/video_vae/source    video VAE                     9.7 GiB
#     audio_vae                 audio VAE                     0.6 GiB
#     FL2VA/tokenizer           tokenizer
#   mvp-lab/MiniMax-H3-RAVEN-Streaming-LoRA
#     the 4-NFE preview LoRA                                  4.7 GiB
#
# The rest of MiniMax-H3 (Ref2VA, the diffusers-format copies) is skipped.
# Re-running resumes: finished files are kept and partial ones continue.
#
# Environment:
#   HF_ENDPOINT     a Hugging Face mirror, e.g. https://hf-mirror.com
#   HF_TOKEN        an access token, if your network or a rate limit needs one
#   HF_MAX_WORKERS  parallel file downloads (default 8)
set -eo pipefail

WEIGHTS_DIR=${1:-/root/models}
mkdir -p "$WEIGHTS_DIR"
WEIGHTS_DIR=$(cd "$WEIGHTS_DIR" && pwd)

python - "$WEIGHTS_DIR" <<'EOF'
import fnmatch
import json
import os
import shutil
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

root = Path(sys.argv[1])
workers = int(os.environ.get("HF_MAX_WORKERS", "8"))

# (repo, local dir under root, files to fetch)
SOURCES = [
    (
        "MiniMaxAI/MiniMax-H3",
        "MiniMax-H3",
        [
            "FL2VA/transformer/*",
            "FL2VA/text_encoder/*",
            "FL2VA/video_vae/source/*",
            "FL2VA/tokenizer/*",
            "audio_vae/*",
        ],
    ),
    (
        "mvp-lab/MiniMax-H3-RAVEN-Streaming-LoRA",
        "mvp-lab/MiniMax-H3-RAVEN-Streaming-LoRA",
        ["minimax_h3_raven_streaming_lora_4nfe_preview.safetensors"],
    ),
]

# The trial keys these files feed, relative to root.
PATHS = {
    "data.args.tokenizer_path": "MiniMax-H3/FL2VA/tokenizer",
    "models.backbone.weight.path": "MiniMax-H3/FL2VA/transformer/model.safetensors.index.json",
    "models.backbone.adapter.weight": "mvp-lab/MiniMax-H3-RAVEN-Streaming-LoRA/minimax_h3_raven_streaming_lora_4nfe_preview.safetensors",
    "models.text_encoder.args.pretrained_model_name_or_path": "MiniMax-H3/FL2VA/text_encoder",
    "models.video_vae.weight.path": "MiniMax-H3/FL2VA/video_vae/source/model.safetensors",
    "models.audio_vae.weight.path": "MiniMax-H3/audio_vae/diffusion_pytorch_model.safetensors",
}


def gib(n):
    return f"{n / 2**30:.1f} GiB"


# 1. Size the download against the free space, counting what is already here.
api = HfApi()
total = missing = 0
for repo, local, patterns in SOURCES:
    for entry in api.list_repo_tree(repo, recursive=True):
        size = getattr(entry, "size", None)
        if size is None or not any(fnmatch.fnmatch(entry.path, p) for p in patterns):
            continue
        total += size
        target = root / local / entry.path
        if not (target.is_file() and target.stat().st_size == size):
            missing += size
free = shutil.disk_usage(root).free
print(f"[weights] {gib(total)} in all, {gib(missing)} still to download, {gib(free)} free in {root}")
if missing > free:
    sys.exit(f"[weights] not enough space in {root}: need {gib(missing)}, have {gib(free)}")

# 2. Download. snapshot_download keeps the repo's layout under local_dir and
#    skips files that are already complete.
for repo, local, patterns in SOURCES:
    print(f"[weights] {repo} -> {root / local}", flush=True)
    snapshot_download(
        repo_id=repo,
        local_dir=root / local,
        allow_patterns=patterns,
        max_workers=workers,
    )

# 3. Check that every file a trial reads is there, including every shard the
#    two sharded models' indexes name.
problems = []
for key, rel in PATHS.items():
    if not (root / rel).exists():
        problems.append(f"{key}: {root / rel} is missing")
for index in ("MiniMax-H3/FL2VA/transformer/model.safetensors.index.json",
              "MiniMax-H3/FL2VA/text_encoder/model.safetensors.index.json"):
    index = root / index
    if not index.is_file():
        problems.append(f"{index} is missing")
        continue
    for shard in sorted(set(json.loads(index.read_text())["weight_map"].values())):
        path = index.parent / shard
        if not path.is_file() or path.stat().st_size == 0:
            problems.append(f"{path} is missing or empty")
if problems:
    sys.exit("[weights] incomplete download; re-run to resume:\n  " + "\n  ".join(problems))

# 4. Tell the trials where the weights are.
overrides = " ".join(f"{key} {root / rel}" for key, rel in PATHS.items())
out = root / "raven_minimax_h3_overrides.txt"
out.write_text(overrides + "\n")
print(f"[weights] all files present under {root}")
if root == Path("/root/models"):
    print("[weights] this is the trials' default root: no path overrides are needed")
else:
    print(f"[weights] path overrides written to {out}; append them to the launch command:")
    print(f"  N=8 bash tools/multi_run.sh <trial.yaml> $(cat {out})")
EOF
