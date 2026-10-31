#!/usr/bin/env bash
# Downloads TinyLlama-1.1B-Chat-v1.0 from Hugging Face (about 2.2 GB, resumable).
set -euo pipefail
REPO="${HF_REPO:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
DIR="${1:-models}"
mkdir -p "$DIR"
for f in config.json tokenizer.json model.safetensors; do
  if [[ -s "$DIR/$f" ]]; then echo "  ok   $f"; continue; fi
  echo "  get  $f"
  curl -L --fail --retry 3 -C - ${HF_TOKEN:+-H "Authorization: Bearer $HF_TOKEN"} \
       -o "$DIR/$f.part" "https://huggingface.co/$REPO/resolve/main/$f"
  mv "$DIR/$f.part" "$DIR/$f"
done
