#!/usr/bin/env python3
"""Creates a toy random model in the Hugging Face layout (config.json,
model.safetensors, tokenizer.json) so `make smoke` can test the whole
pipeline in seconds without downloading TinyLlama."""
import json
import random
import struct
import sys
from pathlib import Path

D, H, L, NH, NKV, V, S = 64, 172, 2, 4, 2, 300, 128
KV = NKV * (D // NH)
out = Path(sys.argv[1] if len(sys.argv) > 1 else "build-tiny/models")
out.mkdir(parents=True, exist_ok=True)
random.seed(0)

(out / "config.json").write_text(json.dumps({
    "hidden_size": D, "intermediate_size": H, "num_hidden_layers": L, "num_attention_heads": NH,
    "num_key_value_heads": NKV, "vocab_size": V, "max_position_embeddings": S}))

shapes = {"model.embed_tokens.weight": [V, D], "model.norm.weight": [D], "lm_head.weight": [V, D]}
for i in range(L):
    p = f"model.layers.{i}."
    shapes.update({p + "input_layernorm.weight": [D], p + "post_attention_layernorm.weight": [D],
                   p + "self_attn.q_proj.weight": [D, D], p + "self_attn.k_proj.weight": [KV, D],
                   p + "self_attn.v_proj.weight": [KV, D], p + "self_attn.o_proj.weight": [D, D],
                   p + "mlp.gate_proj.weight": [H, D], p + "mlp.up_proj.weight": [H, D],
                   p + "mlp.down_proj.weight": [D, H]})
header, blobs, off = {}, [], 0
for name, shape in shapes.items():
    n = 1
    for s in shape:
        n *= s
    norm = name.endswith("norm.weight")
    vals = [1.0 if norm else random.gauss(0, 0.05) for _ in range(n)]
    blob = b"".join(struct.pack("<f", v)[2:4] for v in vals)   # fp32 -> bf16 (truncate)
    header[name] = {"dtype": "BF16", "shape": shape, "data_offsets": [off, off + len(blob)]}
    blobs.append(blob)
    off += len(blob)
hj = json.dumps(header).encode()
hj += b" " * ((-len(hj)) % 8)
(out / "model.safetensors").write_bytes(struct.pack("<Q", len(hj)) + hj + b"".join(blobs))

vocab = {"<unk>": 0, "<s>": 1, "</s>": 2}
for b in range(256):
    vocab[f"<0x{b:02X}>"] = 3 + b
for ch in ["\u2581"] + list("abcdefghijklmnopqrstuvwxyz"):
    vocab[ch] = len(vocab)
merges = [("\u2581", "h"), ("e", "l"), ("el", "l"), ("ell", "o"), ("\u2581h", "ello"),
          ("\u2581", "w"), ("o", "r"), ("l", "d"), ("or", "ld"), ("\u2581w", "orld"),
          ("\u2581", "t"), ("h", "e"), ("\u2581t", "he"), ("\u2581", "a")]
for a, b in merges:
    vocab[a + b] = len(vocab)
assert len(vocab) == V, len(vocab)
(out / "tokenizer.json").write_text(json.dumps(
    {"model": {"type": "BPE", "vocab": vocab, "merges": [f"{a} {b}" for a, b in merges]}}), encoding="utf-8")
print(f"toy model written to {out}")
