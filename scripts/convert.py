#!/usr/bin/env python3
"""Convert TinyLlama safetensors + tokenizer.json into the engine's own files.

Python standard library only (no torch, no numpy), so the "no ML libraries"
rule holds even for tooling.

  models/tinyllama-f32.bin   header + every tensor as float32, 64-byte aligned
  models/tokenizer.bin       vocabulary with BPE merge scores
"""
import argparse
import json
import mmap
import struct
import sys
from pathlib import Path

ALIGN = 64
HEADER_BYTES = 256


def load_safetensors(path):
    f = open(path, "rb")
    n = struct.unpack("<Q", f.read(8))[0]
    header = json.loads(f.read(n))
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    return header, mm, 8 + n


def tensor_as_f32(header, mm, base, name, shape):
    if name not in header:
        sys.exit(f"missing tensor {name}")
    t = header[name]
    if list(t["shape"]) != list(shape):
        sys.exit(f"{name}: shape {t['shape']} != expected {shape}")
    a, b = t["data_offsets"]
    raw = mm[base + a: base + b]
    if t["dtype"] == "F32":
        return raw
    if t["dtype"] == "BF16":
        # bf16 is the top 16 bits of fp32: little-endian bytes [lo, hi] -> [0, 0, lo, hi]
        out = bytearray(len(raw) * 2)
        out[2::4] = raw[0::2]
        out[3::4] = raw[1::2]
        return out
    sys.exit(f"{name}: unsupported dtype {t['dtype']}")


def write_aligned(f, data):
    pad = (-f.tell()) % ALIGN
    if pad:
        f.write(b"\0" * pad)
    f.write(data)


def convert_weights(model_dir, out_path):
    cfg = json.loads((model_dir / "config.json").read_text())
    D, H = cfg["hidden_size"], cfg["intermediate_size"]
    L, NH, NKV = cfg["num_hidden_layers"], cfg["num_attention_heads"], cfg["num_key_value_heads"]
    V, S = cfg["vocab_size"], cfg["max_position_embeddings"]
    KV = NKV * (D // NH)

    st = model_dir / "model.safetensors"
    if not st.exists():
        sys.exit(f"{st} not found (sharded checkpoints are not supported)")
    header, mm, base = load_safetensors(st)

    order = [("model.embed_tokens.weight", [V, D])]
    for i in range(L):
        p = f"model.layers.{i}."
        order += [
            (p + "input_layernorm.weight", [D]),
            (p + "self_attn.q_proj.weight", [D, D]),
            (p + "self_attn.k_proj.weight", [KV, D]),
            (p + "self_attn.v_proj.weight", [KV, D]),
            (p + "self_attn.o_proj.weight", [D, D]),
            (p + "post_attention_layernorm.weight", [D]),
            (p + "mlp.gate_proj.weight", [H, D]),
            (p + "mlp.up_proj.weight", [H, D]),
            (p + "mlp.down_proj.weight", [D, H]),
        ]
    lm_head = "lm_head.weight" if "lm_head.weight" in header else "model.embed_tokens.weight"
    order += [("model.norm.weight", [D]), (lm_head, [V, D])]

    with open(out_path, "wb") as f:
        hdr = b"TLM1" + struct.pack("<8i", 1, D, H, L, NH, NKV, V, S)
        f.write(hdr + b"\0" * (HEADER_BYTES - len(hdr)))
        for k, (name, shape) in enumerate(order):
            write_aligned(f, tensor_as_f32(header, mm, base, name, shape))
            print(f"\r  weights {k + 1}/{len(order)}", end="", flush=True)
    print(f"\n  wrote {out_path} ({out_path.stat().st_size / 1e9:.2f} GB)")


def convert_tokenizer(model_dir, out_path):
    tj = json.loads((model_dir / "tokenizer.json").read_text(encoding="utf-8"))
    vocab = tj["model"]["vocab"]           # token -> id
    merges = tj["model"].get("merges", [])
    score = {}
    for rank, m in enumerate(merges):     # earlier merge = higher priority
        a, b = m.split(" ", 1) if isinstance(m, str) else m
        score.setdefault(a + b, -float(rank))
    for t in tj.get("added_tokens", []):
        vocab.setdefault(t["content"], t["id"])

    n = max(vocab.values()) + 1
    by_id = [""] * n
    for tok, i in vocab.items():
        by_id[i] = tok
    with open(out_path, "wb") as f:
        f.write(b"TLT1" + struct.pack("<I", n))
        for tok in by_id:
            data = tok.encode("utf-8")
            f.write(struct.pack("<fI", score.get(tok, -1e9), len(data)) + data)
    print(f"  wrote {out_path} ({n} tokens)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", default="models", type=Path)
    ap.add_argument("--out-dir", default="models", type=Path)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print("converting tokenizer")
    convert_tokenizer(args.model_dir, args.out_dir / "tokenizer.bin")
    print("converting weights (bf16 -> fp32)")
    convert_weights(args.model_dir, args.out_dir / "tinyllama-f32.bin")


if __name__ == "__main__":
    main()
