#!/usr/bin/env python3
"""Quality evaluation: WikiText-2 perplexity, KL divergence and top-1 agreement vs fp32.

Downloads the wikitext-2-raw-v1 test split once (Hugging Face datasets-server JSON API,
standard library only) into bench/data/wikitext2-test.txt, then runs `tinyllama ppl`.

  python3 bench/eval.py --model models/tinyllama-q4.bin --ref-model models/tinyllama-f32.bin
"""
import argparse
import json
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data" / "wikitext2-test.txt"
API = "https://datasets-server.huggingface.co/rows"


def fetch_wikitext(path):
    if path.exists():
        return
    print("downloading WikiText-2 test split", flush=True)
    rows, offset = [], 0
    while True:
        q = urllib.parse.urlencode({"dataset": "Salesforce/wikitext", "config": "wikitext-2-raw-v1",
                                    "split": "test", "offset": offset, "length": 100})
        with urllib.request.urlopen(f"{API}?{q}", timeout=60) as r:
            page = json.load(r)
        batch = [x["row"]["text"] for x in page["rows"]]
        rows += batch
        offset += len(batch)
        if not batch or offset >= page["num_rows_total"]:
            break
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(rows), encoding="utf-8")
    print(f"  saved {path} ({len(rows)} lines)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin", default="build/tinyllama")
    ap.add_argument("--model", default="models/tinyllama-q4.bin")
    ap.add_argument("--ref-model", default="models/tinyllama-f32.bin", help="'' to skip KL")
    ap.add_argument("--tokenizer", default="models/tokenizer.bin")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--ctx", type=int, default=512, help="chunk length (512 or 2048 in the plan)")
    ap.add_argument("--tokens", type=int, default=1024, help="tokens scored (whole set is ~300k; fp32 ref runs ~3 tok/s)")
    a = ap.parse_args()

    fetch_wikitext(DATA)
    cmd = [a.bin, "ppl", "--model", a.model, "--tokenizer", a.tokenizer, "--text", str(DATA),
           "--ctx", str(a.ctx), "--max-tokens", str(a.tokens)]
    if a.threads:
        cmd += ["--threads", str(a.threads)]
    if a.ref_model and Path(a.ref_model).resolve() != Path(a.model).resolve():
        cmd += ["--ref-model", a.ref_model]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        sys.exit(p.stderr)
    r = json.loads(p.stdout.strip().splitlines()[-1])
    print(f"WikiText-2 test, {r['tokens']} tokens, ctx {a.ctx}: {Path(a.model).name}")
    print(f"  perplexity  {r['ppl']:.4f}")
    if "kl" in r:
        print(f"  fp32 ref    {r['ref_ppl']:.4f}  ({100 * (r['ppl'] / r['ref_ppl'] - 1):+.2f}%)")
        print(f"  KL vs fp32  {r['kl']:.5f} nats/token")
        print(f"  top-1 agree {100 * r['top1_agree']:.2f}%")


if __name__ == "__main__":
    main()
