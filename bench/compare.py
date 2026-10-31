#!/usr/bin/env python3
"""Benchmark the TinyLlama C++ engine against Ollama's tinyllama:1.1b.

Fairness rules applied to both engines:
  * identical prompts and the same Zephyr chat template (Ollama gets raw=true)
  * greedy decoding (temperature 0), same max new tokens, same thread count
  * a warm-up run first (loads weights into RAM), then RUNS measured runs
  * each run uses a different prompt, so Ollama's prompt cache cannot skip prefill
Medians are reported. Every run is saved to bench/results/ and appended to
bench/results/history.csv so you can track optimisation progress over time.
"""
import argparse
import csv
import datetime
import json
import os
import statistics
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

SYSTEM = "You are a helpful AI assistant."
PROMPTS = [
    "Explain how a CPU cache works, in simple terms.",
    "Write a short story about a robot who learns to paint.",
    "Describe the water cycle step by step.",
    "What are the main differences between TCP and UDP?",
    "Give me five tips for learning a new programming language.",
    "Why is the sky blue? Explain like I am ten.",
    "Summarise the history of the printing press.",
]
RESULTS = Path(__file__).resolve().parent / "results"


def chat_template(user):
    return f"<|system|>\n{SYSTEM}</s>\n<|user|>\n{user}</s>\n<|assistant|>\n"


# ---------------------------------------------------------------- our engine
def run_engine(a, user):
    cmd = [a.bin, "bench", "--model", a.model, "--tokenizer", a.tokenizer,
           "-p", user, "--system", SYSTEM, "-n", str(a.n), "--threads", str(a.threads)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        sys.exit(f"engine failed:\n{p.stderr}")
    return json.loads(p.stdout.strip().splitlines()[-1])


# ---------------------------------------------------------------- ollama
def http(url, body=None, timeout=900):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def ollama_model_size(a):
    try:
        tags = http(f"{a.ollama_url}/api/tags", timeout=5)
    except (urllib.error.URLError, OSError):
        sys.exit("Ollama is not running. In another terminal run:  make ollama-serve")
    for m in tags.get("models", []):
        if m.get("name") == a.ollama_model or m.get("model") == a.ollama_model:
            return m.get("size", 0)
    sys.exit(f"Ollama has no model {a.ollama_model}. Run:  make ollama-pull")


def run_ollama(a, user):
    r = http(f"{a.ollama_url}/api/generate", {
        "model": a.ollama_model,
        "prompt": chat_template(user),
        "raw": True,
        "stream": False,
        "options": {"temperature": 0, "seed": 42, "num_predict": a.n,
                    "num_thread": a.threads, "num_ctx": 2048},
    })
    pe, ped = r.get("prompt_eval_count", 0), r.get("prompt_eval_duration", 0) / 1e6
    ec, ed = r.get("eval_count", 0), r.get("eval_duration", 0) / 1e6
    return {"prompt_tokens": pe, "gen_tokens": ec, "prefill_ms": ped, "decode_ms": ed,
            "prefill_tps": pe / (ped / 1000) if ped else 0.0,
            "decode_tps": ec / (ed / 1000) if ed else 0.0}


# ---------------------------------------------------------------- helpers
def measure_bandwidth(a):
    if not a.bandwidth_bin or not Path(a.bandwidth_bin).exists():
        return None
    env = dict(os.environ, OMP_NUM_THREADS=str(a.threads))
    p = subprocess.run([a.bandwidth_bin], capture_output=True, text=True, env=env)
    return json.loads(p.stdout)["read_gbps"] if p.returncode == 0 else None


def median_of(runs):
    keys = ["prompt_tokens", "gen_tokens", "prefill_tps", "decode_tps"]
    return {k: statistics.median(r[k] for r in runs) for k in keys}


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "nogit"


def bench(name, fn, a):
    print(f"  {name}: warm-up", flush=True)
    fn(a, "Say hello in one sentence.")
    runs = []
    for i in range(a.runs):
        user = PROMPTS[i % len(PROMPTS)]
        r = fn(a, user)
        runs.append(r)
        print(f"  {name}: run {i + 1}/{a.runs}  decode {r['decode_tps']:.2f} tok/s  "
              f"prefill {r['prefill_tps']:.1f} tok/s", flush=True)
    return runs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=["engine", "ollama", "both"], default="both")
    ap.add_argument("--bin", default="build/tinyllama")
    ap.add_argument("--model", default="models/tinyllama-f32.bin")
    ap.add_argument("--tokenizer", default="models/tokenizer.bin")
    ap.add_argument("--bandwidth-bin", default="build/bandwidth")
    ap.add_argument("--ollama-model", default="tinyllama:1.1b")
    ap.add_argument("--ollama-url", default="http://localhost:11434")
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    ap.add_argument("-n", type=int, default=128, help="max new tokens per run")
    ap.add_argument("--runs", type=int, default=3)
    a = ap.parse_args()

    print(f"benchmark: {a.runs} runs, up to {a.n} new tokens, {a.threads} threads, greedy")
    gbps = measure_bandwidth(a)
    if gbps:
        print(f"  memory read bandwidth: {gbps:.1f} GB/s")

    rows = []
    if a.only in ("engine", "both"):
        runs = bench("tinyllama-cpp", run_engine, a)
        rows.append(("tinyllama-cpp", median_of(runs), runs[0]["bytes_per_token"], runs))
    if a.only in ("ollama", "both"):
        size = ollama_model_size(a)
        runs = bench("ollama", run_ollama, a)
        rows.append((f"ollama {a.ollama_model}", median_of(runs), size, runs))

    print()
    hdr = f"{'engine':<24}{'prompt tok':>11}{'prefill tok/s':>15}{'gen tok':>9}{'decode tok/s':>14}{'model MB':>10}"
    if gbps:
        hdr += f"{'% of BW limit':>15}"
    print(hdr)
    print("-" * len(hdr))
    for name, m, nbytes, _ in rows:
        line = (f"{name:<24}{m['prompt_tokens']:>11.0f}{m['prefill_tps']:>15.1f}"
                f"{m['gen_tokens']:>9.0f}{m['decode_tps']:>14.2f}{nbytes / 1e6:>10.0f}")
        if gbps and nbytes:
            ceiling = gbps * 1e9 / nbytes
            line += f"{100 * m['decode_tps'] / ceiling:>14.0f}%"
        print(line)
    if len(rows) == 2 and rows[1][1]["decode_tps"]:
        ratio = rows[0][1]["decode_tps"] / rows[1][1]["decode_tps"]
        print(f"\ndecode speed vs Ollama: {ratio:.2f}x")
    if gbps:
        print("% of BW limit = decode tok/s divided by (bandwidth / bytes read per token)")

    RESULTS.mkdir(exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    commit = git_commit()
    record = {"time": stamp, "commit": commit, "threads": a.threads, "n": a.n, "bandwidth_gbps": gbps,
              "results": {name: {"median": m, "model_bytes": nb, "runs": r} for name, m, nb, r in rows}}
    (RESULTS / f"{stamp}.json").write_text(json.dumps(record, indent=2))
    hist = RESULTS / "history.csv"
    new = not hist.exists()
    with hist.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "commit", "engine", "threads", "decode_tps", "prefill_tps", "model_mb"])
        for name, m, nb, _ in rows:
            w.writerow([stamp, commit, name, a.threads, f"{m['decode_tps']:.2f}",
                        f"{m['prefill_tps']:.1f}", f"{nb / 1e6:.0f}"])
    print(f"saved bench/results/{stamp}.json")


if __name__ == "__main__":
    main()
