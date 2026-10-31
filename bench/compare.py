#!/usr/bin/env python3
"""Benchmark the TinyLlama C++ engine against Ollama's tinyllama:1.1b.

Fairness rules applied to both engines:
  * identical prompts and the same Zephyr chat template (Ollama gets raw=true)
  * greedy decoding (temperature 0), same max new tokens, same thread count
  * a warm-up run first (loads weights into RAM), then RUNS measured runs
  * each run uses a different prompt, so Ollama's prompt cache cannot skip prefill
Metrics: decode/prefill tok/s, time to first token, peak RAM, model size,
CPU utilisation, energy (Intel/AMD RAPL, needs read access: make rapl-perms),
and quality (perplexity + top-1 agreement vs the fp32 reference, --quality).
Medians are reported. Every run is saved to bench/results/ and appended to
bench/results/history.csv so you can track optimisation progress over time.
"""
import argparse
import time
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

# Each run gets a unique tag at the very start of the system prompt, so Ollama's
# server-side prompt cache cannot reuse KV from earlier runs or earlier sessions.
SYSTEM = "You are a helpful AI assistant."
RUN_ID = os.urandom(4).hex()
_run_counter = 0


def tagged_system():
    global _run_counter
    _run_counter += 1
    return f"[session {RUN_ID}-{_run_counter}] {SYSTEM}"
PROMPTS = [
    "Explain how a CPU cache works, in simple terms.",
    "Write a short story about a robot who learns to paint.",
    "Describe the water cycle step by step.",
    "What are the main differences between TCP and UDP?",
    "Give me five tips for learning a new programming language.",
    "Why is the sky blue? Explain like I am ten.",
    "Summarise the history of the printing press.",
]
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
EVAL_TEXT = HERE / "data" / "eval.txt"


# ---------------------------------------------------------------- energy + process stats
def rapl_domains():
    """Top-level package counters (intel-rapl:0, intel-rapl:1, ...) that we can read."""
    out = []
    for d in sorted(Path("/sys/class/powercap").glob("intel-rapl:*")):
        if d.name.count(":") != 1:          # skip sub-domains (core, uncore) to avoid double counting
            continue
        try:
            int((d / "energy_uj").read_text())
            out.append((d / "energy_uj", int((d / "max_energy_range_uj").read_text())))
        except (OSError, ValueError):
            pass
    return out


def rapl_read(domains):
    return [int(p.read_text()) for p, _ in domains]


def rapl_joules(domains, before, after):
    total = 0
    for (_, wrap), b, a in zip(domains, before, after):
        total += a - b if a >= b else a + wrap - b
    return total / 1e6


class Energy:
    """Package energy over a block of code; .joules is None if RAPL is unreadable."""
    domains = None

    def __enter__(self):
        if Energy.domains is None:
            Energy.domains = rapl_domains()
        self.joules = None
        self.b = rapl_read(Energy.domains) if Energy.domains else None
        return self

    def __exit__(self, *exc):
        if self.b is not None:
            self.joules = rapl_joules(Energy.domains, self.b, rapl_read(Energy.domains))


CLK_TCK = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100


def ollama_pids():
    """PIDs of Ollama processes (server + model runner)."""
    pids = []
    for d in Path("/proc").iterdir():
        if d.name.isdigit():
            try:
                if b"ollama" in (d / "cmdline").read_bytes():
                    pids.append(int(d.name))
            except OSError:
                pass
    return pids


def cpu_seconds(pids):
    t = 0
    for pid in pids:
        try:
            f = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
            t += int(f[11]) + int(f[12])     # utime + stime
        except (OSError, IndexError):
            pass
    return t / CLK_TCK


def peak_rss_mb(pids):
    """Largest VmHWM (lifetime peak RSS) among the pids; for Ollama that is the runner."""
    best = 0
    for pid in pids:
        try:
            for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                if line.startswith("VmHWM:"):
                    best = max(best, int(line.split()[1]) / 1024)
        except OSError:
            pass
    return best


def chat_template(system, user):
    return f"<|system|>\n{system}</s>\n<|user|>\n{user}</s>\n<|assistant|>\n"


# ---------------------------------------------------------------- our engine
def run_engine(a, user):
    cmd = [a.bin, "bench", "--model", a.model, "--tokenizer", a.tokenizer,
           "-p", user, "--system", tagged_system(), "-n", str(a.n), "--threads", str(a.threads)] + a.engine_args.split()
    with Energy() as e:
        p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        sys.exit(f"engine failed:\n{p.stderr}")
    r = json.loads(p.stdout.strip().splitlines()[-1])
    r["energy_j"] = e.joules     # whole process, includes weight load
    return r


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
    pids = ollama_pids()
    c0, w0 = cpu_seconds(pids), time.perf_counter()
    with Energy() as e:
        r = http(f"{a.ollama_url}/api/generate", {
        "model": a.ollama_model,
        "prompt": chat_template(tagged_system(), user),
        "raw": True,
        "stream": False,
        "options": {"temperature": 0, "seed": 42, "num_predict": a.n,
                    "num_thread": a.threads, "num_ctx": 2048},
        })
    pids = ollama_pids()                     # the runner may have just started
    wall = time.perf_counter() - w0
    cpu = cpu_seconds(pids) - c0
    pe, ped = r.get("prompt_eval_count", 0), r.get("prompt_eval_duration", 0) / 1e6
    ec, ed = r.get("eval_count", 0), r.get("eval_duration", 0) / 1e6
    return {"prompt_tokens": pe, "gen_tokens": ec, "prefill_ms": ped, "decode_ms": ed,
            "prefill_tps": pe / (ped / 1000) if ped else 0.0,
            "decode_tps": ec / (ed / 1000) if ed else 0.0,
            "ttft_ms": ped + r.get("load_duration", 0) / 1e6,
            "peak_rss_mb": peak_rss_mb(pids),
            "cpu_util_pct": 100 * cpu / wall if wall else 0.0,
            "energy_j": e.joules}


# ---------------------------------------------------------------- helpers
def physical_cores():
    """Same default as the Makefile (scripts/physical_cores.sh), not os.cpu_count()."""
    script = Path(__file__).resolve().parent.parent / "scripts" / "physical_cores.sh"
    try:
        return int(subprocess.run([str(script)], capture_output=True, text=True, check=True).stdout)
    except (OSError, ValueError, subprocess.CalledProcessError):
        return os.cpu_count() or 4


def measure_bandwidth(a):
    if not a.bandwidth_bin or not Path(a.bandwidth_bin).exists():
        return None
    env = dict(os.environ, OMP_NUM_THREADS=str(a.threads))
    p = subprocess.run([a.bandwidth_bin], capture_output=True, text=True, env=env)
    return json.loads(p.stdout)["read_gbps"] if p.returncode == 0 else None


def median_of(runs):
    keys = ["prompt_tokens", "gen_tokens", "prefill_tps", "decode_tps", "ttft_ms",
            "peak_rss_mb", "cpu_util_pct"]
    m = {k: statistics.median(r[k] for r in runs) for k in keys}
    if all(r["energy_j"] is not None for r in runs):
        m["energy_j"] = statistics.median(r["energy_j"] for r in runs)
        m["j_per_tok"] = statistics.median(r["energy_j"] / max(r["gen_tokens"], 1) for r in runs)
    else:
        m["energy_j"] = m["j_per_tok"] = None
    return m


# ---------------------------------------------------------------- quality
def run_ppl(a, model):
    cmd = [a.bin, "ppl", "--model", model, "--tokenizer", a.tokenizer, "--text", str(a.eval_text),
           "--max-tokens", str(a.ppl_tokens), "--ctx", str(a.ppl_ctx), "--threads", str(a.threads)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        sys.exit(f"ppl failed:\n{p.stderr}")
    return json.loads(p.stdout.strip().splitlines()[-1])


def reference_ppl(a):
    """fp32 reference result, cached by file size+mtime so it is computed once."""
    st = Path(a.ref_model).stat()
    key = f"{Path(a.ref_model).name}-{st.st_size}-{int(st.st_mtime)}-{a.ppl_tokens}-{a.ppl_ctx}"
    cache = RESULTS / f"ppl-ref-{key}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    print("  quality: scoring fp32 reference (cached after the first time)", flush=True)
    r = run_ppl(a, a.ref_model)
    RESULTS.mkdir(exist_ok=True)
    cache.write_text(json.dumps(r))
    return r


def quality(a):
    ref = reference_ppl(a)
    if Path(a.model).resolve() == Path(a.ref_model).resolve():
        cur = ref
    else:
        print("  quality: scoring model under test", flush=True)
        cur = run_ppl(a, a.model)
    n = min(len(ref["top1"]), len(cur["top1"]))
    agree = sum(x == y for x, y in zip(ref["top1"], cur["top1"])) / n if n else 0.0
    return {"tokens": cur["tokens"], "ppl": cur["ppl"], "ref_ppl": ref["ppl"],
            "ppl_delta_pct": 100 * (cur["ppl"] / ref["ppl"] - 1), "top1_agree_pct": 100 * agree}


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
    ap.add_argument("--model", default="models/tinyllama-q4.bin")
    ap.add_argument("--tokenizer", default="models/tokenizer.bin")
    ap.add_argument("--bandwidth-bin", default="build/bandwidth")
    ap.add_argument("--ollama-model", default="tinyllama:1.1b")
    ap.add_argument("--ollama-url", default="http://localhost:11434")
    ap.add_argument("--threads", type=int, default=physical_cores(),
                    help="default: physical cores (more threads than fast cores disables pinning)")
    ap.add_argument("-n", type=int, default=128, help="max new tokens per run")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--engine-args", default="", help='extra engine flags, e.g. "--kv q8 --spec ngram"')
    ap.add_argument("--quality", action="store_true", help="also measure perplexity vs fp32 reference")
    ap.add_argument("--ref-model", default="models/tinyllama-f32.bin", help="fp32 reference for quality")
    ap.add_argument("--eval-text", default=str(EVAL_TEXT))
    ap.add_argument("--ppl-tokens", type=int, default=512, help="tokens scored for perplexity")
    ap.add_argument("--ppl-ctx", type=int, default=256, help="perplexity window length")
    a = ap.parse_args()

    print(f"benchmark: {a.runs} runs, up to {a.n} new tokens, {a.threads} threads, greedy")
    gbps = measure_bandwidth(a)
    if gbps:
        print(f"  memory read probe: {gbps:.1f} GB/s (pure read loop; on laptops in powersave it can read"
              f" below what a real workload sustains, so it is shown for reference only)")
    if not rapl_domains():
        print("  energy: RAPL not readable, skipped (run: make rapl-perms)")

    rows = []
    if a.only in ("engine", "both"):
        runs = bench("tinyllama-cpp", run_engine, a)
        rows.append(("tinyllama-cpp", median_of(runs), runs[0]["bytes_per_token"], runs))
    if a.only in ("ollama", "both"):
        size = ollama_model_size(a)
        runs = bench("ollama", run_ollama, a)
        rows.append((f"ollama {a.ollama_model}", median_of(runs), size, runs))

    qual = quality(a) if a.quality and a.only in ("engine", "both") else None

    def fmt(v, spec):
        return "n/a" if v is None else format(v, spec)

    cols = [("prompt tok", lambda m, nb: fmt(m["prompt_tokens"], ".0f")),
            ("prefill tok/s", lambda m, nb: fmt(m["prefill_tps"], ".1f")),
            ("decode tok/s", lambda m, nb: fmt(m["decode_tps"], ".2f")),
            ("TTFT ms", lambda m, nb: fmt(m["ttft_ms"], ".0f")),
            ("peak RAM MB", lambda m, nb: fmt(m["peak_rss_mb"], ".0f")),
            ("model MB", lambda m, nb: fmt(nb / 1e6, ".0f")),
            ("CPU %", lambda m, nb: fmt(m["cpu_util_pct"], ".0f")),
            ("J/token", lambda m, nb: fmt(m["j_per_tok"], ".2f"))]
    cols.append(("read GB/s", lambda m, nb: fmt(m["decode_tps"] * nb / 1e9 if nb else None, ".1f")))
    print()
    hdr = f"{'engine':<24}" + "".join(f"{c:>14}" for c, _ in cols)
    print(hdr)
    print("-" * len(hdr))
    for name, m, nb, _ in rows:
        print(f"{name:<24}" + "".join(f"{f(m, nb):>14}" for _, f in cols))
    if len(rows) == 2 and rows[1][1]["decode_tps"]:
        ratio = rows[0][1]["decode_tps"] / rows[1][1]["decode_tps"]
        print(f"\ndecode speed vs Ollama: {ratio:.2f}x")
    if qual:
        print(f"quality ({qual['tokens']} tokens of {Path(a.eval_text).name}): ppl {qual['ppl']:.3f} "
              f"vs fp32 {qual['ref_ppl']:.3f} ({qual['ppl_delta_pct']:+.2f}%), "
              f"top-1 agreement {qual['top1_agree_pct']:.1f}%")
    print(f"CPU % = CPU time / wall time, 100% per busy core ({a.threads} threads -> max {100 * a.threads}%)")
    print("TTFT  = prompt in -> first token (Ollama: + its model load time)")
    print("peak RAM includes mmap'd weight pages (engine) / runner lifetime peak (Ollama)")
    print("read GB/s = decode tok/s x weight bytes per token (effective memory bandwidth used by decode)")

    RESULTS.mkdir(exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    commit = git_commit()
    record = {"time": stamp, "commit": commit, "suite": "compare", "model": Path(a.model).name,
              "engine_args": a.engine_args, "threads": a.threads, "n": a.n, "bandwidth_gbps": gbps, "quality": qual,
              "results": {name: {"median": m, "model_bytes": nb, "runs": r} for name, m, nb, r in rows}}
    (RESULTS / f"{stamp}.json").write_text(json.dumps(record, indent=2))
    hist = RESULTS / "history.csv"
    new = not hist.exists()
    with hist.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "commit", "engine", "threads", "decode_tps", "prefill_tps", "model_mb",
                        "ttft_ms", "peak_rss_mb", "cpu_util_pct", "j_per_tok", "ppl", "ppl_delta_pct",
                        "top1_agree_pct"])
        for name, m, nb, _ in rows:
            q = qual if qual and name == "tinyllama-cpp" else {}
            w.writerow([stamp, commit, name, a.threads, f"{m['decode_tps']:.2f}",
                        f"{m['prefill_tps']:.1f}", f"{nb / 1e6:.0f}", f"{m['ttft_ms']:.0f}",
                        f"{m['peak_rss_mb']:.0f}", f"{m['cpu_util_pct']:.0f}",
                        "" if m["j_per_tok"] is None else f"{m['j_per_tok']:.3f}",
                        f"{q['ppl']:.3f}" if q else "", f"{q['ppl_delta_pct']:.2f}" if q else "",
                        f"{q['top1_agree_pct']:.1f}" if q else ""])
    print(f"saved bench/results/{stamp}.json")


if __name__ == "__main__":
    main()
