#!/usr/bin/env python3
"""Minimal OpenAI-compatible HTTP API for the tinyllama engine, plus a benchmark dashboard
(standard library only).

  GET  /v1/models, POST /v1/chat/completions   (streaming + non-streaming)
  GET  /dashboard                              benchmark results + experiment log
  GET  /api/results, /api/experiments          data for the dashboard

Each chat request runs `tinyllama run` with the whole conversation (--messages); weights are
mmap'd, so loading is ~instant. Replies carry Ollama-style speed stats in `usage` so Open
WebUI's info button shows the same numbers for both engines. One request at a time: the
engine already uses every core.

  python3 tools/openai_server.py --port 8000 --engine-args "--kv q8"
"""
import argparse
import json
import os
import re
import subprocess
import tempfile
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOCK = threading.Lock()   # ponytail: one generation at a time; a queue only if concurrent users matter
# Same stop strings as Ollama's tinyllama Modelfile: never let the model write the next turn.
DEFAULT_STOPS = ["<|user|>", "<|assistant|>", "<|system|>", "</s>"]
STATS = re.compile(r"\[prompt (\d+) tok, ([\d.]+) tok/s \| generated (\d+) tok, ([\d.]+) tok/s")


def content_text(content):
    """OpenAI content is a string or a list of parts; keep the text parts."""
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return str(content)


def engine_cmd(args, req, messages, stops):
    fd, path = tempfile.mkstemp(prefix="tl-msgs-")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write("\x1e".join(f"{m.get('role', 'user')}\x1f{content_text(m.get('content', ''))}" for m in messages))
    num = lambda k, d: float(req[k]) if req.get(k) is not None else d
    cmd = [args.bin, "run", "--model", args.model, "--tokenizer", args.tokenizer, "--messages", path,
           "-n", str(min(int(req.get("max_tokens") or args.max_tokens), 2048)),
           "-t", str(num("temperature", args.temperature)), "--top-p", str(num("top_p", 0.9)),
           "--repeat-penalty", str(num("repeat_penalty", args.repeat_penalty))]
    for s in stops:
        cmd += ["--stop", s]
    if args.threads:
        cmd += ["--threads", str(args.threads)]
    return cmd + args.engine_args.split(), path


def usage_from(stderr, wall_s, ttft_s):
    """Engine stats line -> OpenAI usage plus the Ollama-style fields Open WebUI displays."""
    m = STATS.search(stderr)
    if not m:
        return None
    pt, ptps, gt, gtps = int(m[1]), float(m[2]), int(m[3]), float(m[4])
    return {"prompt_tokens": pt, "completion_tokens": gt, "total_tokens": pt + gt,
            "prompt_token/s": round(ptps, 2), "response_token/s": round(gtps, 2),
            "time_to_first_token_ms": round(ttft_s * 1000) if ttft_s else None,
            "total_duration": round(wall_s, 2), "engine": "tinyllama-cpp"}


def cut_at_stop(text, stops):
    """(text before the earliest stop string, found?)"""
    idx = [text.find(s) for s in stops if s and s in text]
    return (text[:min(idx)], True) if idx else (text, False)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *a):
        print(f"[{time.strftime('%H:%M:%S')}] {fmt % a}", flush=True)

    def send_body(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, code, obj):
        self.send_body(code, json.dumps(obj).encode(), "application/json")

    def do_GET(self):
        path = self.path.split("?")[0].rstrip("/")
        if path in ("/v1/models", "/models"):
            self.send_json(200, {"object": "list", "data": [
                {"id": self.server.args.model_name, "object": "model", "owned_by": "tinyllama-cpp"}]})
        elif path in ("", "/dashboard"):
            self.send_body(200, (ROOT / "tools" / "dashboard.html").read_bytes(), "text/html; charset=utf-8")
        elif path == "/api/results":
            out = []
            for f in sorted((ROOT / "bench" / "results").glob("2*.json")):
                try:
                    d = json.loads(f.read_text())
                except (OSError, ValueError):
                    continue
                for r in d.get("results", {}).values():
                    r.pop("runs", None)   # per-run detail is large and not shown
                out.append(d)
            self.send_json(200, out)
        elif path == "/api/experiments":
            doc = ROOT / "docs" / "EXPERIMENTS.md"
            self.send_body(200, doc.read_bytes() if doc.exists() else b"", "text/markdown; charset=utf-8")
        else:
            self.send_json(404, {"error": {"message": "not found"}})

    def do_POST(self):
        if self.path.rstrip("/") not in ("/v1/chat/completions", "/chat/completions"):
            return self.send_json(404, {"error": {"message": "not found"}})
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            messages = req["messages"]
            if not isinstance(messages, list) or not messages:
                raise ValueError("messages must be a non-empty list")
        except (ValueError, KeyError, TypeError) as e:
            return self.send_json(400, {"error": {"message": f"bad request: {e}"}})

        a = self.server.args
        extra = req.get("stop") or []
        stops = DEFAULT_STOPS + ([extra] if isinstance(extra, str) else list(extra))
        cmd, path = engine_cmd(a, req, messages, stops)
        rid, created = f"chatcmpl-{uuid.uuid4().hex[:12]}", int(time.time())
        hold = max(len(s) for s in stops) - 1   # never emit a partial stop string

        def chunk(delta, finish=None, usage=None):
            c = {"id": rid, "object": "chat.completion.chunk", "created": created, "model": a.model_name,
                 "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
            if usage:
                c["usage"] = usage
            return c

        with LOCK:
            t0, ttft = time.time(), None
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=ROOT)
            try:
                stream = bool(req.get("stream"))
                if stream:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self.close_connection = True
                    send = lambda obj: (self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode()), self.wfile.flush())
                    send(chunk({"role": "assistant", "content": ""}))
                raw, text, sent, done = b"", "", 0, False
                while not done:
                    b = proc.stdout.read1(4096)
                    if not b:
                        break
                    ttft = ttft or time.time() - t0
                    raw += b
                    decoded = raw.decode("utf-8", errors="ignore")   # hold back a split UTF-8 tail
                    keep = len(raw) - len(decoded.encode("utf-8"))
                    raw = raw[len(raw) - keep:] if keep else b""
                    text += decoded
                    text, done = cut_at_stop(text, stops)
                    ready = len(text) if done else max(sent, len(text) - hold)
                    if stream and ready > sent:
                        send(chunk({"content": text[sent:ready]}))
                    sent = max(sent, ready)
                if not done:   # engine finished (EOS / length): flush the held-back tail
                    if stream and len(text) > sent:
                        send(chunk({"content": text[sent:]}))
                proc.wait()
                usage = usage_from(proc.stderr.read().decode(errors="replace"), time.time() - t0, ttft)
                if usage:
                    print("  engine:", usage, flush=True)
                if stream:
                    send(chunk({}, "stop", usage))
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                else:
                    self.send_json(200, {"id": rid, "object": "chat.completion", "created": created,
                                         "model": a.model_name, "usage": usage,
                                         "choices": [{"index": 0, "finish_reason": "stop",
                                                      "message": {"role": "assistant", "content": text}}]})
            except (BrokenPipeError, ConnectionResetError):
                pass   # client went away (e.g. Stop button): fall through and kill the engine
            finally:
                if proc.poll() is None:
                    proc.kill()
                proc.wait()
                os.unlink(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--bin", default=str(ROOT / "build/tinyllama"))
    ap.add_argument("--model", default=str(ROOT / "models/tinyllama-q4a.bin"))
    ap.add_argument("--tokenizer", default=str(ROOT / "models/tokenizer.bin"))
    ap.add_argument("--model-name", default="tinyllama-cpp")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.3, help="used when the request sets none")
    ap.add_argument("--repeat-penalty", type=float, default=1.1, help="Ollama's default; 1 = off")
    ap.add_argument("--engine-args", default="", help='extra engine flags, e.g. "--kv q8"')
    args = ap.parse_args()
    for p in (args.bin, args.model, args.tokenizer):
        if not os.path.exists(p):
            raise SystemExit(f"missing {p} (run `make build` / `make model WTYPE=q4a`)")
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    srv.args = args
    print(f"tinyllama OpenAI API on http://{args.host}:{args.port}/v1  model={os.path.basename(args.model)}"
          f" {args.engine_args}\ndashboard: http://{args.host}:{args.port}/dashboard", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
