#!/usr/bin/env python3
"""Minimal OpenAI-compatible HTTP API for the tinyllama engine (standard library only).

Serves GET /v1/models and POST /v1/chat/completions (streaming and non-streaming) so
chat UIs such as Open WebUI can talk to the engine. Each request runs `tinyllama run`
with the whole conversation (--messages); weights are mmap'd, so loading is ~instant.
One request at a time: the engine already uses every core.

  python3 tools/openai_server.py --port 8000 --engine-args "--kv q8"
"""
import argparse
import json
import os
import subprocess
import tempfile
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCK = threading.Lock()   # ponytail: one generation at a time; a queue only if concurrent users matter


def engine_cmd(args, messages, max_tokens, temperature, top_p):
    fd, path = tempfile.mkstemp(prefix="tl-msgs-")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write("\x1e".join(f"{m.get('role', 'user')}\x1f{content_text(m.get('content', ''))}"
                            for m in messages))
    cmd = [args.bin, "run", "--model", args.model, "--tokenizer", args.tokenizer, "--messages", path,
           "-n", str(max_tokens), "-t", str(temperature), "--top-p", str(top_p)]
    if args.threads:
        cmd += ["--threads", str(args.threads)]
    return cmd + args.engine_args.split(), path


def content_text(content):
    """OpenAI content is a string or a list of parts; keep the text parts."""
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return str(content)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *a):
        print(f"[{time.strftime('%H:%M:%S')}] {fmt % a}", flush=True)

    def send_json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/") in ("/v1/models", "/models"):
            self.send_json(200, {"object": "list", "data": [
                {"id": self.server.args.model_name, "object": "model", "owned_by": "tinyllama-cpp"}]})
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
        max_tokens = min(int(req.get("max_tokens") or a.max_tokens), 2048)
        temperature = float(req.get("temperature", 0.7) if req.get("temperature") is not None else 0.7)
        top_p = float(req.get("top_p") or 0.9)
        cmd, path = engine_cmd(a, messages, max_tokens, temperature, top_p)
        rid, created = f"chatcmpl-{uuid.uuid4().hex[:12]}", int(time.time())

        def chunk(delta, finish=None):
            return {"id": rid, "object": "chat.completion.chunk", "created": created, "model": a.model_name,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

        with LOCK:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=ROOT)
            try:
                if req.get("stream"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self.close_connection = True
                    send = lambda obj: (self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode()), self.wfile.flush())
                    send(chunk({"role": "assistant", "content": ""}))
                    buf = b""
                    while True:
                        b = proc.stdout.read1(4096)
                        if not b:
                            break
                        buf += b
                        text = buf.decode("utf-8", errors="ignore")   # hold back a split UTF-8 tail
                        keep = len(buf) - len(text.encode("utf-8"))
                        buf = buf[len(buf) - keep:] if keep else b""
                        if text:
                            send(chunk({"content": text}))
                    send(chunk({}, "stop"))
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                else:
                    out = proc.stdout.read().decode("utf-8", errors="replace")
                    self.send_json(200, {"id": rid, "object": "chat.completion", "created": created,
                                         "model": a.model_name, "choices": [{"index": 0, "finish_reason": "stop",
                                         "message": {"role": "assistant", "content": out}}]})
            except (BrokenPipeError, ConnectionResetError):
                pass   # client went away (e.g. Stop button): fall through and kill the engine
            finally:
                if proc.poll() is None:
                    proc.kill()
                proc.wait()
                stats = proc.stderr.read().decode(errors="replace").strip().splitlines()
                if stats:
                    print("  engine:", stats[-1], flush=True)
                os.unlink(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--bin", default=os.path.join(ROOT, "build/tinyllama"))
    ap.add_argument("--model", default=os.path.join(ROOT, "models/tinyllama-q4a.bin"))
    ap.add_argument("--tokenizer", default=os.path.join(ROOT, "models/tokenizer.bin"))
    ap.add_argument("--model-name", default="tinyllama-cpp")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--engine-args", default="", help='extra engine flags, e.g. "--kv q8"')
    args = ap.parse_args()
    for p in (args.bin, args.model, args.tokenizer):
        if not os.path.exists(p):
            raise SystemExit(f"missing {p} (run `make build` / `make model WTYPE=q4a`)")
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    srv.args = args
    print(f"tinyllama OpenAI API on http://{args.host}:{args.port}/v1  model={os.path.basename(args.model)}"
          f" {args.engine_args}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
