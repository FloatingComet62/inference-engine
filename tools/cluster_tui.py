#!/usr/bin/env python3
"""Terminal UI for running tinyllama across machines on the LAN (`make cluster`).

Finds machines by scanning this machine's LAN subnet for SSH (port 22), then logs in to
each one (key-based SSH) to check what it has: the repo, the MPI build, the model, and
the PRTE version (every machine must match). From the list you can install your SSH key,
set a machine up (clone, link, build, copy the model), and run the pipeline or the
benchmark on the machines you select. Standard library only.

Keys: up/down move | space select | r rescan | u set SSH user | k install SSH key
      s set up machine | g generate | b benchmark | c clear output | q quit
"""
import concurrent.futures as cf
import curses
import ipaddress
import json
import os
import socket
import subprocess
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CONF_DIR = Path.home() / ".config" / "tinyllama-cluster"
USERS_FILE = CONF_DIR / "users.json"     # {ip: ssh user}
SSH_CONFIG = CONF_DIR / "ssh_config"     # host aliases handed to mpirun's ssh
WDIR = "/tmp/inference-engine"          # every rank starts here (a symlink to its repo)
KEY = Path.home() / ".ssh" / "id_ed25519"

# Run on each machine (and locally): one key=value per line describing its state.
PROBE = r"""
r=$(readlink -f /tmp/inference-engine 2>/dev/null)
if [ ! -e "$r/src/pipeline.cpp" ]; then
  r=$(find ~ -maxdepth 6 -path '*/src/pipeline.cpp' -not -path '*/.*' 2>/dev/null | head -1)
  r=${r%/src/pipeline.cpp}
fi
echo "host=$(hostname)"
echo "repo=$r"
[ -n "$r" ] && [ "$(readlink -f /tmp/inference-engine)" = "$r" ] && echo linked=1 || echo linked=0
[ -x "$r/build-mpi/tinyllama" ] && echo built=1 || echo built=0
[ -f "$r/models/tinyllama-q4a.bin" ] && [ -f "$r/models/tokenizer.bin" ] && echo model=1 || echo model=0
echo "prte=$(prted --version 2>/dev/null | head -1 | awk '{print $NF}')"
echo "cores=$(nproc)"
echo "ram=$(awk '/MemAvailable/ {printf "%.1f", $2 / 1e6}' /proc/meminfo)"
echo "cpu=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ *//; s/(R)//g; s/(TM)//g; s/ CPU//')"
"""


def sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def lan():
    """(this machine's LAN address, its network) from the default route."""
    out = sh(["ip", "-o", "-4", "route", "show", "default"]).stdout.split()
    dev = out[out.index("dev") + 1]
    addr = sh(["ip", "-o", "-4", "addr", "show", "dev", dev]).stdout.split()
    iface = ipaddress.ip_interface(addr[addr.index("inet") + 1])
    return str(iface.ip), iface.network


def local_mpi_home():
    homes = sorted(Path.home().glob(".local/openmpi-*"))
    return homes[-1] if homes else None


def local_prte():
    home = local_mpi_home()
    prted = str(home / "bin" / "prted") if home else "prted"
    out = sh([prted, "--version"]).stdout.splitlines()
    return out[0].split()[-1] if out else ""   # "prted (PRRTE) 3.0.8"


def ssh_open(ip, timeout=1.5):   # Wi-Fi round trips reach 200 ms; 254 probes at once add more
    try:
        with socket.create_connection((ip, 22), timeout=timeout):
            return True
    except OSError:
        return False


def ssh_base(user, ip):
    cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=4", "-o", "StrictHostKeyChecking=accept-new"]
    if KEY.exists():
        cmd += ["-i", str(KEY), "-o", "IdentitiesOnly=yes"]
    return cmd + [f"{user}@{ip}"]


def default_user(ip):
    """The user ~/.ssh/config gives this address (else the local user)."""
    for line in sh(["ssh", "-G", ip]).stdout.splitlines():
        if line.startswith("user "):
            return line.split(None, 1)[1]
    return os.environ.get("USER", "")


def parse(out):
    return dict(line.split("=", 1) for line in out.splitlines() if "=" in line)


class Node:
    def __init__(self, ip, user, local=False):
        self.ip, self.user, self.local = ip, user, local
        self.info, self.state, self.selected = {}, "checking", local

    def status(self, want_prte):
        if self.state != "ok":
            return self.state
        i = self.info
        if not i.get("prte"):
            return "no Open MPI"
        if want_prte and i["prte"] != want_prte:
            return f"PRTE {i['prte']} != {want_prte}"
        missing = [k for k in ("repo", "linked", "built", "model") if i.get(k) in ("", "0", None)]
        return "ready" if not missing else "setup: " + ",".join(missing)


class App:
    def __init__(self):
        CONF_DIR.mkdir(parents=True, exist_ok=True)
        self.users = json.loads(USERS_FILE.read_text()) if USERS_FILE.exists() else {}
        self.me = os.environ.get("USER", "")
        self.my_ip, self.net = lan()
        self.want_prte = local_prte()
        self.nodes, self.cur = [], 0
        self.out, self.busy, self.msg = [], None, ""
        self.lock = threading.Lock()

    # ---------- discovery ----------
    def scan(self):
        self.msg = f"scanning {self.net} for SSH..."
        hosts = list(self.net.hosts())
        if len(hosts) > 4096:   # very large subnet: only this machine's /24
            hosts = list(ipaddress.ip_network(self.my_ip + "/24", strict=False).hosts())
        neigh = sh(["ip", "-4", "neigh"]).stdout.split()
        first = [h for h in hosts if str(h) in neigh]   # known neighbours first
        with cf.ThreadPoolExecutor(256) as ex:
            found = [str(h) for h, ok in zip(first + hosts, ex.map(lambda h: ssh_open(str(h)), first + hosts)) if ok]
            # second, slower try for known neighbours that missed the burst
            missed = [str(h) for h in first if str(h) not in found]
            found += [ip for ip, ok in zip(missed, ex.map(lambda ip: ssh_open(ip, 3.0), missed)) if ok]
        found = sorted(set(found) - {self.my_ip}, key=lambda s: ipaddress.ip_address(s))
        old = {n.ip: n for n in self.nodes}
        self.nodes = [old.get(self.my_ip) or Node(self.my_ip, self.me, local=True)]
        self.nodes += [old.get(ip) or Node(ip, self.users.get(ip) or default_user(ip)) for ip in found]
        self.msg = f"{len(found)} other machine(s) with SSH on {self.net}"
        for n in self.nodes:
            threading.Thread(target=self.probe, args=(n,), daemon=True).start()

    def probe(self, n):
        n.state = "checking"
        cmd = ["bash", "-c", PROBE] if n.local else ssh_base(n.user, n.ip) + ["bash -s"]
        try:
            r = sh(cmd, input=None if n.local else PROBE, timeout=40)
        except subprocess.TimeoutExpired:
            n.state = "timeout"
            return
        if r.returncode != 0 and not r.stdout:
            err = r.stderr.lower()
            n.state = "needs SSH key (k) or user (u)" if "permission denied" in err else "unreachable"
            return
        n.info = parse(r.stdout)
        if n.local:
            n.info["prte"] = self.want_prte or n.info.get("prte", "")
        n.state = "ok"

    # ---------- actions ----------
    def write_ssh_config(self, nodes):
        """Host aliases (hostname -> ip, user) for mpirun's ssh, then the user's own config."""
        lines = []
        for n in nodes:
            if n.local:
                continue
            lines += [f"Host {n.info['host']}", f"    HostName {n.ip}", f"    User {n.user}"]
            if KEY.exists():
                lines += [f"    IdentityFile {KEY}", "    IdentitiesOnly yes"]
        lines += ["Match all", "    Include ~/.ssh/config", ""]
        SSH_CONFIG.write_text("\n".join(lines))

    def run(self, title, cmd, env=None, after=None):
        if self.busy:
            self.msg = "busy: " + self.busy
            return
        self.busy = title
        self.log(f"$ {' '.join(cmd) if isinstance(cmd, list) else cmd}")

        def go():
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                 cwd=REPO, env={**os.environ, **(env or {})}, shell=isinstance(cmd, str))
            for line in p.stdout:
                self.log(line.rstrip())
            p.wait()
            self.log(f"[{title}: {'done' if p.returncode == 0 else 'failed, exit %d' % p.returncode}]")
            self.busy = None
            if after:
                after()
        threading.Thread(target=go, daemon=True).start()

    def log(self, line):
        with self.lock:
            self.out.append(line)
            self.out = self.out[-500:]

    def mpi_env(self, sel):
        self.write_ssh_config(sel)
        env = {"PRTE_MCA_plm_ssh_agent": f"ssh -F {SSH_CONFIG}"}
        return env

    def launch(self, target, extra):
        sel = [n for n in self.nodes if n.selected]
        bad = [n for n in sel if n.status(self.want_prte) != "ready"]
        if bad:
            self.msg = f"not ready: {bad[0].info.get('host', bad[0].ip)} ({bad[0].status(self.want_prte)})"
            return
        hosts = ",".join(n.info["host"] + ":1" for n in sel)
        args = ["make", target, f"NP={len(sel)}"] + ([f"MPI_HOSTS={hosts}"] if len(sel) > 1 else ["MPI_IFACE=lo"]) + extra
        self.run(target, args, env=self.mpi_env(sel))

    def setup(self, n):
        """Clone if needed, link /tmp/inference-engine, build, copy the model if missing."""
        if n.state != "ok":
            self.msg = f"{n.ip}: {n.state}"
            return
        i = n.info
        origin = sh(["git", "-C", str(REPO), "remote", "get-url", "origin"]).stdout.strip()
        repo = i.get("repo") or "~/inference-engine"
        remote = (f"set -e; [ -e {repo}/src/pipeline.cpp ] || git clone {origin} {repo}; "
                  f"ln -sfn $(cd {repo} && pwd) {WDIR}; cd {repo}; make mpi-build")
        cmd = f"bash -c {sh_quote(remote)}" if n.local else " ".join(map(sh_quote, ssh_base(n.user, n.ip) + [remote]))
        if i.get("model") != "1" and not n.local:
            dest = f"{n.user}@{n.ip}:{repo}/models/"
            files = " ".join(str(REPO / "models" / f) for f in ("tinyllama-q4a.bin", "tokenizer.bin"))
            scp = ["scp", "-o", "BatchMode=yes"] + (["-i", str(KEY)] if KEY.exists() else [])
            cmd += f" && {' '.join(map(sh_quote, scp))} {files} {sh_quote(dest)}"
        self.run(f"set up {i.get('host', n.ip)}", cmd, after=lambda: self.probe(n))

    def install_key(self, scr, n):
        if not KEY.exists():
            subprocess.run(["ssh-keygen", "-t", "ed25519", "-N", "", "-f", str(KEY)])
        curses.endwin()
        print(f"\nInstalling {KEY}.pub on {n.user}@{n.ip} (enter that machine's password)\n")
        subprocess.run(["ssh-copy-id", "-i", f"{KEY}.pub", f"{n.user}@{n.ip}"])
        input("\nPress Enter to return...")
        scr.refresh()
        threading.Thread(target=self.probe, args=(n,), daemon=True).start()

    # ---------- UI ----------
    def ask(self, scr, prompt, default=""):
        h, w = scr.getmaxyx()
        curses.echo()
        curses.curs_set(1)
        scr.addstr(h - 1, 0, (f"{prompt} [{default}]: ")[: w - 1].ljust(w - 1), curses.A_REVERSE)
        scr.move(h - 1, min(w - 2, len(prompt) + len(default) + 5))
        scr.nodelay(False)
        val = scr.getstr().decode().strip()
        scr.nodelay(True)
        curses.noecho()
        curses.curs_set(0)
        return val or default

    def draw(self, scr):
        scr.erase()
        h, w = scr.getmaxyx()
        def put(y, x, s, a=0):   # clipped to the window; curses errors on the last cell are ignored
            if 0 <= y < h and x < w - 1 and s:
                try:
                    scr.addstr(y, x, s[: w - x - 1], a)
                except curses.error:
                    pass
        put(0, 0, f" tinyllama cluster | {self.my_ip} on {self.net} | PRTE needed: {self.want_prte or '?'} ",
            curses.A_REVERSE)
        put(1, 0, "  sel host            status                         ip               user       cores ram GB  cpu")
        for k, n in enumerate(self.nodes):
            i = n.info
            st = n.status(self.want_prte)
            color = curses.color_pair(1 if st == "ready" else 3 if st == "checking" else 2)
            hi = curses.A_REVERSE if k == self.cur else 0
            head = f"  [{'x' if n.selected else ' '}] {(i.get('host', '?') + (' (this)' if n.local else ''))[:15]:15} "
            put(2 + k, 0, head, hi)
            put(2 + k, len(head), f"{st[:30]:30} ", color | hi)
            put(2 + k, len(head) + 31, f"{n.ip:16} {n.user[:10]:10} {i.get('cores', ''):>5} {i.get('ram', ''):>6}  "
                f"{i.get('cpu', '')}", hi)
        top = 3 + len(self.nodes)
        put(top, 0, "-" * (w - 1))
        with self.lock:
            lines = self.out[-(h - top - 3):]
        for k, line in enumerate(lines):
            put(top + 1 + k, 0, line)
        put(h - 2, 0, (self.msg + (f"   [running: {self.busy}]" if self.busy else "")).ljust(w - 1), curses.A_BOLD)
        put(h - 1, 0, " space select  r rescan  u user  k ssh key  s set up  g generate  b benchmark  c clear  q quit "
            .ljust(w - 1), curses.A_REVERSE)
        scr.refresh()

    def main(self, scr):
        curses.curs_set(0)
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_GREEN, -1)
        curses.init_pair(2, curses.COLOR_RED, -1)
        curses.init_pair(3, curses.COLOR_YELLOW, -1)
        scr.nodelay(True)
        threading.Thread(target=self.scan, daemon=True).start()
        while True:
            self.draw(scr)
            c = scr.getch()
            if c == -1:
                time.sleep(0.1)
                continue
            n = self.nodes[self.cur] if self.nodes else None
            if c in (ord("q"), 27):
                return
            elif c == curses.KEY_UP:
                self.cur = max(0, self.cur - 1)
            elif c == curses.KEY_DOWN:
                self.cur = min(len(self.nodes) - 1, self.cur + 1)
            elif c == ord(" ") and n and not n.local:
                n.selected = not n.selected
            elif c == ord("r"):
                threading.Thread(target=self.scan, daemon=True).start()
            elif c == ord("c"):
                with self.lock:
                    self.out = []
            elif c == ord("u") and n and not n.local:
                n.user = self.ask(scr, f"SSH user for {n.ip}", n.user)
                self.users[n.ip] = n.user
                USERS_FILE.write_text(json.dumps(self.users, indent=1))
                threading.Thread(target=self.probe, args=(n,), daemon=True).start()
            elif c == ord("k") and n and not n.local:
                self.install_key(scr, n)
            elif c == ord("s") and n:
                self.setup(n)
            elif c == ord("g"):
                prompt = self.ask(scr, "prompt", "Explain how a CPU cache works, in simple terms.")
                streams = self.ask(scr, "requests", "4")
                micro = self.ask(scr, "requests per batch", "4")
                ntok = self.ask(scr, "tokens per request", "64")
                self.launch("mpi-run", [f"PROMPT={prompt}", f"STREAMS={streams}", f"MICRO={micro}", f"N={ntok}"])
            elif c == ord("b"):
                ntok = self.ask(scr, "tokens per request", "48")
                self.launch("mpi-bench", [f"N={ntok}"])


def sh_quote(s):
    return "'" + str(s).replace("'", "'\\''") + "'"


if __name__ == "__main__":
    app = App()
    curses.wrapper(app.main)
