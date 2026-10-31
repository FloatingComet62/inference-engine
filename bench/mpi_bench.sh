#!/usr/bin/env bash
# Multi-machine benchmark sweep: aggregate and per-stream tok/s over streams x micro-batch,
# one JSON line per run appended to bench/results/mpi.jsonl (plus the settings used).
# Usage: bench/mpi_bench.sh <np> <iface> [hosts] [n]
#   np     MPI ranks (machines x ranks per machine)
#   iface  LAN subnet (e.g. 192.168.1.0/24, see scripts/lan_subnet.sh) or interface; lo = one machine
#   hosts  mpirun --host list of LAN IPs, e.g. "192.168.1.10:1,192.168.1.11:1" (empty = this machine)
set -euo pipefail
NP=${1:?np}; IFACE=${2:?iface}; HOSTS=${3:-}; N=${4:-64}
OUT=bench/results/mpi.jsonl
mkdir -p bench/results
MPI=(mpirun -np "$NP" --mca btl tcp,self --mca btl_tcp_if_include "$IFACE")
[ "$IFACE" != lo ] && MPI+=(--mca oob_tcp_if_include "$IFACE")   # control channel refuses loopback
[ -n "$HOSTS" ] && MPI+=(--host "$HOSTS")
for cfg in "1 1" "4 1" "4 4" "8 4" "8 8" "16 8"; do
    set -- $cfg
    line=$("${MPI[@]}" build-mpi/tinyllama pipe --model models/tinyllama-q4a.bin --tokenizer models/tokenizer.bin \
        -n "$N" --streams "$1" --micro "$2" 2>/dev/null | grep '^{')
    line=${line%\}}", \"iface\": \"$IFACE\", \"date\": \"$(date -Iseconds)\"}"
    echo "$line" >> "$OUT"
    python3 -c 'import json, sys
d = json.loads(sys.argv[1])
print("ranks %(ranks)d streams %(streams)2d micro %(micro)2d: aggregate %(aggregate_tps)6.1f tok/s, per stream %(per_stream_tps)5.1f" % d,
      "| layers", d["layers"], "rtt ms", d["rtt_ms"], "busy %", d["busy_pct"])' "$line"
done
echo "appended to $OUT"
