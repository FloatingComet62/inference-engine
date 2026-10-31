#!/usr/bin/env bash
# Default worker thread count.
#   Hybrid Intel CPUs (P-cores + E-cores): the P-cores' hardware threads only, like
#   llama.cpp's default. Adding E-cores makes every barrier wait for the slowest core
#   (measured: 24 threads 39 tok/s vs 16 threads 66 tok/s on a hybrid i9). The kernel
#   lists the core types in /sys/devices/cpu_core and /sys/devices/cpu_atom.
#   Other CPUs: one thread per physical core (hyperthreads share memory bandwidth).
if [[ "$(uname)" == "Darwin" ]]; then sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || sysctl -n hw.physicalcpu; exit; fi
if [[ -r /sys/devices/cpu_atom/cpus && -r /sys/devices/cpu_core/cpus ]]; then
  n=0
  IFS=, read -ra parts < /sys/devices/cpu_core/cpus
  for p in "${parts[@]}"; do
    if [[ "$p" == *-* ]]; then n=$(( n + ${p#*-} - ${p%-*} + 1 )); else n=$(( n + 1 )); fi
  done
  echo "$n"; exit
fi
if command -v lscpu >/dev/null 2>&1; then
  n=$(lscpu -p=Core,Socket 2>/dev/null | grep -v '^#' | sort -u | wc -l)
  if [[ "$n" -gt 0 ]]; then echo "$n"; exit; fi
fi
nproc 2>/dev/null || echo 4
