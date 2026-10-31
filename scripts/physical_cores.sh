#!/usr/bin/env bash
# Prints the number of physical cores (hyperthreads share memory bandwidth, so
# one thread per physical core is usually fastest for decode).
if [[ "$(uname)" == "Darwin" ]]; then sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || sysctl -n hw.physicalcpu; exit; fi
if command -v lscpu >/dev/null 2>&1; then
  n=$(lscpu -p=Core,Socket 2>/dev/null | grep -v '^#' | sort -u | wc -l)
  if [[ "$n" -gt 0 ]]; then echo "$n"; exit; fi
fi
nproc 2>/dev/null || echo 4
