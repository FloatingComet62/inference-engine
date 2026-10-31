#!/usr/bin/env bash

if [[ "$(uname)" == "Darwin" ]]; then
  sysctl -n hw.logicalcpu
  exit
fi

nproc 2>/dev/null || echo 4
