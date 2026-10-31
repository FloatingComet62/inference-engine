#!/usr/bin/env bash
# Checks the tools the project needs.
ok=1
check() {
  if command -v "$1" >/dev/null 2>&1; then printf "  ok       %-8s %s\n" "$1" "$($1 --version 2>&1 | head -n1)"
  else printf "  MISSING  %-8s %s\n" "$1" "$2"; [[ "$3" == "required" ]] && ok=0; fi
}
echo "checking tools"
check cmake   "install cmake >= 3.16"            required
check c++     "install g++ or clang++"           required
check python3 "install python 3.8+"              required
check curl    "install curl"                     required
check ollama  "optional, needed for make compare" optional
if grep -q avx2 /proc/cpuinfo 2>/dev/null; then echo "  ok       CPU supports AVX2"
elif [[ "$(uname -m)" == "x86_64" ]]; then echo "  note     could not confirm AVX2 support"; fi
[[ $ok == 1 ]] || { echo "install the missing tools and re-run"; exit 1; }
