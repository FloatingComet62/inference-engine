#!/usr/bin/env bash
# Interactive installer for the TinyLlama CPU engine.
#
# Asks which features you need and installs exactly those dependencies, nothing
# more. Safe to re-run: anything already installed (system packages, binaries,
# Ollama models, model weights) is skipped. Pass --reinstall to force
# re-installs / re-downloads.
#
#   core   build toolchain every feature needs (cmake >= 3.16, C++ compiler,
#          make, python3 >= 3.8, curl; python3-venv on Debian/Ubuntu; libomp
#          on macOS)
#   model  TinyLlama-1.1B-Chat weights (~2.2 GB from Hugging Face)
#   mpi    Open MPI + ssh client for the multi-machine pipeline (make mpi-run)
#   ollama Ollama plus the tinyllama:1.1b model (make compare)
#   webui  uv, which `make webui` uses to run Open WebUI
#
# Usage: ./install.sh [options]
#   -y, --yes          no prompts: core only unless feature flags are given
#   --reinstall        re-install / re-download even if already present
#   --dry-run          print the plan and exit without changing anything
#   --model/--no-model --mpi/--no-mpi --ollama/--no-ollama --webui/--no-webui
#                      pick features non-interactively
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"          # repo root

OLLAMA_MODEL="tinyllama:1.1b"
UV_INSTALLER="https://astral.sh/uv/install.sh"
OLLAMA_INSTALLER="https://ollama.com/install.sh"

# ---------------------------------------------------------------- options
YES=0 REINSTALL=0 DRY=0
INSTALL_MODEL=0 INSTALL_MPI=0 INSTALL_OLLAMA=0 INSTALL_WEBUI=0
EXPLICIT_MODEL=0 EXPLICIT_MPI=0 EXPLICIT_OLLAMA=0 EXPLICIT_WEBUI=0

usage() {
  sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
  exit 0
}
die() { echo "error: $*" >&2; exit 1; }

while (($#)); do
  case "$1" in
    -y|--yes)        YES=1 ;;
    --reinstall)     REINSTALL=1 ;;
    --dry-run)       DRY=1 ;;
    --model)         INSTALL_MODEL=1   EXPLICIT_MODEL=1 ;;
    --no-model)      INSTALL_MODEL=0   EXPLICIT_MODEL=1 ;;
    --mpi)           INSTALL_MPI=1     EXPLICIT_MPI=1 ;;
    --no-mpi)        INSTALL_MPI=0     EXPLICIT_MPI=1 ;;
    --ollama)        INSTALL_OLLAMA=1  EXPLICIT_OLLAMA=1 ;;
    --no-ollama)     INSTALL_OLLAMA=0  EXPLICIT_OLLAMA=1 ;;
    --webui)         INSTALL_WEBUI=1   EXPLICIT_WEBUI=1 ;;
    --no-webui)      INSTALL_WEBUI=0   EXPLICIT_WEBUI=1 ;;
    -h|--help)       usage ;;
    *) die "unknown option: $1 (try --help)" ;;
  esac
  shift
done

# ---------------------------------------------------------------- prompts
prompt_yn() {   # question default(y|n); returns 0 = yes
  local q=$1 d=$2 ans
  while true; do
    if [[ "$d" == y ]]; then printf '%s [Y/n] ' "$q"; else printf '%s [y/N] ' "$q"; fi
    read -r ans || { echo; die "no input"; }
    [[ -z "$ans" ]] && ans=$d
    [[ "$ans" == [yY]* ]] && return 0
    [[ "$ans" == [nN]* ]] && return 1
    echo "  please answer y or n"
  done
}

if (( ! YES )); then
  echo "TinyLlama CPU engine - dependency installer"
  echo "Pick the features you need; anything already installed is skipped."
  echo
  if (( ! EXPLICIT_MODEL )); then
    if prompt_yn "Download the TinyLlama model weights (about 2.2 GB)?" y; then INSTALL_MODEL=1; fi
  fi
  if (( ! EXPLICIT_MPI )); then
    if prompt_yn "Multi-machine MPI pipeline (make mpi-run)?" n; then INSTALL_MPI=1; fi
  fi
  if (( ! EXPLICIT_OLLAMA )); then
    if prompt_yn "Ollama + tinyllama:1.1b to benchmark against (make compare)?" n; then INSTALL_OLLAMA=1; fi
  fi
  if (( ! EXPLICIT_WEBUI )); then
    if prompt_yn "Open WebUI chat interface (make up)?" n; then INSTALL_WEBUI=1; fi
  fi
fi

# ---------------------------------------------------------------- platform
PM="${INSTALL_PM:-}"
if [[ -z "$PM" ]]; then
  case "$(uname -s)" in
    Darwin) command -v brew >/dev/null || die "macOS: Homebrew is required (https://brew.sh)"
            PM=brew ;;
    *) for c in apt-get dnf pacman; do command -v "$c" >/dev/null && { PM=${c%-*}; break; }; done ;;
  esac
fi
case "$PM" in
  apt|dnf|pacman|brew) ;;
  *) die "unsupported package manager '$PM' (supported: apt, dnf, pacman, brew; set INSTALL_PM to override)" ;;
esac

if (( ! DRY )); then
  if [[ "$(id -u)" -eq 0 ]]; then SUDO=""
  elif command -v sudo >/dev/null; then SUDO="sudo"
  else die "root access needed to install system packages (no sudo found)"; fi
else
  SUDO="sudo"   # shown in the plan only; never executed
fi

# ---------------------------------------------------------------- helpers
have_cmd() { command -v "$1" >/dev/null 2>&1; }

pkg_installed() {   # is a system package installed?
  case "$PM" in
    apt)    dpkg -s "$1" 2>/dev/null | grep -q '^Status: install ok installed' ;;
    dnf)    rpm -q "$1" >/dev/null 2>&1 ;;
    pacman) pacman -Q "$1" >/dev/null 2>&1 ;;
    brew)   brew list --versions "$1" >/dev/null 2>&1 ;;
  esac
}

ver_ge() {   # is $1 >= $2? dotted versions, portable (no sort -V)
  local a=$1 b=$2
  [[ "$a" == "$b" ]] && return 0
  local IFS=. ai bi i
  read -ra ai <<< "$a"
  read -ra bi <<< "$b"
  for ((i = 0; i < ${#bi[@]}; i++)); do
    if (( ${ai[i]:-0} > ${bi[i]} )); then return 0; fi
    if (( ${ai[i]:-0} < ${bi[i]} )); then return 1; fi
  done
  return 0
}
cmake_ok()   { have_cmd cmake && ver_ge "$(cmake --version 2>/dev/null | awk 'NR==1{print $3}')" "3.16"; }
python3_ok() { have_cmd python3 && ver_ge "$(python3 --version 2>&1 | awk '{print $2}')" "3.8"; }
cxx_ok()     { have_cmd c++ || have_cmd g++ || have_cmd clang++; }
# Fedora keeps MPI off $PATH (/usr/lib64/openmpi/bin, or `module load mpi/openmpi-x86_64`)
have_mpirun() { have_cmd mpirun || [[ -x /usr/lib64/openmpi/bin/mpirun ]] || [[ -x /usr/lib64/mpi/gcc/openmpi/bin/mpirun ]]; }
have_uv()    { have_cmd uv || [[ -x "$HOME/.local/bin/uv" ]]; }
have_ollama_model() { have_cmd ollama && ollama list 2>/dev/null | awk -v m="$OLLAMA_MODEL" '$1 == m {f=1} END {exit !f}'; }

# feature -> "pkg cmd" lines (cmd = binary to check instead of the package;
# "-" = package only). Names are real package names per package manager.
core_deps() {
  case "$PM" in
    apt)    echo "cmake -"; echo "g++ c++"; echo "make make"; echo "python3 -"; echo "python3-venv -"; echo "curl curl" ;;
    dnf)    echo "cmake -"; echo "gcc-c++ c++"; echo "make make"; echo "python3 -"; echo "curl curl" ;;
    pacman) echo "cmake -"; echo "gcc c++"; echo "make make"; echo "python -"; echo "python-venv -"; echo "curl curl" ;;
    brew)   echo "cmake -"; echo "gcc c++"; echo "make make"; echo "python3 -"; echo "libomp -"; echo "curl curl" ;;
  esac
}
mpi_deps() {
  case "$PM" in
    apt)    echo "openmpi-bin mpirun"; echo "libopenmpi-dev -"; echo "openssh-client ssh" ;;
    dnf)    echo "openmpi mpirun";     echo "openmpi-devel -";  echo "openssh-clients ssh" ;;
    pacman) echo "openmpi mpirun";     echo "openssh ssh" ;;
    brew)   echo "open-mpi mpirun" ;;
  esac
}

dep_ok() {   # $1 pkg $2 cmd; 0 = already satisfied
  local pkg=$1 cmd=$2
  case "$pkg" in
    cmake)          cmake_ok ;;
    python3|python) python3_ok ;;
    g++|gcc-c++|gcc) cxx_ok ;;
    *) if [[ "$cmd" == mpirun ]]; then have_mpirun
       elif [[ -n "$cmd" ]] && [[ "$cmd" != "-" ]]; then have_cmd "$cmd"
       else pkg_installed "$pkg"; fi ;;
  esac
}

# ---------------------------------------------------------------- plan
NEED=()     # package names or __action__ placeholders, in install order
PLAN=()     # human-readable summary lines

pending_pkgs() {   # append a feature's "pkg cmd" lines to NEED when missing
  local line pkg cmd
  while read -r line; do
    [[ -z "$line" ]] && continue
    pkg=${line%% *}; cmd=${line#* }
    if (( REINSTALL )); then NEED+=("$pkg")
    elif ! dep_ok "$pkg" "$cmd"; then NEED+=("$pkg"); fi
  done <<< "$1"
}

pending_pkgs "$(core_deps)"
if (( INSTALL_MPI )); then pending_pkgs "$(mpi_deps)"; fi
if (( INSTALL_OLLAMA )) && { (( REINSTALL )) || ! have_cmd ollama; }; then NEED+=(__ollama__); fi
if (( INSTALL_OLLAMA )) && { (( REINSTALL )) || ! have_ollama_model; }; then NEED+=(__ollama_model__); fi
if (( INSTALL_WEBUI )) && { (( REINSTALL )) || ! have_uv; }; then NEED+=(__uv__); fi
if (( INSTALL_MODEL )) && { (( REINSTALL )) || [[ ! -s models/model.safetensors ]]; }; then NEED+=(__model__); fi

declare -A DESCR=(
  [__ollama__]="ollama binary (official installer)"
  [__ollama_model__]="ollama model $OLLAMA_MODEL (pull)"
  [__uv__]="uv (official installer, ~/.local/bin)"
  [__model__]="TinyLlama weights (~2.2 GB) into models/"
)
for item in "${NEED[@]}"; do
  if [[ "$item" == __*__ ]]; then PLAN+=("  ${DESCR[$item]}")
  else PLAN+=("  $item"); fi
done

if (( ${#NEED[@]} == 0 )); then
  echo
  echo "Everything you selected is already installed - nothing to do."
  echo "(use --reinstall to force re-installs / re-downloads)"
  exit 0
fi

echo
echo "$PM install plan (using $SUDO for system packages):"
printf '%s\n' "${PLAN[@]}"

if (( DRY )); then
  echo "(--dry-run: plan only, nothing changed.)"
  exit 0
fi

if (( ! YES )); then
  prompt_yn "Proceed with the install?" y || { echo "aborted"; exit 1; }
fi

# ---------------------------------------------------------------- install
install_pkg() {   # $1 real package name
  echo "  [$PM] install $1"
  case "$PM" in
    apt)    if (( ! apt_updated )); then $SUDO apt-get update; apt_updated=1; fi
            $SUDO apt-get install -y "$1" ;;
    dnf)    $SUDO dnf install -y "$1" ;;
    pacman) $SUDO pacman -S --noconfirm --needed "$1" ;;
    brew)   brew install "$1" ;;
  esac
}
reinstall_pkg() {   # force reinstall of an already-installed package
  echo "  [$PM] reinstall $1"
  case "$PM" in
    apt)    if (( ! apt_updated )); then $SUDO apt-get update; apt_updated=1; fi
            $SUDO apt-get install --reinstall -y "$1" ;;
    dnf)    $SUDO dnf reinstall -y "$1" ;;
    pacman) $SUDO pacman -S --noconfirm "$1" ;;
    brew)   brew reinstall "$1" ;;
  esac
}

apt_updated=0
for item in "${NEED[@]}"; do
  case "$item" in
    __ollama__)
      echo "  [ollama] installing from $OLLAMA_INSTALLER"
      tmp=$(mktemp)
      curl -fsSL "$OLLAMA_INSTALLER" -o "$tmp"
      $SUDO sh "$tmp"
      rm -f "$tmp"
      ;;
    __ollama_model__)
      echo "  [ollama] pulling $OLLAMA_MODEL"
      if (( REINSTALL )); then ollama rm "$OLLAMA_MODEL" >/dev/null 2>&1 || true; fi
      ollama pull "$OLLAMA_MODEL"
      ;;
    __uv__)
      echo "  [uv] installing from $UV_INSTALLER"
      curl -LsSf "$UV_INSTALLER" | sh
      ;;
    __model__)
      echo "  [model] downloading TinyLlama-1.1B-Chat into models/ (~2.2 GB)"
      if (( REINSTALL )); then rm -f models/model.safetensors; fi
      scripts/download.sh models
      ;;
    *)
      if (( REINSTALL )) && pkg_installed "$item"; then reinstall_pkg "$item"
      else install_pkg "$item"; fi
      ;;
  esac
done

# ---------------------------------------------------------------- verify
echo
echo "Verifying..."
fail=0
checkv() {   # $1 label  $2... command (must succeed)
  local label=$1; shift
  if "$@" >/dev/null 2>&1; then echo "  ok  $label"; else echo "  MISSING  $label"; fail=1; fi
}
checkv "cmake (>= 3.16)"  cmake_ok
checkv "C++ compiler"     cxx_ok
checkv "make"             have_cmd make
checkv "python3 (>= 3.8)" python3_ok
checkv "curl"             have_cmd curl
if (( INSTALL_MPI )); then checkv "MPI (mpirun)" have_mpirun; fi
if (( INSTALL_OLLAMA )); then
  checkv "ollama" have_cmd ollama
  checkv "$OLLAMA_MODEL" have_ollama_model
fi
if (( INSTALL_WEBUI )); then checkv "uv" have_uv; fi
if (( INSTALL_MODEL )); then
  if [[ -s models/model.safetensors ]]; then echo "  ok  model weights"
  else echo "  MISSING  model weights"; fail=1; fi
fi
if (( fail )); then die "some checks failed - see above"; fi

# ---------------------------------------------------------------- next steps
echo
echo "Done. Next steps:"
echo "  make setup    # build the engine, convert + quantize the model (once)"
echo "  make run      # generate a first answer"
if (( INSTALL_OLLAMA )); then echo "  make compare  # benchmark this engine vs Ollama"; fi
if (( INSTALL_MPI )); then
  echo "  make mpi-build   # build the MPI variant; see docs/EXPERIMENTS.md for hosts"
  if [[ "$PM" == dnf ]]; then
    echo "  (Fedora: add /usr/lib64/openmpi/bin to PATH or: module load mpi/openmpi-x86_64)"
  fi
fi
if (( INSTALL_WEBUI )); then
  echo "  make up       # start Open WebUI + this engine's API (first start takes ~1 min)"
fi