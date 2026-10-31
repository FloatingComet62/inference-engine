# One-command interface for the TinyLlama CPU engine.  Run `make` for help.
SHELL := /bin/bash
.DEFAULT_GOAL := help

BUILD        ?= build
BUILD_TYPE   ?= Release
MODELS       ?= models
THREADS      ?= $(shell scripts/physical_cores.sh)
N            ?= 128
RUNS         ?= 3
TEMP         ?= 0.7
PROMPT       ?= Explain how a CPU cache works, in simple terms.
OLLAMA_MODEL ?= tinyllama:1.1b
OLLAMA_URL   ?= http://localhost:11434
QUALITY      ?= 1
ENGINE_FLAGS ?=
# ENGINE_FLAGS: extra engine flags for bench/compare, e.g. ENGINE_FLAGS="--kv q8"
VENV         ?= .venv
PY           := $(if $(wildcard $(VENV)/bin/python),$(VENV)/bin/python,python3)

BIN         := $(BUILD)/tinyllama
WTYPE       ?= q4
F32_WEIGHTS := $(MODELS)/tinyllama-f32.bin
WEIGHTS     := $(MODELS)/tinyllama-$(WTYPE).bin
TOKENIZER   := $(MODELS)/tokenizer.bin
#CACHE       := baseline
#CACHE       := paged
CACHE       := radix
#CACHE       := streaming
ENGINE_ARGS := --model $(WEIGHTS) --tokenizer $(TOKENIZER) --threads $(THREADS) --cache $(CACHE) $(ENGINE_FLAGS)
BENCH_ARGS  := --bin $(BIN) --model $(WEIGHTS) --tokenizer $(TOKENIZER) --threads $(THREADS) \
               -n $(N) --runs $(RUNS) --ollama-model $(OLLAMA_MODEL) --ollama-url $(OLLAMA_URL) \
               --bandwidth-bin $(BUILD)/bandwidth --ref-model $(F32_WEIGHTS) --engine-args "$(ENGINE_FLAGS)" $(if $(filter 1,$(QUALITY)),--quality)

.PHONY: help setup deps build debug model run chat bench bench-ollama compare bandwidth \
        ollama-serve ollama-pull ollama-run smoke clean distclean venv rapl-perms ppl eval serve webui up down mpi-build mpi-run mpi-bench

help:
	@echo "TinyLlama CPU inference engine"
	@echo ""
	@echo "  make setup         check tools, build, download + convert the model (first time)"
	@echo "  make run           generate one answer      (PROMPT=\"...\" N=128 TEMP=0.7)"
	@echo "  make chat          interactive chat"
	@echo "  make bench         benchmark this engine    (N=128 RUNS=3 THREADS=$(THREADS))"
	@echo "  make compare       benchmark this engine vs Ollama tinyllama:1.1b"
	@echo "  make bench-ollama  benchmark Ollama only"
	@echo "  make bandwidth     measure memory bandwidth (the speed ceiling)"
	@echo "  make ppl           perplexity on bench/data/eval.txt (quality)"
	@echo "  make mpi-run       pipeline across machines with MPI (NP=2 MPI_IFACE=tailscale0 MPI_HOSTS=a:1,b:1)"
	@echo "  make mpi-bench     MPI throughput sweep (streams x micro-batch) -> bench/results/mpi.jsonl"
	@echo "  make up            start Ollama + this engine's API + Open WebUI (make down stops them)"
	@echo "  make serve         OpenAI-compatible API for this engine on :8000"
	@echo "  make webui         Open WebUI (via uv) on :3000, connected to Ollama + this engine"
	@echo "  make eval          WikiText-2 perplexity + KL divergence vs fp32"
	@echo "                     every target takes WTYPE=q4 (default) | q4a | q8 | f32"
	@echo "  make venv          create the Python virtual environment in .venv"
	@echo "  make rapl-perms    allow reading CPU energy counters (sudo, until reboot)"
	@echo "  make smoke         30-second pipeline test with a toy model (no download)"
	@echo "  make debug         debug build with sanitizers (then: make run BUILD=build-debug)"
	@echo ""
	@echo "  make ollama-serve  start Ollama CPU-only with AVX2 (keep running in its own terminal)"
	@echo "  make ollama-pull   download tinyllama:1.1b into Ollama"
	@echo "  make ollama-run    chat with Ollama's tinyllama for a side-by-side feel"

setup: deps venv build model

venv: $(VENV)/.installed

$(VENV)/.installed: requirements.txt
	python3 -m venv $(VENV)
	$(VENV)/bin/python -m pip install -q -r requirements.txt
	@touch $@

rapl-perms:
	sudo chmod a+r /sys/class/powercap/intel-rapl:*/energy_uj

ppl: build model
	@$(BIN) ppl --model $(WEIGHTS) --tokenizer $(TOKENIZER) --threads $(THREADS) \
	    --text bench/data/eval.txt --max-tokens 512 --ctx 256 | cut -c1-200

deps:
	@scripts/check_deps.sh

build:
	@cmake -S . -B $(BUILD) -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) $(CMAKE_ARGS) > /dev/null
	@cmake --build $(BUILD) -j

debug:
	@$(MAKE) --no-print-directory build BUILD=build-debug BUILD_TYPE=Debug CMAKE_ARGS="-DTL_SANITIZE=ON"

model: $(WEIGHTS) $(TOKENIZER)

$(MODELS)/model.safetensors:
	@echo "downloading TinyLlama-1.1B-Chat-v1.0 (about 2.2 GB)"
	@scripts/download.sh $(MODELS)

$(F32_WEIGHTS): $(MODELS)/model.safetensors scripts/convert.py
	@$(PY) scripts/convert.py --model-dir $(MODELS) --out-dir $(MODELS)

$(MODELS)/tinyllama-q4.bin $(MODELS)/tinyllama-q8.bin: $(F32_WEIGHTS) | build
	@$(BIN) quantize --model $(F32_WEIGHTS) --out $@ --wtype $(patsubst tinyllama-%.bin,%,$(notdir $@))

# Q4 with activation-weighted scales, calibrated on the fp32 model (same format and speed as q4)
$(MODELS)/tinyllama-q4a.bin: $(F32_WEIGHTS) | build
	@$(BIN) quantize --model $(F32_WEIGHTS) --tokenizer $(TOKENIZER) --out $@ --wtype q4a \
	    --text bench/data/eval.txt --max-tokens 512

eval: build model
	@$(PY) bench/eval.py --bin $(BIN) --model $(WEIGHTS) --ref-model $(F32_WEIGHTS) \
	    --tokenizer $(TOKENIZER) --threads $(THREADS)

$(TOKENIZER): $(F32_WEIGHTS)

run: build model
	@$(BIN) run $(ENGINE_ARGS) -n $(N) -t $(TEMP) -p "$(PROMPT)"

chat: build model
	@$(BIN) chat $(ENGINE_ARGS) -t $(TEMP) -n 512

bench: build model
	@$(PY) bench/compare.py $(BENCH_ARGS) --only engine

bench-ollama:
	@$(PY) bench/compare.py $(BENCH_ARGS) --only ollama

compare: build model
	@$(PY) bench/compare.py $(BENCH_ARGS) --only both

bench-longer: build model
	@$(PY) bench/compare-longer.py $(BENCH_ARGS) --only engine

bench-ollama-longer:
	@$(PY) bench/compare-longer.py $(BENCH_ARGS) --only ollama

compare-longer: build model
	@$(PY) bench/compare-longer.py $(BENCH_ARGS) --only both

bandwidth: build
	@OMP_NUM_THREADS=$(THREADS) $(BUILD)/bandwidth

# ---- chat UI: Open WebUI talking to Ollama and to this engine (OpenAI-compatible API) ----
API_PORT   ?= 8000
WEBUI_PORT ?= 3000

serve: build model
	@$(PY) tools/openai_server.py --port $(API_PORT) --model $(WEIGHTS) --tokenizer $(TOKENIZER) \
	    --threads $(THREADS) --engine-args "$(ENGINE_FLAGS)"

# Runs via uv (fetches Python 3.11, which Open WebUI needs; no Docker daemon). First run
# installs its dependencies (several GB) into uv's cache. Chats are kept in WEBUI_DATA.
WEBUI_DATA ?= $(HOME)/.open-webui

webui:
	@mkdir -p $(WEBUI_DATA)
	@echo "Open WebUI: http://localhost:$(WEBUI_PORT)  (also run: make serve, make ollama-serve)"
	UV_TORCH_BACKEND=cpu DATA_DIR=$(WEBUI_DATA) WEBUI_AUTH=False \
	    OLLAMA_BASE_URL=http://127.0.0.1:11434 \
	    OPENAI_API_BASE_URL=http://127.0.0.1:$(API_PORT)/v1 OPENAI_API_KEY=none \
	    uvx --python 3.11 open-webui@latest serve --host 127.0.0.1 --port $(WEBUI_PORT)

# One command for the whole chat setup: starts Ollama (CPU-only), this engine's API and
# Open WebUI in the background, skipping any that already answer. Logs go to runs/logs/.
UP_FLAGS ?= --kv q8
up: build
	@mkdir -p runs/logs
	@curl -s -m 2 -o /dev/null localhost:11434/api/tags || { \
	    OLLAMA_LLM_LIBRARY=cpu_avx2 CUDA_VISIBLE_DEVICES="" nohup ollama serve > runs/logs/ollama.log 2>&1 & \
	    echo "started Ollama"; }
	@curl -s -m 2 -o /dev/null localhost:$(API_PORT)/v1/models || { \
	    nohup $(MAKE) --no-print-directory serve WTYPE=q4a ENGINE_FLAGS="$(UP_FLAGS)" > runs/logs/api.log 2>&1 & \
	    echo "started tinyllama API (q4a $(UP_FLAGS))"; }
	@curl -s -m 2 -o /dev/null localhost:$(WEBUI_PORT) || { \
	    nohup $(MAKE) --no-print-directory webui > runs/logs/webui.log 2>&1 & \
	    echo "started Open WebUI (first start takes ~1 min)"; }
	@echo "chat:      http://localhost:$(WEBUI_PORT)"
	@echo "dashboard: http://localhost:$(API_PORT)/dashboard"

down:
	@# patterns are anchored so they cannot match this recipe's own shell command
	-@pkill -f "^(uv tool uvx|/[^ ]*python[^ ]*) .*open-webui" && echo "stopped Open WebUI" || true
	-@pkill -f "^[^ ]*python[^ ]* tools/openai_server.py" && echo "stopped tinyllama API" || true
	-@pkill -x ollama && echo "stopped Ollama" || true

ollama-serve:
	sudo pkill ollama && OLLAMA_LLM_LIBRARY=cpu_avx2 CUDA_VISIBLE_DEVICES="" ollama serve

ollama-pull:
	ollama pull $(OLLAMA_MODEL)

ollama-run:
	ollama run $(OLLAMA_MODEL)

smoke:
	@cmake -S . -B build-tiny -DCMAKE_BUILD_TYPE=Release -DTL_TINY_TEST=ON > /dev/null
	@cmake --build build-tiny -j
	@$(PY) tests/make_tiny_model.py build-tiny/models
	@$(PY) scripts/convert.py --model-dir build-tiny/models --out-dir build-tiny/models
	@build-tiny/tinyllama bench --model build-tiny/models/tinyllama-f32.bin \
	    --tokenizer build-tiny/models/tokenizer.bin -p "hello world" -n 16 --ignore-eos
	@build-tiny/tinyllama ppl --model build-tiny/models/tinyllama-f32.bin \
	    --tokenizer build-tiny/models/tokenizer.bin --text bench/data/eval.txt --max-tokens 64 --ctx 32 | cut -c1-80
	@build-tiny/tinyllama quantize --model build-tiny/models/tinyllama-f32.bin --out build-tiny/models/tinyllama-q4.bin
	@build-tiny/tinyllama ppl --model build-tiny/models/tinyllama-q4.bin --ref-model build-tiny/models/tinyllama-f32.bin \
	    --tokenizer build-tiny/models/tokenizer.bin --text bench/data/eval.txt --max-tokens 64 --ctx 32 | cut -c1-100
	@for m in f32 q4; do \
	    a=$$(build-tiny/tinyllama run --model build-tiny/models/tinyllama-$$m.bin --tokenizer build-tiny/models/tokenizer.bin \
	        -t 0 -n 24 --ignore-eos -p "hello world" 2>/dev/null | md5sum); \
	    b=$$(build-tiny/tinyllama run --model build-tiny/models/tinyllama-$$m.bin --tokenizer build-tiny/models/tokenizer.bin \
	        -t 0 -n 24 --ignore-eos -p "hello world" --prefill seq --pool omp 2>/dev/null | md5sum); \
	    [ -n "$$a" ] && [ "$$a" = "$$b" ] || { echo "batched prefill / pool output differs ($$m)"; exit 1; }; \
	done
	@for p in "hello world" "the cat the cat the cat the cat"; do \
	    a=$$(build-tiny/tinyllama run --model build-tiny/models/tinyllama-q4.bin --tokenizer build-tiny/models/tokenizer.bin \
	        -t 0 -n 40 --ignore-eos -p "$$p" 2>/dev/null | md5sum); \
	    b=$$(build-tiny/tinyllama run --model build-tiny/models/tinyllama-q4.bin --tokenizer build-tiny/models/tokenizer.bin \
	        -t 0 -n 40 --ignore-eos -p "$$p" --spec ngram 2>/dev/null | md5sum); \
	    [ -n "$$a" ] && [ "$$a" = "$$b" ] || { echo "speculative (ngram) output differs: $$p"; exit 1; }; \
	done
	@echo "smoke test passed"

clean:
	rm -rf build build-debug build-tiny

distclean: clean
	rm -rf $(MODELS)

# ---- multi-machine: pipeline parallel over Open MPI (src/pipeline.cpp) ----
# Every machine needs the same repo path, `make mpi-build`, the q4a model and passwordless SSH.
NP        ?= 2
MPI_IFACE ?= tailscale0
MPI_HOSTS ?=
STREAMS   ?= 4
MICRO     ?= 4
MPIRUN    := mpirun -np $(NP) --mca btl tcp,self --mca btl_tcp_if_include $(MPI_IFACE) \
             $(if $(filter-out lo,$(MPI_IFACE)),--mca oob_tcp_if_include $(MPI_IFACE)) \
             $(if $(MPI_HOSTS),--host $(MPI_HOSTS))

mpi-build:
	@cmake -S . -B build-mpi -DCMAKE_BUILD_TYPE=Release -DTL_MPI=ON > /dev/null
	@cmake --build build-mpi -j

mpi-run: mpi-build
	$(MPIRUN) build-mpi/tinyllama pipe --model $(MODELS)/tinyllama-q4a.bin --tokenizer $(TOKENIZER) \
	    -p "$(PROMPT)" -n $(N) --streams $(STREAMS) --micro $(MICRO)

mpi-bench: mpi-build
	bench/mpi_bench.sh $(NP) $(MPI_IFACE) "$(MPI_HOSTS)" $(N)
