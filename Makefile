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
VENV         ?= .venv
PY           := $(if $(wildcard $(VENV)/bin/python),$(VENV)/bin/python,python3)

BIN         := $(BUILD)/tinyllama
WEIGHTS     := $(MODELS)/tinyllama-f32.bin
TOKENIZER   := $(MODELS)/tokenizer.bin
ENGINE_ARGS := --model $(WEIGHTS) --tokenizer $(TOKENIZER) --threads $(THREADS)
BENCH_ARGS  := --bin $(BIN) --model $(WEIGHTS) --tokenizer $(TOKENIZER) --threads $(THREADS) \
               -n $(N) --runs $(RUNS) --ollama-model $(OLLAMA_MODEL) --ollama-url $(OLLAMA_URL) \
               --bandwidth-bin $(BUILD)/bandwidth $(if $(filter 1,$(QUALITY)),--quality)

.PHONY: help setup deps build debug model run chat bench bench-ollama compare bandwidth \
        ollama-serve ollama-pull ollama-run smoke clean distclean venv rapl-perms ppl

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

$(WEIGHTS): $(MODELS)/model.safetensors scripts/convert.py
	@$(PY) scripts/convert.py --model-dir $(MODELS) --out-dir $(MODELS)

$(TOKENIZER): $(WEIGHTS)

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

bandwidth: build
	@OMP_NUM_THREADS=$(THREADS) $(BUILD)/bandwidth

ollama-serve:
	OLLAMA_LLM_LIBRARY=cpu_avx2 CUDA_VISIBLE_DEVICES="" sudo pkill ollama && ollama serve

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
	@echo "smoke test passed"

clean:
	rm -rf build build-debug build-tiny

distclean: clean
	rm -rf $(MODELS)
