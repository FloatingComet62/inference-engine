// Multi-machine pipeline-parallel inference over Open MPI (see pipeline.cpp).
#pragma once
#include <string>

namespace tl {

struct PipeArgs {
    std::string model, tokenizer, prompt, system, split;
    int n = 64;          // tokens generated per stream
    int streams = 1;     // independent requests in flight
    int micro = 1;       // requests per micro-batch (rows sharing one weight read)
    int ctx = 512;       // max tokens per request (K/V buffer size)
    int threads = 0;     // per rank; 0 = this machine's fast cores / ranks on it
};

int run_pipeline(int argc, char** argv, const PipeArgs& a);

}  // namespace tl
