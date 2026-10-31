// tinyllama: a from-scratch CPU inference engine for TinyLlama-1.1B-Chat.
//
//   tinyllama run   -p "prompt"   stream one answer
//   tinyllama chat                interactive multi-turn chat
//   tinyllama bench -p "prompt"   greedy run, prints one JSON line of timings
//   tinyllama tokenize -p "text"  print the chat-template token IDs (debugging)
#include "model.hpp"
#include "sampler.hpp"
#include "tokenizer.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>
#include <sys/resource.h>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

struct Args {
    std::string mode = "run";
    std::string model = "models/tinyllama-f32.bin";
    std::string tokenizer = "models/tokenizer.bin";
    std::string prompt = "Hello! Who are you?";
    std::string system = "You are a helpful AI assistant.";
    int n = 256;
    float temp = 0.7f;
    float top_p = 0.9f;
    int threads = 0;
    unsigned long long seed = 42;
    bool ignore_eos = false;
};

[[noreturn]] void usage(int code) {
    std::fprintf(stderr,
        "usage: tinyllama <run|chat|bench|tokenize> [options]\n"
        "  --model PATH        weights file   (default models/tinyllama-f32.bin)\n"
        "  --tokenizer PATH    tokenizer file (default models/tokenizer.bin)\n"
        "  -p, --prompt TEXT   user message\n"
        "  --system TEXT       system prompt\n"
        "  -n N                max new tokens (default 256)\n"
        "  -t, --temp T        temperature, 0 = greedy (default 0.7; bench forces 0)\n"
        "  --top-p P           nucleus sampling (default 0.9)\n"
        "  --threads N         worker threads (default: OpenMP default)\n"
        "  --seed S            RNG seed\n"
        "  --ignore-eos        always generate exactly N tokens\n");
    std::exit(code);
}

Args parse(int argc, char** argv) {
    Args a;
    if (argc < 2) usage(1);
    a.mode = argv[1];
    if (a.mode == "-h" || a.mode == "--help") usage(0);
    if (a.mode != "run" && a.mode != "chat" && a.mode != "bench" && a.mode != "tokenize") usage(1);
    for (int i = 2; i < argc; i++) {
        std::string k = argv[i];
        auto val = [&]() -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", k.c_str()); usage(1); }
            return argv[++i];
        };
        if (k == "--model") a.model = val();
        else if (k == "--tokenizer") a.tokenizer = val();
        else if (k == "-p" || k == "--prompt") a.prompt = val();
        else if (k == "--system") a.system = val();
        else if (k == "-n") a.n = std::atoi(val().c_str());
        else if (k == "-t" || k == "--temp") a.temp = std::strtof(val().c_str(), nullptr);
        else if (k == "--top-p") a.top_p = std::strtof(val().c_str(), nullptr);
        else if (k == "--threads") a.threads = std::atoi(val().c_str());
        else if (k == "--seed") a.seed = std::strtoull(val().c_str(), nullptr, 10);
        else if (k == "--ignore-eos") a.ignore_eos = true;
        else if (k == "-h" || k == "--help") usage(0);
        else { std::fprintf(stderr, "unknown option %s\n", k.c_str()); usage(1); }
    }
    if (a.mode == "bench") a.temp = 0.0f;
    return a;
}

struct Stats {
    int prompt_tokens = 0, gen_tokens = 0;
    double prefill_ms = 0, decode_ms = 0;
};

// Feeds `prompt` starting at `pos`, then samples up to max_new tokens.
// Every sampled token is fed back so the KV cache stays complete for chat.
Stats generate(tl::Model& model, const tl::Tokenizer& tok, tl::Sampler& sampler,
               const std::vector<int>& prompt, int& pos, int max_new, bool stream, bool ignore_eos) {
    Stats st;
    st.prompt_tokens = (int)prompt.size();
    if (pos + (int)prompt.size() >= tl::MAX_SEQ) {
        std::fprintf(stderr, "\n[context full: %d tokens]\n", tl::MAX_SEQ);
        return st;
    }
    auto t0 = Clock::now();
    const float* logits = nullptr;
    for (int t : prompt) logits = model.forward(t, pos++);
    st.prefill_ms = ms_since(t0);

    auto t1 = Clock::now();
    while (st.gen_tokens < max_new && pos < tl::MAX_SEQ) {
        int next = sampler.sample(logits);
        if (next == tl::EOS && !ignore_eos) break;
        st.gen_tokens++;
        if (stream) {
            std::string piece = tok.decode(next);
            if (st.gen_tokens == 1 && !piece.empty() && piece[0] == ' ') piece.erase(0, 1);
            std::fwrite(piece.data(), 1, piece.size(), stdout);
            std::fflush(stdout);
        }
        logits = model.forward(next, pos++);
    }
    st.decode_ms = ms_since(t1);
    return st;
}

double peak_rss_mb() {
    struct rusage ru{};
    getrusage(RUSAGE_SELF, &ru);
#ifdef __APPLE__
    return ru.ru_maxrss / (1024.0 * 1024.0);   // bytes on macOS
#else
    return ru.ru_maxrss / 1024.0;              // kilobytes on Linux
#endif
}

int threads_in_use() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

}  // namespace

int main(int argc, char** argv) {
    Args a = parse(argc, argv);
#ifdef _OPENMP
    if (a.threads > 0) omp_set_num_threads(a.threads);
#endif
    try {
        if (a.mode == "tokenize") {
            tl::Tokenizer tok(a.tokenizer);
            auto ids = tok.encode_first_turn(a.system, a.prompt);
            for (int id : ids) std::printf("%d ", id);
            std::printf("\n[%zu tokens]\n", ids.size());
            return 0;
        }
        auto t_load = Clock::now();
        tl::Model model(a.model);
        tl::Tokenizer tok(a.tokenizer);
        const double load_ms = ms_since(t_load);
        tl::Sampler sampler(a.temp, a.top_p, a.seed);
        int pos = 0;

        if (a.mode == "run") {
            auto prompt = tok.encode_first_turn(a.system, a.prompt);
            Stats s = generate(model, tok, sampler, prompt, pos, a.n, true, a.ignore_eos);
            std::fprintf(stderr, "\n\n[prompt %d tok, %.1f tok/s | generated %d tok, %.2f tok/s | %d threads]\n",
                         s.prompt_tokens, s.prompt_tokens / (s.prefill_ms / 1000.0),
                         s.gen_tokens, s.gen_tokens / (s.decode_ms / 1000.0), threads_in_use());
        } else if (a.mode == "chat") {
            std::fprintf(stderr, "TinyLlama chat. Type /exit to quit.\n");
            bool first = true;
            std::string line;
            while (true) {
                std::fprintf(stderr, "\n> ");
                if (!std::getline(std::cin, line) || line == "/exit") break;
                if (line.empty()) continue;
                auto prompt = first ? tok.encode_first_turn(a.system, line) : tok.encode_next_turn(line);
                first = false;
                Stats s = generate(model, tok, sampler, prompt, pos, a.n, true, false);
                std::fprintf(stderr, "\n[%.2f tok/s, context %d/%d]\n",
                             s.gen_tokens / (s.decode_ms / 1000.0), pos, tl::MAX_SEQ);
            }
        } else {  // bench
            auto prompt = tok.encode_first_turn(a.system, a.prompt);
            Stats s = generate(model, tok, sampler, prompt, pos, a.n, false, a.ignore_eos);
            std::printf("{\"engine\": \"tinyllama-cpp\", \"prompt_tokens\": %d, \"gen_tokens\": %d, "
                        "\"load_ms\": %.1f, \"prefill_ms\": %.1f, \"decode_ms\": %.1f, "
                        "\"prefill_tps\": %.3f, \"decode_tps\": %.3f, \"threads\": %d, "
                        "\"bytes_per_token\": %zu, \"peak_rss_mb\": %.1f}\n",
                        s.prompt_tokens, s.gen_tokens, load_ms, s.prefill_ms, s.decode_ms,
                        s.prompt_tokens / (s.prefill_ms / 1000.0),
                        s.gen_tokens > 0 ? s.gen_tokens / (s.decode_ms / 1000.0) : 0.0,
                        threads_in_use(), model.bytes_per_token(), peak_rss_mb());
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
