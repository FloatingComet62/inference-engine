// tinyllama: a from-scratch CPU inference engine for TinyLlama-1.1B-Chat.
//
//   tinyllama run   -p "prompt"   stream one answer
//   tinyllama chat                interactive multi-turn chat
//   tinyllama bench -p "prompt"   greedy run, prints one JSON line of timings
//   tinyllama tokenize -p "text"  print the chat-template token IDs (debugging)
//   tinyllama ppl --text FILE     perplexity + top-1 predictions (quality check);
//                                 with --ref-model also KL divergence and top-1 agreement
//   tinyllama quantize --out F    write a Q4 (or --wtype q8) copy of an fp32 model
#include "kv_cache.hpp"
#include "model.hpp"
#include "threadpool.hpp"
#include "sampler.hpp"
#include "tokenizer.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <memory>
#include <fstream>
#include <stdexcept>
#include <sstream>
#include <iostream>
#include <memory>
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
    std::string text;     // ppl: evaluation text file
    int ctx = 512;        // ppl: tokens per window
    int max_tokens = 2048;  // ppl: cap on scored tokens
    std::string ref_model;  // ppl: fp32 reference for KL divergence
    std::string out;        // quantize: output path
    std::string wtype = "q4";  // quantize: q4 | q8
    std::string pool = "spin";     // spin (persistent pool) | omp
    std::string prefill = "batch"; // batch | seq (token by token)
    std::string kv = "f32";       // attention working copy: f32|q8
    std::string cache = "radix";  // baseline|paged|radix|streaming
    int window = tl::MAX_SEQ;     // streaming: window tokens (incl. sinks)
    int sinks = 4;                // streaming: attention-sink tokens
};

[[noreturn]] void usage(int code) {
    std::fprintf(stderr,
        "usage: tinyllama <run|chat|bench|tokenize|ppl|quantize> [options]\n"
        "  --model PATH        weights file   (default models/tinyllama-f32.bin)\n"
        "  --tokenizer PATH    tokenizer file (default models/tokenizer.bin)\n"
        "  -p, --prompt TEXT   user message\n"
        "  --system TEXT       system prompt\n"
        "  -n N                max new tokens (default 256)\n"
        "  -t, --temp T        temperature, 0 = greedy (default 0.7; bench forces 0)\n"
        "  --top-p P           nucleus sampling (default 0.9)\n"
        "  --threads N         worker threads (default: OpenMP default)\n"
        "  --seed S            RNG seed\n"
        "  --ignore-eos        always generate exactly N tokens\n"
        "  --text FILE         ppl: text to score\n"
        "  --ctx N             ppl: window length (default 512)\n"
        "  --max-tokens N      ppl: max tokens scored (default 2048)\n"
        "  --ref-model PATH    ppl: reference model for KL divergence / top-1 agreement\n"
        "  --out PATH          quantize: output file\n"
        "  --wtype q4|q8       quantize: matrix format (output head is always q8)\n"
        "  --pool spin|omp     thread runtime (default spin)\n"
        "  --prefill batch|seq prompt processing (default batch)\n"
        "  --kv f32|q8         attention KV working copy precision (default f32)\n"
        "  --cache NAME        kv cache backend: baseline|paged|radix|streaming\n"
        "                      (default radix)\n"
        "  --window N          streaming: window tokens incl. sinks (default 2048)\n"
        "  --sinks N           streaming: attention-sink tokens (default 4)\n");
    std::exit(code);
}

Args parse(int argc, char** argv) {
    Args a;
    if (argc < 2) usage(1);
    a.mode = argv[1];
    if (a.mode == "-h" || a.mode == "--help") usage(0);
    if (a.mode != "run" && a.mode != "chat" && a.mode != "bench" && a.mode != "tokenize" && a.mode != "ppl" && a.mode != "quantize") usage(1);
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
        else if (k == "--text") a.text = val();
        else if (k == "--ctx") a.ctx = std::atoi(val().c_str());
        else if (k == "--max-tokens") a.max_tokens = std::atoi(val().c_str());
        else if (k == "--ref-model") a.ref_model = val();
        else if (k == "--out") a.out = val();
        else if (k == "--wtype") a.wtype = val();
        else if (k == "--pool") a.pool = val();
        else if (k == "--prefill") a.prefill = val();
        else if (k == "--cache") a.cache = val();
        else if (k == "--kv") a.kv = val();
        else if (k == "--window") a.window = std::atoi(val().c_str());
        else if (k == "--sinks") a.sinks = std::atoi(val().c_str());
        else if (k == "-h" || k == "--help") usage(0);
        else { std::fprintf(stderr, "unknown option %s\n", k.c_str()); usage(1); }
    }
    if (a.mode == "bench") a.temp = 0.0f;
    return a;
}

struct Stats {
    int prompt_tokens = 0, gen_tokens = 0;
    double prefill_ms = 0, decode_ms = 0;
    double ttft_ms = 0;   // prompt in -> first token sampled
    double cpu_ms = 0;    // user+sys CPU time over prefill+decode, all threads
};

double cpu_ms_now() {
    struct rusage ru{};
    getrusage(RUSAGE_SELF, &ru);
    return (ru.ru_utime.tv_sec + ru.ru_stime.tv_sec) * 1e3 + (ru.ru_utime.tv_usec + ru.ru_stime.tv_usec) / 1e3;
}

// Feeds `prompt` starting at `pos`, then samples up to max_new tokens.
// Every sampled token is fed back so the KV cache stays complete for chat.
bool g_batch_prefill = true;

Stats generate(tl::Model& model, const tl::Tokenizer& tok, tl::Sampler& sampler,
               const std::vector<int>& prompt, int& pos, int max_new, bool stream, bool ignore_eos) {
    Stats st;
    st.prompt_tokens = (int)prompt.size();
    if (pos + (int)prompt.size() >= tl::MAX_SEQ) {
        std::fprintf(stderr, "\n[context full: %d tokens]\n", tl::MAX_SEQ);
        return st;
    }
    const double cpu0 = cpu_ms_now();
    auto t0 = Clock::now();
    const float* logits = nullptr;
    if (g_batch_prefill) {
        logits = model.prefill(prompt.data(), (int)prompt.size(), pos);
        pos += (int)prompt.size();
    } else {
        for (int t : prompt) logits = model.forward(t, pos++);
    }
    st.prefill_ms = ms_since(t0);

    auto t1 = Clock::now();
    while (st.gen_tokens < max_new && pos < tl::MAX_SEQ) {
        int next = sampler.sample(logits);
        if (st.ttft_ms == 0) st.ttft_ms = ms_since(t0);
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
    st.cpu_ms = cpu_ms_now() - cpu0;
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

int threads_in_use() { return tl::Parallel::get().n; }

}  // namespace

int main(int argc, char** argv) {
    Args a = parse(argc, argv);
    tl::Parallel::get().init(a.threads, a.pool != "omp");
    g_batch_prefill = a.prefill != "seq";
    try {
        if (a.mode == "tokenize") {
            tl::Tokenizer tok(a.tokenizer);
            auto ids = tok.encode_first_turn(a.system, a.prompt);
            for (int id : ids) std::printf("%d ", id);
            std::printf("\n[%zu tokens]\n", ids.size());
            return 0;
        }
        if (a.mode == "quantize") {
            if (a.out.empty() || (a.wtype != "q4" && a.wtype != "q8")) usage(1);
            tl::BaselineKVCache unused;   // the forward pass is not run
            tl::Model src(a.model, unused);
            src.save_quantized(a.out, a.wtype == "q4" ? tl::WType::Q4 : tl::WType::Q8, tl::WType::Q8);
            std::fprintf(stderr, "wrote %s\n", a.out.c_str());
            return 0;
        }
        if (a.kv != "f32" && a.kv != "q8") throw std::runtime_error("unknown --kv " + a.kv + " (f32|q8)");
        auto t_load = Clock::now();
        std::unique_ptr<tl::KVCache> kv_cache;
        if (a.cache == "baseline") kv_cache = std::make_unique<tl::BaselineKVCache>();
        else if (a.cache == "paged") kv_cache = std::make_unique<tl::PagedKVCache>();
        else if (a.cache == "streaming")
            kv_cache = std::make_unique<tl::StreamingKVCache>((size_t)a.window, a.sinks);
        else if (a.cache != "radix")
            throw std::runtime_error("unknown --cache " + a.cache +
                                     " (baseline|paged|radix|streaming)");
        else kv_cache = std::make_unique<tl::RadixKVCache>();  // default: prefix reuse
        tl::Model model(a.model, *kv_cache, a.kv == "q8");
        tl::Tokenizer tok(a.tokenizer);
        const double load_ms = ms_since(t_load);
        tl::Sampler sampler(a.temp, a.top_p, a.seed);
        int pos = 0;

        if (a.mode == "run") {
            auto prompt = tok.encode_first_turn(a.system, a.prompt);
            Stats s = generate(model, tok, sampler, prompt, pos, a.n, true, a.ignore_eos);
            std::fprintf(stderr, "\n\n[prompt %d tok, %.1f tok/s | generated %d tok, %.2f tok/s | %d threads"
                         " | cache %zu/%zu tok, %zu evict]\n",
                         s.prompt_tokens, s.prompt_tokens / (s.prefill_ms / 1000.0),
                         s.gen_tokens, s.gen_tokens / (s.decode_ms / 1000.0), threads_in_use(),
                         kv_cache->used_tokens(), kv_cache->capacity_tokens(),
                         kv_cache->stats().evictions);
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
                std::fprintf(stderr, "\n[%.2f tok/s, context %d/%d, cache %zu/%zu tok, %zu evict]\n",
                             s.gen_tokens / (s.decode_ms / 1000.0), pos, tl::MAX_SEQ,
                             kv_cache->used_tokens(), kv_cache->capacity_tokens(),
                             kv_cache->stats().evictions);
            }
        } else if (a.mode == "ppl") {
            // Scores the text in independent windows of a.ctx tokens (each starts with BOS).
            // Prints perplexity and the argmax prediction at each position, so the
            // benchmark can compare a quantised model against the fp32 reference.
            std::ifstream in(a.text);
            if (!in) throw std::runtime_error("cannot open --text " + a.text);
            std::stringstream ss; ss << in.rdbuf();
            std::string text = ss.str();
            for (char& c : text) if (c == '\n' || c == '\r' || c == '\t') c = ' ';
            auto ids = tok.encode(text, true);
            if ((int)ids.size() > a.max_tokens) ids.resize(a.max_tokens);
            const int ctx = std::max(2, std::min(a.ctx, tl::MAX_SEQ));
            tl::BaselineKVCache ref_cache;
            std::unique_ptr<tl::Model> ref;
            if (!a.ref_model.empty()) ref = std::make_unique<tl::Model>(a.ref_model, ref_cache);
            // log-softmax normaliser and argmax of a logits row
            auto lse = [](const float* l, int& arg) {
                float mx = l[0]; arg = 0;
                for (int v = 1; v < tl::VOCAB; v++) if (l[v] > mx) { mx = l[v]; arg = v; }
                double sum = 0;
                for (int v = 0; v < tl::VOCAB; v++) sum += std::exp((double)l[v] - mx);
                return mx + std::log(sum);
            };
            double nll = 0, kl = 0, ref_nll = 0; int scored = 0, agree = 0;
            std::string top1;
            for (size_t start = 0; start < ids.size(); start += ctx - 1) {
                const float* logits = model.forward(tl::BOS, 0);
                const float* rlog = ref ? ref->forward(tl::BOS, 0) : nullptr;
                for (int p = 0; p < ctx - 1 && start + p < ids.size(); p++) {
                    const int target = ids[start + p];
                    int arg;
                    const double z = lse(logits, arg);
                    nll += z - logits[target];
                    if (ref) {   // KL(P_ref || P) over the full vocabulary
                        int rarg;
                        const double rz = lse(rlog, rarg);
                        ref_nll += rz - rlog[target];
                        agree += rarg == arg;
                        for (int v = 0; v < tl::VOCAB; v++) {
                            const double lr = rlog[v] - rz;
                            kl += std::exp(lr) * (lr - (logits[v] - z));
                        }
                        rlog = ref->forward(target, p + 1);
                    }
                    scored++;
                    top1 += (top1.empty() ? "" : ",") + std::to_string(arg);
                    logits = model.forward(target, p + 1);
                }
            }
            std::printf("{\"tokens\": %d, \"nll\": %.6f, \"ppl\": %.4f", scored, nll / scored,
                        std::exp(nll / scored));
            if (ref)
                std::printf(", \"ref_ppl\": %.4f, \"kl\": %.6f, \"top1_agree\": %.4f",
                            std::exp(ref_nll / scored), kl / scored, (double)agree / scored);
            std::printf(", \"top1\": [%s]}\n", top1.c_str());
        } else {  // bench
            auto prompt = tok.encode_first_turn(a.system, a.prompt);
            Stats s = generate(model, tok, sampler, prompt, pos, a.n, false, a.ignore_eos);
            std::printf("{\"engine\": \"tinyllama-cpp\", \"prompt_tokens\": %d, \"gen_tokens\": %d, "
                        "\"load_ms\": %.1f, \"prefill_ms\": %.1f, \"decode_ms\": %.1f, "
                        "\"prefill_tps\": %.3f, \"decode_tps\": %.3f, \"threads\": %d, "
                        "\"bytes_per_token\": %zu, \"peak_rss_mb\": %.1f, \"ttft_ms\": %.1f, "
                        "\"cpu_ms\": %.1f, \"cpu_util_pct\": %.1f}\n",
                        s.prompt_tokens, s.gen_tokens, load_ms, s.prefill_ms, s.decode_ms,
                        s.prompt_tokens / (s.prefill_ms / 1000.0),
                        s.gen_tokens > 0 ? s.gen_tokens / (s.decode_ms / 1000.0) : 0.0,
                        threads_in_use(), model.bytes_per_token(), peak_rss_mb(), s.ttft_ms,
                        s.cpu_ms, 100.0 * s.cpu_ms / (s.prefill_ms + s.decode_ms));
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
