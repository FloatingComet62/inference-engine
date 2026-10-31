// Persistent worker pool. One dispatch = one task run by every thread, then one barrier.
// Workers spin briefly for the next task (cheap at 100+ dispatches per token), then sleep
// on a condition variable so an idle chat does not burn CPU.
// Threads are pinned to the fastest cores (P-cores on hybrid Intel), one per physical core
// first, then their hyperthread siblings: unpinned, Linux mixes in E-cores and every barrier
// waits for the slowest thread (measured: 13.5 vs 26.8 tok/s at 8 threads on an i5-13420H).
// ponytail: fastest-cores heuristic; the auto-tuner (Priority 2) will choose the core set.
#pragma once
#include <algorithm>
#include <atomic>
#include <fstream>
#include <string>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
#include <type_traits>
#include <vector>
#ifdef __AVX2__
#include <immintrin.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#endif

namespace tl {

constexpr int SPIN_PAUSE = 4000;    // busy-wait iterations before yielding the core
constexpr int SPIN_YIELD = 20000;   // yields before sleeping on the condition variable

// Parses a kernel CPU list such as "0-7,16-23".
inline std::vector<int> parse_cpu_list(const std::string& s) {
    std::vector<int> out;
    size_t i = 0;
    while (i < s.size()) {
        size_t j = s.find(',', i);
        if (j == std::string::npos) j = s.size();
        const std::string part = s.substr(i, j - i);
        const size_t dash = part.find('-');
        try {
            const int a = std::stoi(part), b = dash == std::string::npos ? a : std::stoi(part.substr(dash + 1));
            for (int c = a; c <= b; c++) out.push_back(c);
        } catch (...) {}
        i = j + 1;
    }
    return out;
}

// CPUs to pin to: the P-cores on hybrid Intel (kernel's /sys/devices/cpu_core list),
// otherwise every online CPU; physical cores first, then their hyperthread siblings.
// (Not "highest max frequency": favored-core turbo makes 1-2 cores report a higher max.)
inline std::vector<int> fast_cpus() {
    std::vector<int> out;
#ifdef __linux__
    std::string list;
    std::ifstream atom("/sys/devices/cpu_atom/cpus"), core("/sys/devices/cpu_core/cpus");
    if (!(atom && core && std::getline(core, list))) {
        std::ifstream online("/sys/devices/system/cpu/online");
        std::getline(online, list);
    }
    const std::vector<int> cpus = parse_cpu_list(list);
    for (int pass = 0; pass < 2; pass++)
        for (int c : cpus) {
            std::ifstream sib("/sys/devices/system/cpu/cpu" + std::to_string(c) + "/topology/thread_siblings_list");
            int first = c;
            sib >> first;   // list starts with the lowest sibling, e.g. "0-1" or "0,1"
            if ((first == c) == (pass == 0)) out.push_back(c);
        }
#endif
    return out;
}

inline void pin_to(int cpu) {
#ifdef __linux__
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
#else
    (void)cpu;
#endif
}

class ThreadPool {
public:
    explicit ThreadPool(int n) : n_(n < 1 ? 1 : n) {
        const std::vector<int> cpus = fast_cpus();
        const bool pin = (int)cpus.size() >= n_;   // more threads than fast cores: let the OS place them
        if (pin) pin_to(cpus[0]);
        for (int t = 1; t < n_; t++)
            workers_.emplace_back([this, t, pin, cpus] { if (pin) pin_to(cpus[t]); loop(t); });
    }
    ~ThreadPool() {
        stop_ = true;
        gen_.fetch_add(1);
        { std::lock_guard<std::mutex> lk(m_); cv_.notify_all(); }
        for (auto& w : workers_) w.join();
    }
    int size() const { return n_; }

    // Runs f(tid, nthreads) on every thread (the caller is thread 0) and waits for all.
    template <class F> void run(F&& f) {
        auto call = [](void* ctx, int tid, int nth) { (*static_cast<std::remove_reference_t<F>*>(ctx))(tid, nth); };
        dispatch(+call, &f);
    }

private:
    void dispatch(void (*fn)(void*, int, int), void* ctx) {
        fn_ = fn; ctx_ = ctx;
        done_.store(0);
        gen_.fetch_add(1);
        if (sleepers_.load() > 0) { std::lock_guard<std::mutex> lk(m_); cv_.notify_all(); }
        if (fn) fn(ctx, 0, n_);
        for (int spin = 0; done_.load() < n_ - 1; spin++) {
            if (spin < SPIN_PAUSE) pause(); else std::this_thread::yield();
        }
    }
    void loop(int tid) {
        unsigned seen = 0;   // initial gen_, not a load: the first task may be dispatched before this thread starts
        while (true) {
            for (int spin = 0; gen_.load() == seen; spin++) {
                if (spin < SPIN_PAUSE) { pause(); continue; }
                if (spin < SPIN_PAUSE + SPIN_YIELD) { std::this_thread::yield(); continue; }
                std::unique_lock<std::mutex> lk(m_);
                sleepers_++;
                cv_.wait(lk, [&] { return gen_.load() != seen; });
                sleepers_--;
            }
            seen = gen_.load();
            if (stop_) return;
            fn_(ctx_, tid, n_);
            done_.fetch_add(1);
        }
    }
    static void pause() {
#ifdef __AVX2__
        _mm_pause();
#endif
    }

    int n_;
    std::vector<std::thread> workers_;
    // separate cache lines: workers poll gen_ while finishing threads write done_
    void (*fn_)(void*, int, int) = nullptr;
    void* ctx_ = nullptr;
    alignas(64) std::atomic<unsigned> gen_{0};
    alignas(64) std::atomic<int> done_{0};
    alignas(64) std::atomic<int> sleepers_{0};
    std::atomic<bool> stop_{false};
    std::mutex m_;
    std::condition_variable cv_;
};

// Global parallel runtime: the spin pool, or OpenMP (--pool omp) for comparison.
struct Parallel {
    static Parallel& get() { static Parallel p; return p; }
    void init(int threads, bool use_pool) {
        n = threads > 0 ? threads : (int)std::thread::hardware_concurrency();
        pool_on = use_pool;
        pool.reset(use_pool ? new ThreadPool(n) : nullptr);
#ifdef _OPENMP
        omp_set_num_threads(n);
#endif
    }
    int n = 1;
    bool pool_on = false;
    std::unique_ptr<ThreadPool> pool;
};

// Calls f(begin, end) over [0, count) in small chunks that threads take from a shared
// counter (ggml's scheme): faster cores simply take more chunks, so a slow core (an
// E-core, a busy hyperthread, an unpinned thread) cannot hold up the whole barrier.
// Each index is still processed by exactly one call, so results are unchanged.
constexpr int CHUNKS_PER_THREAD = 4;
template <class F> void parallel_for(int count, F&& f) {
    Parallel& P = Parallel::get();
    std::atomic<int> next{0};
    auto chunk = [&](int, int nth) {
        int cs = count / (nth * CHUNKS_PER_THREAD);
        if (cs < 1) cs = 1;
        for (int b; (b = next.fetch_add(cs, std::memory_order_relaxed)) < count;)
            f(b, b + cs < count ? b + cs : count);
    };
    if (P.pool) return P.pool->run(chunk);
#ifdef _OPENMP
#pragma omp parallel
    chunk(omp_get_thread_num(), omp_get_num_threads());
#else
    chunk(0, 1);
#endif
}

}  // namespace tl
