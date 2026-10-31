// Measures sustained memory read bandwidth: the hard ceiling for decode speed.
// tokens/s ceiling = read_gbps * 1e9 / bytes_per_token
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

int main(int argc, char** argv) {
    const size_t mb = argc > 1 ? std::strtoul(argv[1], nullptr, 10) : 1024;
    const long long n = (long long)(mb * 1024 * 1024 / sizeof(uint64_t));
    std::vector<uint64_t> a(n);

#pragma omp parallel for schedule(static)
    for (long long i = 0; i < n; i++) a[i] = (uint64_t)i;   // first touch per thread

    double best = 1e30;
    uint64_t sink = 0;
    for (int rep = 0; rep < 7; rep++) {
        uint64_t s = 0;
        auto t0 = std::chrono::steady_clock::now();
#pragma omp parallel for reduction(+ : s) schedule(static)
        for (long long i = 0; i < n; i++) s += a[i];
        double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        best = std::min(best, dt);
        sink += s;
    }
    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif
    std::printf("{\"read_gbps\": %.2f, \"buffer_mb\": %zu, \"threads\": %d, \"checksum\": %llu}\n",
                (double)n * sizeof(uint64_t) / best / 1e9, mb, threads, (unsigned long long)(sink & 0xffff));
    return 0;
}
