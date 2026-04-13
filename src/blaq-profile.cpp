//
// blaq-profile.cpp
//
// BLAQ Hardware Profiler — implements Algorithm 1 from the BLAQ paper.
//
// Step 1: Detect cache-line size via OS/hardware query (priority-ordered).
// Step 2: Measure peak single-agent memory bandwidth (STREAM read benchmark).
// Step 3: Count shared-bus processors.
// Step 4: Measure contention bandwidth (all processors reading simultaneously).
// Step 5: Compute contention ratio sigma = 1 - BW_obs / BW_peak.
// Step 6: Derive aligned group sizes for b=4.
// Step 7: Compute penalty weights lambda_bw and lambda_mem.
//

#include "blaq-profile.h"

#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <chrono>
#include <thread>
#include <vector>
#include <algorithm>
#include <fstream>
#include <string>
#include <sstream>

#if defined(__linux__)
#  include <unistd.h>    // sysconf, access
#endif
#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#endif
#if defined(__APPLE__)
#  include <sys/sysctl.h>
#endif

#include <nlohmann/json.hpp>

using json = nlohmann::json;

// ---------------------------------------------------------------------------
// Step 1: Cache-line detection (priority-ordered)
// ---------------------------------------------------------------------------

static uint32_t detect_cache_line_bytes() {
#if defined(__linux__)
    long val = sysconf(_SC_LEVEL1_DCACHE_LINESIZE);
    if (val > 0) return (uint32_t)val;
#endif
#if defined(__APPLE__)
    size_t linesize = 0;
    size_t sz = sizeof(linesize);
    if (sysctlbyname("hw.cachelinesize", &linesize, &sz, nullptr, 0) == 0
            && linesize > 0) {
        return (uint32_t)linesize;
    }
#endif
#if defined(_WIN32)
    SYSTEM_LOGICAL_PROCESSOR_INFORMATION info[64];
    DWORD len = sizeof(info);
    if (GetLogicalProcessorInformation(info, &len)) {
        for (DWORD i = 0; i < len / sizeof(*info); ++i) {
            if (info[i].Relationship == RelationCache
                    && info[i].Cache.Level == 1
                    && info[i].Cache.Type  == CacheData) {
                return (uint32_t)info[i].Cache.LineSize;
            }
        }
    }
#endif
    // Compile-time hints for well-known microarchitectures
#if defined(__aarch64__)
    return 64;   // ARM Neoverse V3, Apple Avalon (reports 128 via sysctl on Apple)
#endif
    return 64;   // safe universal default
}

// ---------------------------------------------------------------------------
// Step 2 & 4: STREAM-style sequential read benchmark
// ---------------------------------------------------------------------------
//
// Allocates a buffer larger than any LLC (256 MB), then scans it from
// n_threads threads to saturate the memory bus.
// Returns per-thread-average bytes/second.

static double stream_benchmark(int n_threads, double duration_s) {
    const size_t BUF_BYTES = 256ULL * 1024 * 1024;  // 256 MB

    // Allocate and touch all pages to avoid soft-fault overhead
    std::vector<char> buf(BUF_BYTES, (char)0x42);

    std::vector<std::thread> threads;
    std::vector<double>      results(n_threads, 0.0);

    auto worker = [&](int tid) {
        // Cast to volatile to prevent the compiler from optimising the reads away
        const volatile char * p   = buf.data();
        const volatile char * end = p + BUF_BYTES;
        double acc   = 0.0;
        uint64_t bytes = 0;

        auto t0 = std::chrono::steady_clock::now();
        while (true) {
            for (const volatile char * q = p; q < end; q += 64) {
                acc += *q;  // one read per cache line — not eliminated by compiler
            }
            bytes += BUF_BYTES;

            double elapsed = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0).count();
            if (elapsed >= duration_s) {
                results[tid] = (double)bytes / elapsed;
                break;
            }
        }
        (void)acc;
    };

    for (int i = 0; i < n_threads; ++i) {
        threads.emplace_back(worker, i);
    }
    for (auto & t : threads) {
        t.join();
    }

    double total = 0.0;
    for (double r : results) total += r;
    return total / n_threads;
}

// ---------------------------------------------------------------------------
// Step 3: Shared-bus processor count
// ---------------------------------------------------------------------------

// Returns whether a discrete or integrated GPU is visible on this system.
// Used to decide whether the CPU-only contention benchmark is meaningful.
// Public C wrapper: blaq_gpu_on_shared_bus()
static bool gpu_present_on_shared_bus() {
#if defined(__APPLE__)
    // Apple Silicon always has an integrated GPU on the same UMA bus.
    return true;
#elif defined(__linux__)
    // Check for a CUDA-capable GPU via /proc/driver/nvidia/version (world-readable
    // when the NVIDIA kernel module is loaded) or DRM render nodes (existence only —
    // we don't open them, just test presence with access(F_OK)).
    if (access("/proc/driver/nvidia/version", F_OK) == 0) {
        return true;  // NVIDIA driver loaded → GPU present
    }
    // Fallback: check for any DRM render node (covers AMD, Intel, Nouveau, etc.)
    for (int i = 0; i < 8; ++i) {
        char path[64];
        snprintf(path, sizeof(path), "/dev/dri/renderD%d", 128 + i);
        if (access(path, F_OK) == 0) {
            return true;
        }
    }
    return false;
#else
    return false;
#endif
}

static uint32_t count_shared_bus_processors() {
    // On a true UMA SoC (Apple M-series, Grace-Blackwell NVLink-C2C) the
    // relevant agents are the CPU and the GPU sharing the same memory bus.
    // We can only measure CPU-side contention here; GPU-side contention
    // requires a concurrently running GPU kernel (CUDA/Metal), which is
    // beyond the scope of this CPU-only profiler.
    //
    // Return 2 only on Apple (where CPU-only multi-thread does reflect the
    // shared LPDDR bus), 1 elsewhere so the benchmark is repeatable.
    // The caller should check gpu_present_on_shared_bus() and warn.
#if defined(__APPLE__)
    return 2;
#else
    return 1;  // CPU-only; GPU contention is not measurable here
#endif
}

// ---------------------------------------------------------------------------
// Public C wrapper (declared in blaq-profile.h)
// ---------------------------------------------------------------------------

bool blaq_gpu_on_shared_bus(void) {
    return gpu_present_on_shared_bus();
}

// ---------------------------------------------------------------------------
// Main profiling function (Algorithm 1)
// ---------------------------------------------------------------------------

bool blaq_profile_measure(blaq_profile_t * out, float gamma, float beta,
                          float sigma_override) {
    if (!out) return false;

    // Step 1
    out->cache_line_bytes = detect_cache_line_bytes();

    // Step 2: peak single-agent bandwidth (0.5 s warm-up)
    out->peak_bw_bytes_per_sec = stream_benchmark(1, 0.5);

    // Step 3
    out->shared_bus_procs = count_shared_bus_processors();

    // Step 4: contention bandwidth (all procs reading simultaneously)
    //
    // On Apple Silicon, shared_bus_procs=2 stresses the LPDDR bus as both
    // CPU and GPU would, giving a meaningful sigma.
    //
    // On Linux/NVIDIA, shared_bus_procs=1 so obs_bw ≈ peak_bw (same thread
    // count).  True GPU-HBM contention cannot be measured from CPU code.
    // If the caller passes sigma_override >= 0, skip the benchmark and derive
    // obs_bw synthetically so that the printed table stays consistent.
    if (sigma_override >= 0.f) {
        // Synthetic: obs_bw that would produce exactly sigma_override
        out->obs_bw_bytes_per_sec =
            out->peak_bw_bytes_per_sec * (1.0 - (double)sigma_override);
    } else {
        out->obs_bw_bytes_per_sec = stream_benchmark(
            (int)out->shared_bus_procs, 0.5);
    }

    // Step 5: compute or override sigma
    if (sigma_override >= 0.f) {
        // Caller supplied an explicit value (from published specs or
        // a prior CUDA-assisted measurement).
        out->contention_ratio = sigma_override;
    } else if (out->peak_bw_bytes_per_sec > 0.0) {
        out->contention_ratio =
            (float)(1.0 - out->obs_bw_bytes_per_sec / out->peak_bw_bytes_per_sec);
    } else {
        out->contention_ratio = 0.f;
    }
    // Clamp to [0, 1]
    if (out->contention_ratio < 0.f) out->contention_ratio = 0.f;
    if (out->contention_ratio > 1.f) out->contention_ratio = 1.f;

    // Warn when GPU is present but we could not measure contention
    if (gpu_present_on_shared_bus() && out->shared_bus_procs == 1
            && sigma_override < 0.f) {
        fprintf(stderr,
            "WARNING: GPU detected but contention ratio (sigma) was measured\n"
            "         using CPU threads only.  On discrete-GPU or NVLink-C2C\n"
            "         systems (e.g. Grace-Blackwell GB10) the GPU-HBM bus is\n"
            "         independent of CPU LPDDR, so sigma=%.3f may be inaccurate.\n"
            "         Use --sigma <value> to supply a known contention ratio.\n"
            "         Typical values: 0.10-0.25 for shared-BW systems.\n\n",
            (double)out->contention_ratio);
    }

    // Step 6: aligned group sizes for b=4
    // s*(b, L) = 8*L/b  =>  for b=4: 2*L
    out->aligned_group_128 = 2 * out->cache_line_bytes;      // e.g. 128 for L=64
    out->aligned_group_256 = 4 * out->cache_line_bytes;      // e.g. 256 for L=64

    // Step 7: penalty weights
    out->lambda_bw  = (out->peak_bw_bytes_per_sec > 0.0)
        ? (float)(gamma / out->peak_bw_bytes_per_sec)
        : 0.f;
    out->lambda_mem = beta * out->contention_ratio;

    return true;
}

// ---------------------------------------------------------------------------
// Defaults (no profiling data)
// ---------------------------------------------------------------------------

void blaq_profile_defaults(blaq_profile_t * out) {
    if (!out) return;
    memset(out, 0, sizeof(*out));
    out->cache_line_bytes      = 64;
    out->peak_bw_bytes_per_sec = 100.0 * 1e9;  // 100 GB/s nominal
    out->obs_bw_bytes_per_sec  = 80.0  * 1e9;
    out->contention_ratio      = 0.20f;
    out->shared_bus_procs      = 2;
    out->aligned_group_128     = 128;           // 2 * 64
    out->aligned_group_256     = 256;           // 4 * 64
    out->lambda_bw             = (float)(1.0 / out->peak_bw_bytes_per_sec);
    out->lambda_mem            = 1.0f * out->contention_ratio;
}

// ---------------------------------------------------------------------------
// JSON serialisation (using nlohmann/json already in the repo)
// ---------------------------------------------------------------------------

bool blaq_profile_save_json(const blaq_profile_t * p, const char * path) {
    if (!p || !path) return false;

    json j;
    j["cache_line_bytes"]       = p->cache_line_bytes;
    j["peak_bw_bytes_per_sec"]  = p->peak_bw_bytes_per_sec;
    j["obs_bw_bytes_per_sec"]   = p->obs_bw_bytes_per_sec;
    j["contention_ratio"]       = p->contention_ratio;
    j["shared_bus_procs"]       = p->shared_bus_procs;
    j["aligned_group_128"]      = p->aligned_group_128;
    j["aligned_group_256"]      = p->aligned_group_256;
    j["lambda_bw"]              = p->lambda_bw;
    j["lambda_mem"]             = p->lambda_mem;

    std::ofstream ofs(path);
    if (!ofs) return false;
    ofs << j.dump(4) << "\n";
    return true;
}

bool blaq_profile_load_json(blaq_profile_t * out, const char * path) {
    if (!out || !path) return false;

    std::ifstream ifs(path);
    if (!ifs) return false;

    json j;
    try {
        ifs >> j;
    } catch (...) {
        return false;
    }

    blaq_profile_defaults(out);  // seed with safe values first

    auto get_u32 = [&](const char * key, uint32_t & dst) {
        if (j.contains(key) && j[key].is_number()) dst = j[key].get<uint32_t>();
    };
    auto get_dbl = [&](const char * key, double & dst) {
        if (j.contains(key) && j[key].is_number()) dst = j[key].get<double>();
    };
    auto get_flt = [&](const char * key, float & dst) {
        if (j.contains(key) && j[key].is_number()) dst = j[key].get<float>();
    };

    get_u32("cache_line_bytes",       out->cache_line_bytes);
    get_dbl("peak_bw_bytes_per_sec",  out->peak_bw_bytes_per_sec);
    get_dbl("obs_bw_bytes_per_sec",   out->obs_bw_bytes_per_sec);
    get_flt("contention_ratio",       out->contention_ratio);
    get_u32("shared_bus_procs",       out->shared_bus_procs);
    get_u32("aligned_group_128",      out->aligned_group_128);
    get_u32("aligned_group_256",      out->aligned_group_256);
    get_flt("lambda_bw",              out->lambda_bw);
    get_flt("lambda_mem",             out->lambda_mem);

    return true;
}

// ---------------------------------------------------------------------------
// Human-readable print
// ---------------------------------------------------------------------------

void blaq_profile_print(const blaq_profile_t * p) {
    if (!p) return;
    fprintf(stderr, "\n=== BLAQ Hardware Profile ===\n");
    fprintf(stderr, "  Cache-line size       : %u bytes\n",  p->cache_line_bytes);
    fprintf(stderr, "  Peak BW               : %.1f GB/s\n",
            p->peak_bw_bytes_per_sec / 1e9);
    fprintf(stderr, "  Contention BW         : %.1f GB/s\n",
            p->obs_bw_bytes_per_sec  / 1e9);

    const bool gpu_present = gpu_present_on_shared_bus();
    const bool sigma_zero  = (p->contention_ratio < 0.005f);
    fprintf(stderr, "  Contention ratio (σ)  : %.3f%s\n",
            (double)p->contention_ratio,
            (gpu_present && sigma_zero) ? "  [GPU present — CPU-only measurement, see WARNING above]" : "");

    fprintf(stderr, "  Shared-bus procs      : %u%s\n",
            p->shared_bus_procs,
            gpu_present ? "  (CPU only; GPU not included)" : "");
    fprintf(stderr, "  Aligned group (4-bit) : %u weights  (target: %u-byte cache line)\n",
            p->aligned_group_128, p->cache_line_bytes);
    fprintf(stderr, "  Double-line group     : %u weights\n", p->aligned_group_256);
    fprintf(stderr, "  lambda_bw             : %.3e\n",     (double)p->lambda_bw);
    fprintf(stderr, "  lambda_mem            : %.3f\n",     (double)p->lambda_mem);
    fprintf(stderr, "==============================\n\n");
}
