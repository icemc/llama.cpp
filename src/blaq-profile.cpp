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
// Measurement methodology (variance reduction):
//   - 1 warmup pass (discarded) before each BW direction
//   - N timed passes per BW direction (default N=5, 1 s each)
//   - Median across passes used for peak_bw and obs_bw
//   - CV (stddev/mean) recorded in the profile for transparency
//

#include "blaq-profile.h"

#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <cctype>
#include <chrono>
#include <thread>
#include <vector>
#include <numeric>
#include <algorithm>
#include <fstream>
#include <string>
#include <sstream>

#if defined(__linux__)
#  include <unistd.h>    // sysconf, access
#  include <dlfcn.h>     // dlopen/dlsym for NVML dynamic loading
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
// Portable string helpers
// ---------------------------------------------------------------------------

#if defined(_WIN32)
#  define BLAQ_STRNCASECMP _strnicmp
#else
#  define BLAQ_STRNCASECMP strncasecmp
#endif

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
#if defined(__aarch64__)
    return 64;   // conservative ARM default; Apple Silicon reports 128 via sysctl above
#endif
    return 64;
}

// ---------------------------------------------------------------------------
// Step 2 & 4: STREAM-style sequential read benchmark — single timed pass
// ---------------------------------------------------------------------------
//
// Allocates a buffer larger than any LLC (256 MB), then scans it from
// n_threads threads to saturate the memory bus.
// Returns per-thread-average bytes/second for one timed window of duration_s.

static double stream_benchmark_once(int n_threads, double duration_s) {
    const size_t BUF_BYTES = 256ULL * 1024 * 1024;  // 256 MB

    std::vector<char> buf(BUF_BYTES, (char)0x42);

    std::vector<std::thread> threads;
    std::vector<double>      results(n_threads, 0.0);

    auto worker = [&](int tid) {
        const volatile char * p   = buf.data();
        const volatile char * end = p + BUF_BYTES;
        double acc   = 0.0;
        uint64_t bytes = 0;

        auto t0 = std::chrono::steady_clock::now();
        while (true) {
            for (const volatile char * q = p; q < end; q += 64) {
                acc += *q;
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

    for (int i = 0; i < n_threads; ++i) threads.emplace_back(worker, i);
    for (auto & t : threads)           t.join();

    double total = 0.0;
    for (double r : results) total += r;
    return total / n_threads;
}

// ---------------------------------------------------------------------------
// Multi-pass benchmark: warmup + N measured passes → median, mean, stddev, CV
// ---------------------------------------------------------------------------

struct bw_stats_t {
    double median;
    double mean;
    double stddev;
    float  cv;
};

static bw_stats_t stream_benchmark_stats(int n_threads, double warmup_s,
                                         double measure_s, int n_runs) {
    // Warmup pass — discarded; warms up OS page-faults and memory controller
    if (warmup_s > 0.0) {
        stream_benchmark_once(n_threads, warmup_s);
    }

    // Measured passes
    std::vector<double> samples;
    samples.reserve(n_runs);
    for (int i = 0; i < n_runs; ++i) {
        samples.push_back(stream_benchmark_once(n_threads, measure_s));
    }

    // Sort for median
    std::vector<double> sorted = samples;
    std::sort(sorted.begin(), sorted.end());

    double median = (n_runs % 2 == 0)
        ? (sorted[n_runs/2 - 1] + sorted[n_runs/2]) / 2.0
        : sorted[n_runs/2];

    double sum = std::accumulate(samples.begin(), samples.end(), 0.0);
    double mean = sum / n_runs;

    double sq_sum = 0.0;
    for (double v : samples) sq_sum += (v - mean) * (v - mean);
    double stddev = (n_runs > 1) ? sqrt(sq_sum / (n_runs - 1)) : 0.0;
    float  cv     = (mean > 0.0) ? (float)(stddev / mean) : 0.f;

    return { median, mean, stddev, cv };
}

// ---------------------------------------------------------------------------
// Step 3: Shared-bus processor count
// ---------------------------------------------------------------------------

static bool gpu_present_on_shared_bus() {
#if defined(__APPLE__)
    return true;  // Apple Silicon always has an integrated GPU on the same UMA bus
#elif defined(__linux__)
    if (access("/proc/driver/nvidia/version", F_OK) == 0) return true;
    for (int i = 0; i < 8; ++i) {
        char path[64];
        snprintf(path, sizeof(path), "/dev/dri/renderD%d", 128 + i);
        if (access(path, F_OK) == 0) return true;
    }
    return false;
#else
    return false;
#endif
}

static uint32_t count_shared_bus_processors() {
#if defined(__APPLE__)
    return 2;   // CPU + integrated GPU on LPDDR bus
#else
    return 1;   // CPU-only; GPU contention is not measurable from CPU code
#endif
}

bool blaq_gpu_on_shared_bus(void) {
    return gpu_present_on_shared_bus();
}

// ---------------------------------------------------------------------------
// NVML GPU detection (dynamically loaded — no hard link dependency)
// ---------------------------------------------------------------------------
//
// On NVIDIA systems, NVML gives us:
//   - GPU name (e.g. "NVIDIA GeForce RTX 3070 Ti")
//   - Total VRAM
//   - Memory bus width (bits) and max memory clock (MHz)
//   - Theoretical peak BW = (bus_width/8) * clock_MHz * 2 * 1e6   (DDR formula)
//
// Dynamic loading means the binary runs on non-NVIDIA hardware without
// needing libnvidia-ml installed.

#define BLAQ_NVML_SUCCESS  0
#define BLAQ_NVML_CLOCK_MEM 2

typedef int   blaq_nvml_return_t;
typedef void *blaq_nvml_device_t;
typedef struct { unsigned long long total, free, used; } blaq_nvml_memory_t;

typedef blaq_nvml_return_t (*pfn_nvmlInit)(void);
typedef blaq_nvml_return_t (*pfn_nvmlShutdown)(void);
typedef blaq_nvml_return_t (*pfn_nvmlDeviceGetCount)(unsigned int *);
typedef blaq_nvml_return_t (*pfn_nvmlDeviceGetHandleByIndex)(unsigned int, blaq_nvml_device_t *);
typedef blaq_nvml_return_t (*pfn_nvmlDeviceGetName)(blaq_nvml_device_t, char *, unsigned int);
typedef blaq_nvml_return_t (*pfn_nvmlDeviceGetMemoryInfo)(blaq_nvml_device_t, blaq_nvml_memory_t *);
typedef blaq_nvml_return_t (*pfn_nvmlDeviceGetMemoryBusWidth)(blaq_nvml_device_t, unsigned int *);
typedef blaq_nvml_return_t (*pfn_nvmlDeviceGetMaxClockInfo)(blaq_nvml_device_t, int, unsigned int *);

struct blaq_nvml_gpu_t {
    char     name[256];
    uint64_t vram_bytes;
    uint32_t bus_width_bits;
    uint32_t mem_clock_mhz;
    double   peak_bw_bytes_per_sec;
    bool     valid;
};

static bool nvml_query_first_gpu(blaq_nvml_gpu_t * out) {
    memset(out, 0, sizeof(*out));

#if defined(__APPLE__)
    (void)out;
    return false;  // Apple Silicon uses sysctl, not NVML
#else

#if defined(__linux__)
    void * lib = dlopen("libnvidia-ml.so.1", RTLD_LAZY | RTLD_LOCAL);
    if (!lib) lib = dlopen("libnvidia-ml.so",   RTLD_LAZY | RTLD_LOCAL);
#elif defined(_WIN32)
    void * lib = (void *)LoadLibraryA("nvml.dll");
    if (!lib) {
        char path[MAX_PATH];
        const char * pf = getenv("ProgramFiles");
        snprintf(path, sizeof(path), "%s\\NVIDIA Corporation\\NVSMI\\nvml.dll",
                 pf ? pf : "C:\\Program Files");
        lib = (void *)LoadLibraryA(path);
    }
#else
    void * lib = nullptr;
#endif
    if (!lib) return false;

#if defined(__linux__)
#  define BLAQ_DLSYM(name) dlsym(lib, name)
#elif defined(_WIN32)
#  define BLAQ_DLSYM(name) (void *)GetProcAddress((HMODULE)lib, name)
#else
#  define BLAQ_DLSYM(name) nullptr
#endif

    // Prefer versioned v2 symbols; fall back to v1
    pfn_nvmlInit init_fn =
        (pfn_nvmlInit)BLAQ_DLSYM("nvmlInit_v2");
    if (!init_fn) init_fn =
        (pfn_nvmlInit)BLAQ_DLSYM("nvmlInit");

    pfn_nvmlShutdown shutdown_fn =
        (pfn_nvmlShutdown)BLAQ_DLSYM("nvmlShutdown");

    pfn_nvmlDeviceGetCount count_fn =
        (pfn_nvmlDeviceGetCount)BLAQ_DLSYM("nvmlDeviceGetCount_v2");
    if (!count_fn) count_fn =
        (pfn_nvmlDeviceGetCount)BLAQ_DLSYM("nvmlDeviceGetCount");

    pfn_nvmlDeviceGetHandleByIndex handle_fn =
        (pfn_nvmlDeviceGetHandleByIndex)BLAQ_DLSYM("nvmlDeviceGetHandleByIndex_v2");
    if (!handle_fn) handle_fn =
        (pfn_nvmlDeviceGetHandleByIndex)BLAQ_DLSYM("nvmlDeviceGetHandleByIndex");

    pfn_nvmlDeviceGetName           name_fn     = (pfn_nvmlDeviceGetName)          BLAQ_DLSYM("nvmlDeviceGetName");
    pfn_nvmlDeviceGetMemoryInfo     meminfo_fn  = (pfn_nvmlDeviceGetMemoryInfo)     BLAQ_DLSYM("nvmlDeviceGetMemoryInfo");
    pfn_nvmlDeviceGetMemoryBusWidth buswidth_fn = (pfn_nvmlDeviceGetMemoryBusWidth) BLAQ_DLSYM("nvmlDeviceGetMemoryBusWidth");
    pfn_nvmlDeviceGetMaxClockInfo   maxclock_fn = (pfn_nvmlDeviceGetMaxClockInfo)   BLAQ_DLSYM("nvmlDeviceGetMaxClockInfo");

#undef BLAQ_DLSYM

    bool ok = false;
    if (init_fn && count_fn && handle_fn && name_fn && meminfo_fn
            && init_fn() == BLAQ_NVML_SUCCESS) {
        unsigned int n = 0;
        if (count_fn(&n) == BLAQ_NVML_SUCCESS && n > 0) {
            blaq_nvml_device_t dev = nullptr;
            if (handle_fn(0, &dev) == BLAQ_NVML_SUCCESS) {
                name_fn(dev, out->name, sizeof(out->name));

                blaq_nvml_memory_t mem = {};
                if (meminfo_fn(dev, &mem) == BLAQ_NVML_SUCCESS)
                    out->vram_bytes = mem.total;

                if (buswidth_fn) buswidth_fn(dev, &out->bus_width_bits);
                if (maxclock_fn) maxclock_fn(dev, BLAQ_NVML_CLOCK_MEM, &out->mem_clock_mhz);

                if (out->bus_width_bits > 0 && out->mem_clock_mhz > 0) {
                    // DDR: effective data rate = clock × 2
                    out->peak_bw_bytes_per_sec =
                        (double)(out->bus_width_bits / 8) *
                        (double)out->mem_clock_mhz * 2.0 * 1e6;
                }
                out->valid = out->name[0] != '\0';
                ok = out->valid;
            }
        }
        if (shutdown_fn) shutdown_fn();
    }

#if defined(__linux__)
    dlclose(lib);
#elif defined(_WIN32)
    FreeLibrary((HMODULE)lib);
#endif

    return ok;
#endif // !__APPLE__
}

// ---------------------------------------------------------------------------
// CUDA device-to-device STREAM benchmark (dynamically loaded)
// ---------------------------------------------------------------------------
//
// Measures GPU memory bandwidth using cudaMemcpy device-to-device.
// D2D copies perform one full read + one full write, so:
//   measured_bw = 2 × size × passes / elapsed_s
// This matches the "total bandwidth" convention used in NVIDIA spec sheets.
//
// Returns measured median BW in bytes/sec, or 0.0 if libcudart is unavailable.

#define BLAQ_CUDA_SUCCESS    0
#define BLAQ_CUDA_MEMCPY_D2D 3

typedef int   blaq_cuda_err_t;
typedef void *blaq_cuda_event_t;

typedef blaq_cuda_err_t (*pfn_cudaMalloc)           (void **, size_t);
typedef blaq_cuda_err_t (*pfn_cudaFree)             (void *);
typedef blaq_cuda_err_t (*pfn_cudaMemset)           (void *, int, size_t);
typedef blaq_cuda_err_t (*pfn_cudaMemcpy)           (void *, const void *, size_t, int);
typedef blaq_cuda_err_t (*pfn_cudaEventCreate)      (blaq_cuda_event_t *);
typedef blaq_cuda_err_t (*pfn_cudaEventDestroy)     (blaq_cuda_event_t);
typedef blaq_cuda_err_t (*pfn_cudaEventRecord)      (blaq_cuda_event_t, void *);
typedef blaq_cuda_err_t (*pfn_cudaEventSynchronize) (blaq_cuda_event_t);
typedef blaq_cuda_err_t (*pfn_cudaEventElapsedTime) (float *, blaq_cuda_event_t, blaq_cuda_event_t);
typedef blaq_cuda_err_t (*pfn_cudaDeviceSynchronize)(void);

static double cuda_stream_benchmark(int n_runs, bw_stats_t * stats_out) {
    if (stats_out) memset(stats_out, 0, sizeof(*stats_out));

#if defined(__APPLE__)
    return 0.0;
#else
#if defined(__linux__)
    void * lib = dlopen("libcudart.so",    RTLD_LAZY | RTLD_LOCAL);
    if (!lib) lib = dlopen("libcudart.so.12", RTLD_LAZY | RTLD_LOCAL);
    if (!lib) lib = dlopen("libcudart.so.11", RTLD_LAZY | RTLD_LOCAL);
#elif defined(_WIN32)
    void * lib = (void *)LoadLibraryA("cudart64_12.dll");
    if (!lib) lib = (void *)LoadLibraryA("cudart64_11.dll");
#else
    void * lib = nullptr;
#endif
    if (!lib) return 0.0;

#if defined(__linux__)
#  define BLAQ_CUDA_SYM(n) dlsym(lib, n)
#elif defined(_WIN32)
#  define BLAQ_CUDA_SYM(n) (void *)GetProcAddress((HMODULE)lib, n)
#else
#  define BLAQ_CUDA_SYM(n) nullptr
#endif

    pfn_cudaMalloc            fn_malloc  = (pfn_cudaMalloc)           BLAQ_CUDA_SYM("cudaMalloc");
    pfn_cudaFree              fn_free    = (pfn_cudaFree)             BLAQ_CUDA_SYM("cudaFree");
    pfn_cudaMemset            fn_memset  = (pfn_cudaMemset)           BLAQ_CUDA_SYM("cudaMemset");
    pfn_cudaMemcpy            fn_memcpy  = (pfn_cudaMemcpy)           BLAQ_CUDA_SYM("cudaMemcpy");
    pfn_cudaEventCreate       fn_evcreate= (pfn_cudaEventCreate)      BLAQ_CUDA_SYM("cudaEventCreate");
    pfn_cudaEventDestroy      fn_evdest  = (pfn_cudaEventDestroy)     BLAQ_CUDA_SYM("cudaEventDestroy");
    pfn_cudaEventRecord       fn_evrec   = (pfn_cudaEventRecord)      BLAQ_CUDA_SYM("cudaEventRecord");
    pfn_cudaEventSynchronize  fn_evsync  = (pfn_cudaEventSynchronize) BLAQ_CUDA_SYM("cudaEventSynchronize");
    pfn_cudaEventElapsedTime  fn_elapsed = (pfn_cudaEventElapsedTime) BLAQ_CUDA_SYM("cudaEventElapsedTime");
    pfn_cudaDeviceSynchronize fn_devsync = (pfn_cudaDeviceSynchronize)BLAQ_CUDA_SYM("cudaDeviceSynchronize");

#undef BLAQ_CUDA_SYM

    double result = 0.0;

    if (fn_malloc && fn_free && fn_memset && fn_memcpy &&
        fn_evcreate && fn_evdest && fn_evrec && fn_evsync && fn_elapsed) {

        const size_t BUF_BYTES = 256ULL * 1024 * 1024;  // 256 MB >> L2 on all GPUs
        void * d_src = nullptr;
        void * d_dst = nullptr;

        if (fn_malloc(&d_src, BUF_BYTES) == BLAQ_CUDA_SUCCESS &&
            fn_malloc(&d_dst, BUF_BYTES) == BLAQ_CUDA_SUCCESS &&
            fn_memset(d_src, 0x42, BUF_BYTES) == BLAQ_CUDA_SUCCESS) {

            blaq_cuda_event_t ev_start = nullptr, ev_stop = nullptr;
            fn_evcreate(&ev_start);
            fn_evcreate(&ev_stop);

            // Warmup — wakes up clocks and fills caches
            fn_memcpy(d_dst, d_src, BUF_BYTES, BLAQ_CUDA_MEMCPY_D2D);
            if (fn_devsync) fn_devsync();

            // Timed passes — one event pair per pass for per-pass stats
            std::vector<double> samples;
            samples.reserve(n_runs);
            for (int i = 0; i < n_runs; ++i) {
                fn_evrec(ev_start, nullptr);
                fn_memcpy(d_dst, d_src, BUF_BYTES, BLAQ_CUDA_MEMCPY_D2D);
                fn_evrec(ev_stop, nullptr);
                fn_evsync(ev_stop);
                float ms = 0.f;
                fn_elapsed(&ms, ev_start, ev_stop);
                if (ms > 0.f) {
                    // D2D = 1 read + 1 write = 2 × BUF_BYTES total traffic
                    samples.push_back(2.0 * (double)BUF_BYTES / ((double)ms * 1e-3));
                }
            }

            fn_evdest(ev_start);
            fn_evdest(ev_stop);

            if (!samples.empty()) {
                std::vector<double> sorted = samples;
                std::sort(sorted.begin(), sorted.end());
                int n = (int)sorted.size();
                double median = (n % 2 == 0)
                    ? (sorted[n/2 - 1] + sorted[n/2]) / 2.0
                    : sorted[n/2];
                double sum = std::accumulate(samples.begin(), samples.end(), 0.0);
                double mean = sum / n;
                double sq = 0.0;
                for (double v : samples) sq += (v - mean) * (v - mean);
                double stddev = (n > 1) ? sqrt(sq / (n - 1)) : 0.0;
                float  cv     = (mean > 0.0) ? (float)(stddev / mean) : 0.f;
                if (stats_out) { stats_out->median = median; stats_out->mean = mean;
                                 stats_out->stddev = stddev; stats_out->cv   = cv; }
                result = median;
            }
        }
        if (d_src) fn_free(d_src);
        if (d_dst) fn_free(d_dst);
    }

#if defined(__linux__)
    dlclose(lib);
#elif defined(_WIN32)
    FreeLibrary((HMODULE)lib);
#endif
    return result;
#endif // !__APPLE__
}

static void nvml_derive_vendor(const char * gpu_name, char * vendor, size_t vlen) {
    if (strstr(gpu_name, "NVIDIA") || strstr(gpu_name, "GeForce") ||
        strstr(gpu_name, "Quadro") || strstr(gpu_name, "Tesla")   ||
        strstr(gpu_name, "RTX")    || strstr(gpu_name, "GTX"))
        snprintf(vendor, vlen, "NVIDIA");
    else if (strstr(gpu_name, "AMD") || strstr(gpu_name, "Radeon"))
        snprintf(vendor, vlen, "AMD");
    else if (strstr(gpu_name, "Intel") || strstr(gpu_name, "Arc"))
        snprintf(vendor, vlen, "Intel");
    else
        snprintf(vendor, vlen, "GPU");
}

// ---------------------------------------------------------------------------
// Hardware identification helpers
// ---------------------------------------------------------------------------

static void detect_hw_vendor_chip(char * vendor_out, size_t vlen,
                                  char * chip_out,   size_t clen) {
    vendor_out[0] = chip_out[0] = '\0';

#if defined(__APPLE__)
    char brand[256] = {};
    size_t brand_len = sizeof(brand);
    if (sysctlbyname("machdep.cpu.brand_string", brand, &brand_len, nullptr, 0) == 0
            && brand[0]) {
        snprintf(chip_out, clen, "%s", brand);
        // Derive vendor from chip name
        if (strstr(brand, "Apple")) snprintf(vendor_out, vlen, "Apple");
        else                        snprintf(vendor_out, vlen, "Unknown");
    } else {
        // Fallback for Apple Silicon (M-series reports brand_string correctly
        // but some older SDKs may not expose it)
        char model[64] = {};
        size_t model_len = sizeof(model);
        sysctlbyname("hw.model", model, &model_len, nullptr, 0);
        snprintf(chip_out, clen, "%s", model[0] ? model : "Apple Silicon");
        snprintf(vendor_out, vlen, "Apple");
    }

#elif defined(__linux__)
    FILE * f = fopen("/proc/cpuinfo", "r");
    if (f) {
        char line[256];
        while (fgets(line, sizeof(line), f)) {
            // x86:  "model name	: Intel(R) Core(TM)..." or "model name	: AMD Ryzen..."
            // ARM:  "Model		: Raspberry Pi..." or "Hardware	: BCM..."
            const char * prefixes[] = {
                "model name\t:", "model name :",
                "Model\t\t:",    "Model\t:",
                "Hardware\t:",   nullptr
            };
            for (int i = 0; prefixes[i]; ++i) {
                size_t plen = strlen(prefixes[i]);
                if (strncmp(line, prefixes[i], plen) == 0) {
                    char * val = line + plen;
                    while (*val == ' ') val++;
                    size_t n = strlen(val);
                    if (n > 0 && val[n-1] == '\n') val[--n] = '\0';
                    snprintf(chip_out, clen, "%s", val);
                    break;
                }
            }
            if (chip_out[0]) break;
        }
        fclose(f);
    }

    if (!chip_out[0]) snprintf(chip_out, clen, "Unknown CPU");

    // Detect vendor from chip name
    if      (strstr(chip_out, "Apple"))   snprintf(vendor_out, vlen, "Apple");
    else if (strstr(chip_out, "Intel"))   snprintf(vendor_out, vlen, "Intel");
    else if (strstr(chip_out, "AMD"))     snprintf(vendor_out, vlen, "AMD");
    else if (strstr(chip_out, "Neoverse") ||
             strstr(chip_out, "Cortex")  ||
             strstr(chip_out, "ARM"))     snprintf(vendor_out, vlen, "ARM");
    else if (strstr(chip_out, "NVIDIA")  ||
             strstr(chip_out, "Grace"))   snprintf(vendor_out, vlen, "NVIDIA");
    else                                  snprintf(vendor_out, vlen, "Unknown");

#elif defined(_WIN32)
    HKEY hKey;
    if (RegOpenKeyExA(HKEY_LOCAL_MACHINE,
                      "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0",
                      0, KEY_READ, &hKey) == ERROR_SUCCESS) {
        char name[256] = {};
        DWORD cbData = sizeof(name);
        RegQueryValueExA(hKey, "ProcessorNameString", nullptr, nullptr,
                         (LPBYTE)name, &cbData);
        RegCloseKey(hKey);
        snprintf(chip_out, clen, "%s", name[0] ? name : "Unknown CPU");
    } else {
        snprintf(chip_out, clen, "Unknown CPU");
    }
    if      (strstr(chip_out, "Intel")) snprintf(vendor_out, vlen, "Intel");
    else if (strstr(chip_out, "AMD"))   snprintf(vendor_out, vlen, "AMD");
    else                                snprintf(vendor_out, vlen, "Unknown");
#else
    snprintf(chip_out,   clen, "Unknown CPU");
    snprintf(vendor_out, vlen, "Unknown");
#endif
}

static uint32_t detect_hw_memory_gb() {
#if defined(__APPLE__)
    int64_t memsize = 0;
    size_t sz = sizeof(memsize);
    if (sysctlbyname("hw.memsize", &memsize, &sz, nullptr, 0) == 0 && memsize > 0) {
        return (uint32_t)(memsize / (1024LL * 1024 * 1024));
    }
#elif defined(__linux__)
    FILE * f = fopen("/proc/meminfo", "r");
    if (f) {
        char line[256];
        while (fgets(line, sizeof(line), f)) {
            unsigned long kbytes = 0;
            if (sscanf(line, "MemTotal: %lu kB", &kbytes) == 1) {
                fclose(f);
                return (uint32_t)(kbytes / (1024UL * 1024));
            }
        }
        fclose(f);
    }
#elif defined(_WIN32)
    MEMORYSTATUSEX stat;
    stat.dwLength = sizeof(stat);
    if (GlobalMemoryStatusEx(&stat)) {
        return (uint32_t)(stat.ullTotalPhys / (1024ULL * 1024 * 1024));
    }
#endif
    return 0;
}

static void detect_os_string(char * out, size_t len) {
#if defined(__APPLE__)
    char ver[32] = "unknown";
    size_t sz = sizeof(ver);
    sysctlbyname("kern.osproductversion", ver, &sz, nullptr, 0);
    snprintf(out, len, "macOS %s", ver);
#elif defined(__linux__)
    FILE * f = fopen("/etc/os-release", "r");
    if (f) {
        char line[256];
        while (fgets(line, sizeof(line), f)) {
            if (strncmp(line, "PRETTY_NAME=", 12) == 0) {
                char * start = line + 12;
                if (*start == '"') start++;
                size_t n = strlen(start);
                if (n > 0 && start[n-1] == '\n') start[--n] = '\0';
                if (n > 0 && start[n-1] == '"')  start[--n] = '\0';
                snprintf(out, len, "%s", start);
                fclose(f);
                return;
            }
        }
        fclose(f);
    }
    snprintf(out, len, "Linux");
#elif defined(_WIN32)
    snprintf(out, len, "Windows");
#else
    snprintf(out, len, "Unknown OS");
#endif
}

static void get_timestamp(char * out, size_t len) {
    time_t t = time(nullptr);
    struct tm * tm_info = localtime(&t);
    if (tm_info) strftime(out, len, "%Y-%m-%dT%H:%M:%S", tm_info);
    else         snprintf(out, len, "unknown");
}

// ---------------------------------------------------------------------------
// Main profiling function (Algorithm 1)
// ---------------------------------------------------------------------------

bool blaq_profile_measure(blaq_profile_t * out, float gamma, float beta,
                          float sigma_override,
                          int n_runs, float warmup_s, float measure_s) {
    if (!out) return false;
    if (n_runs < 1)       n_runs    = 1;
    if (measure_s <= 0.f) measure_s = 0.5f;
    if (warmup_s  <  0.f) warmup_s  = 0.f;

    memset(out, 0, sizeof(*out));

    // Step 1: cache-line size
    out->cache_line_bytes = detect_cache_line_bytes();

    // Step 2: detect inference device and measure peak BW
    //   - NVIDIA GPU: query via NVML (theoretical DDR formula, no benchmark needed)
    //   - Apple UMA:  CPU STREAM benchmark == GPU BW (same physical DRAM)
    //   - CPU-only:   CPU STREAM benchmark

#if !defined(__APPLE__)
    blaq_nvml_gpu_t nvml_gpu = {};
    bool gpu_found = nvml_query_first_gpu(&nvml_gpu);
#else
    bool gpu_found = false;
#endif

    if (gpu_found) {
        // --- Discrete GPU path ---
        nvml_derive_vendor(nvml_gpu.name, out->hw_vendor, sizeof(out->hw_vendor));

        // Strip vendor prefix ("NVIDIA GeForce RTX …" → "GeForce RTX …")
        const char * chip = nvml_gpu.name;
        size_t vl = strlen(out->hw_vendor);
        if (BLAQ_STRNCASECMP(chip, out->hw_vendor, vl) == 0 && chip[vl] == ' ')
            chip += vl + 1;
        snprintf(out->hw_chip, sizeof(out->hw_chip), "%s", chip);

        out->hw_memory_gb = (uint32_t)(nvml_gpu.vram_bytes / (1024ULL * 1024 * 1024));
        snprintf(out->hw_type, sizeof(out->hw_type), "gpu");

        fprintf(stderr, "  GPU detected: %s  (%u GB VRAM)\n",
                nvml_gpu.name, out->hw_memory_gb);
        fprintf(stderr, "  Measuring GPU BW (%d pass%s, cudaMemcpy D2D + 1 warmup) ...\n",
                n_runs, n_runs == 1 ? "" : "es");

        bw_stats_t cuda_stats = {};
        double cuda_bw = cuda_stream_benchmark(n_runs, &cuda_stats);

        if (cuda_bw > 0.0) {
            out->peak_bw_bytes_per_sec = cuda_stats.median;
            out->peak_bw_cv            = cuda_stats.cv;
            out->n_runs                = (uint32_t)n_runs;
            snprintf(out->bw_source, sizeof(out->bw_source), "cuda-stream-median");
            fprintf(stderr, "  Peak BW: %.1f GB/s  (CUDA D2D median of %d runs, CV=%.3f)\n",
                    cuda_bw / 1e9, n_runs, (double)cuda_stats.cv);
        } else if (nvml_gpu.peak_bw_bytes_per_sec > 0.0) {
            // CUDA runtime unavailable — fall back to NVML theoretical
            fprintf(stderr,
                "  WARNING: libcudart not found — using NVML theoretical BW as fallback.\n"
                "  Peak BW: %.1f GB/s  (NVML: %u-bit × %u MHz DDR)\n",
                nvml_gpu.peak_bw_bytes_per_sec / 1e9,
                nvml_gpu.bus_width_bits, nvml_gpu.mem_clock_mhz);
            out->peak_bw_bytes_per_sec = nvml_gpu.peak_bw_bytes_per_sec;
            out->peak_bw_cv            = 0.f;
            out->n_runs                = 0;
            snprintf(out->bw_source, sizeof(out->bw_source), "nvml-theoretical");
        } else {
            fprintf(stderr, "  WARNING: Could not measure GPU BW (no CUDA, no NVML clock info).\n");
            out->peak_bw_bytes_per_sec = 0.0;
            out->n_runs                = 0;
            snprintf(out->bw_source, sizeof(out->bw_source), "unknown");
        }

        // Contention: discrete GPU VRAM is not shared with the CPU bus.
        // sigma=0 unless the user overrides (e.g. for a true UMA discrete GPU).
        if (sigma_override >= 0.f) {
            out->contention_ratio     = sigma_override;
            out->obs_bw_bytes_per_sec = out->peak_bw_bytes_per_sec * (1.0 - sigma_override);
            out->obs_bw_cv            = 0.f;
        } else {
            out->contention_ratio     = 0.f;
            out->obs_bw_bytes_per_sec = out->peak_bw_bytes_per_sec;
            out->obs_bw_cv            = 0.f;
            fprintf(stderr,
                "  Discrete GPU: separate VRAM bus → sigma=0.0 (no CPU/GPU contention).\n"
                "  Use --sigma <value> if your GPU shares a memory bus with the CPU.\n\n");
        }

        out->shared_bus_procs = 1;  // GPU VRAM bus is private to the GPU

    } else {
        // --- CPU / Apple UMA path ---
        detect_hw_vendor_chip(out->hw_vendor, sizeof(out->hw_vendor),
                              out->hw_chip,   sizeof(out->hw_chip));
        out->hw_memory_gb = detect_hw_memory_gb();
#if defined(__APPLE__)
        snprintf(out->hw_type, sizeof(out->hw_type), "uma");
#else
        snprintf(out->hw_type, sizeof(out->hw_type), "cpu");
#endif
        snprintf(out->bw_source, sizeof(out->bw_source), "stream-median");

        // Step 2: peak single-agent BW
        fprintf(stderr, "  Measuring peak BW (%d pass%s × %.1f s, %.1f s warmup) ...\n",
                n_runs, n_runs == 1 ? "" : "es", (double)measure_s, (double)warmup_s);
        bw_stats_t peak = stream_benchmark_stats(1, warmup_s, measure_s, n_runs);
        out->peak_bw_bytes_per_sec = peak.median;
        out->peak_bw_cv            = peak.cv;
        out->n_runs                = (uint32_t)n_runs;

        // Step 3: shared-bus processor count
        out->shared_bus_procs = count_shared_bus_processors();

        // Step 4: contention BW
        if (sigma_override >= 0.f) {
            out->obs_bw_bytes_per_sec = out->peak_bw_bytes_per_sec * (1.0 - sigma_override);
            out->obs_bw_cv            = 0.f;
        } else {
            fprintf(stderr, "  Measuring contention BW (%d pass%s × %.1f s, %.1f s warmup) ...\n",
                    n_runs, n_runs == 1 ? "" : "es", (double)measure_s, (double)warmup_s);
            bw_stats_t obs = stream_benchmark_stats((int)out->shared_bus_procs,
                                                    warmup_s, measure_s, n_runs);
            out->obs_bw_bytes_per_sec = obs.median;
            out->obs_bw_cv            = obs.cv;
        }

        // Contention ratio
        if (sigma_override >= 0.f) {
            out->contention_ratio = sigma_override;
        } else if (out->peak_bw_bytes_per_sec > 0.0) {
            out->contention_ratio =
                (float)(1.0 - out->obs_bw_bytes_per_sec / out->peak_bw_bytes_per_sec);
        }
        if (out->contention_ratio < 0.f) out->contention_ratio = 0.f;
        if (out->contention_ratio > 1.f) out->contention_ratio = 1.f;

        if (gpu_present_on_shared_bus() && out->shared_bus_procs == 1
                && sigma_override < 0.f) {
            fprintf(stderr,
                "\nWARNING: GPU detected but contention was measured with CPU threads only.\n"
                "         On NVLink-C2C systems (e.g. GB10) the GPU-HBM bus is independent\n"
                "         of CPU LPDDR, so sigma=%.3f may be inaccurate.\n"
                "         Use --sigma <value> to supply a known contention ratio.\n\n",
                (double)out->contention_ratio);
        }
    }

    // Step 6: penalty weights
    out->lambda_bw  = (out->peak_bw_bytes_per_sec > 0.0)
        ? (float)(gamma / out->peak_bw_bytes_per_sec) : 0.f;
    out->lambda_mem = beta * out->contention_ratio;

    detect_os_string(out->hw_os, sizeof(out->hw_os));
    get_timestamp(out->timestamp, sizeof(out->timestamp));

    return true;
}

// ---------------------------------------------------------------------------
// Defaults (no profiling data)
// ---------------------------------------------------------------------------

void blaq_profile_defaults(blaq_profile_t * out) {
    if (!out) return;
    memset(out, 0, sizeof(*out));
    out->cache_line_bytes      = 64;
    out->peak_bw_bytes_per_sec = 100.0 * 1e9;
    out->obs_bw_bytes_per_sec  = 80.0  * 1e9;
    out->contention_ratio      = 0.20f;
    out->shared_bus_procs      = 2;
    out->lambda_bw             = (float)(1.0 / out->peak_bw_bytes_per_sec);
    out->lambda_mem            = 1.0f * out->contention_ratio;
    out->n_runs                = 0;
    out->peak_bw_cv            = 0.f;
    out->obs_bw_cv             = 0.f;
    snprintf(out->hw_vendor,  sizeof(out->hw_vendor),  "Unknown");
    snprintf(out->hw_chip,    sizeof(out->hw_chip),    "Unknown");
    snprintf(out->hw_os,      sizeof(out->hw_os),      "Unknown");
    snprintf(out->hw_type,   sizeof(out->hw_type),   "cpu");
    snprintf(out->bw_source, sizeof(out->bw_source), "defaults");
    snprintf(out->timestamp, sizeof(out->timestamp), "defaults");
}

// ---------------------------------------------------------------------------
// Suggested filename
// ---------------------------------------------------------------------------

char * blaq_profile_suggest_filename(const blaq_profile_t * p, char * buf, size_t len) {
    // Normalise vendor to lowercase
    char vendor[64] = "unknown";
    if (p->hw_vendor[0]) {
        size_t i = 0;
        for (; p->hw_vendor[i] && i < sizeof(vendor) - 1; i++) {
            vendor[i] = (char)tolower((unsigned char)p->hw_vendor[i]);
        }
        vendor[i] = '\0';
    }

    // Chip: strip leading vendor prefix ("Apple M1" → "M1"), normalise
    const char * chip_src = p->hw_chip;
    if (p->hw_vendor[0]) {
        size_t vl = strlen(p->hw_vendor);
        if (BLAQ_STRNCASECMP(chip_src, p->hw_vendor, vl) == 0 && chip_src[vl] == ' ') {
            chip_src += vl + 1;
        }
    }
    char chip[128] = "unknown";
    size_t ci = 0;
    for (size_t i = 0; chip_src[i] && ci < sizeof(chip) - 1; i++) {
        // Skip trademark/registered symbols: (R), (TM), (r), (tm)
        if (chip_src[i] == '(') {
            size_t j = i + 1;
            while (chip_src[j] && chip_src[j] != ')') j++;
            if (chip_src[j] == ')') { i = j; continue; }
        }
        char c = chip_src[i];
        if (c == ' ' || c == '_') {
            c = '-';
        } else {
            c = (char)tolower((unsigned char)c);
        }
        if (isalnum((unsigned char)c) || c == '-') {
            // Collapse consecutive hyphens
            if (c == '-' && ci > 0 && chip[ci-1] == '-') continue;
            chip[ci++] = c;
        }
    }
    // Strip trailing hyphen
    while (ci > 0 && chip[ci-1] == '-') ci--;
    chip[ci] = '\0';

    const char * type = p->hw_type[0] ? p->hw_type : "cpu";
    if (p->hw_memory_gb > 0) {
        snprintf(buf, len, "blaq_profile_%s_%s_%s_%ugb.json", type, vendor, chip, p->hw_memory_gb);
    } else {
        snprintf(buf, len, "blaq_profile_%s_%s_%s.json", type, vendor, chip);
    }
    return buf;
}

// ---------------------------------------------------------------------------
// JSON serialisation
// ---------------------------------------------------------------------------

bool blaq_profile_save_json(const blaq_profile_t * p, const char * path) {
    if (!p || !path) return false;

    nlohmann::ordered_json j;
    // Inference device identity
    j["hw_vendor"]             = p->hw_vendor;
    j["hw_chip"]               = p->hw_chip;
    j["hw_os"]                 = p->hw_os;
    j["hw_memory_gb"]          = p->hw_memory_gb;
    j["hw_type"]               = p->hw_type;
    j["timestamp"]             = p->timestamp;

    // Measurement metadata
    j["bw_source"]             = p->bw_source;
    j["n_runs"]                = p->n_runs;
    j["peak_bw_cv"]            = p->peak_bw_cv;
    j["obs_bw_cv"]             = p->obs_bw_cv;

    // BLAQ parameters
    j["cache_line_bytes"]      = p->cache_line_bytes;
    j["peak_bw_bytes_per_sec"] = p->peak_bw_bytes_per_sec;
    j["obs_bw_bytes_per_sec"]  = p->obs_bw_bytes_per_sec;
    j["contention_ratio"]      = p->contention_ratio;
    j["shared_bus_procs"]      = p->shared_bus_procs;
    j["lambda_bw"]             = p->lambda_bw;
    j["lambda_mem"]            = p->lambda_mem;

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
    try { ifs >> j; } catch (...) { return false; }

    memset(out, 0, sizeof(*out));

    // Helper: required fields — return false immediately if absent or wrong type
    auto req_u32 = [&](const char * key, uint32_t & dst) -> bool {
        if (!j.contains(key) || !j[key].is_number()) {
            fprintf(stderr, "blaq_profile_load_json: missing required field '%s'\n", key);
            return false;
        }
        dst = j[key].get<uint32_t>();
        return true;
    };
    auto req_dbl = [&](const char * key, double & dst) -> bool {
        if (!j.contains(key) || !j[key].is_number()) {
            fprintf(stderr, "blaq_profile_load_json: missing required field '%s'\n", key);
            return false;
        }
        dst = j[key].get<double>();
        return true;
    };
    auto req_flt = [&](const char * key, float & dst) -> bool {
        if (!j.contains(key) || !j[key].is_number()) {
            fprintf(stderr, "blaq_profile_load_json: missing required field '%s'\n", key);
            return false;
        }
        dst = j[key].get<float>();
        return true;
    };
    auto req_str = [&](const char * key, char * dst, size_t dlen) -> bool {
        if (!j.contains(key) || !j[key].is_string()) {
            fprintf(stderr, "blaq_profile_load_json: missing required field '%s'\n", key);
            return false;
        }
        std::string s = j[key].get<std::string>();
        snprintf(dst, dlen, "%s", s.c_str());
        return true;
    };

    // Optional fields — silently ignored if absent
    auto opt_u32 = [&](const char * key, uint32_t & dst) {
        if (j.contains(key) && j[key].is_number()) dst = j[key].get<uint32_t>();
    };
    auto opt_flt = [&](const char * key, float & dst) {
        if (j.contains(key) && j[key].is_number()) dst = j[key].get<float>();
    };
    auto opt_str = [&](const char * key, char * dst, size_t dlen) {
        if (j.contains(key) && j[key].is_string()) {
            std::string s = j[key].get<std::string>();
            snprintf(dst, dlen, "%s", s.c_str());
        }
    };

    // Required: inference device identity
    if (!req_str("hw_vendor",             out->hw_vendor,  sizeof(out->hw_vendor)))  return false;
    if (!req_str("hw_chip",               out->hw_chip,    sizeof(out->hw_chip)))    return false;
    if (!req_str("hw_type",               out->hw_type,    sizeof(out->hw_type)))    return false;

    // Required: core BLAQ parameters
    if (!req_u32("cache_line_bytes",      out->cache_line_bytes))        return false;
    if (!req_dbl("peak_bw_bytes_per_sec", out->peak_bw_bytes_per_sec))   return false;
    if (!req_dbl("obs_bw_bytes_per_sec",  out->obs_bw_bytes_per_sec))    return false;
    if (!req_flt("contention_ratio",      out->contention_ratio))        return false;
    if (!req_u32("shared_bus_procs",      out->shared_bus_procs))        return false;
    if (!req_flt("lambda_bw",             out->lambda_bw))               return false;
    if (!req_flt("lambda_mem",            out->lambda_mem))              return false;

    // Optional: supplementary fields
    opt_str("hw_os",        out->hw_os,        sizeof(out->hw_os));
    opt_u32("hw_memory_gb", out->hw_memory_gb);
    opt_str("bw_source",    out->bw_source,    sizeof(out->bw_source));
    opt_str("timestamp",    out->timestamp,    sizeof(out->timestamp));
    opt_u32("n_runs",       out->n_runs);
    opt_flt("peak_bw_cv",   out->peak_bw_cv);
    opt_flt("obs_bw_cv",    out->obs_bw_cv);

    return true;
}

// ---------------------------------------------------------------------------
// Human-readable print
// ---------------------------------------------------------------------------

void blaq_profile_print(const blaq_profile_t * p) {
    if (!p) return;
    fprintf(stderr, "\n=== BLAQ Hardware Profile ===\n");

    // Infer device kind label from hw_type
    const bool is_gpu = (strncmp(p->hw_type, "gpu", 3) == 0);
    const bool is_uma = (strncmp(p->hw_type, "uma", 3) == 0);
    const char * kind = is_gpu ? "discrete GPU" : (is_uma ? "unified memory" : "CPU");

    if (p->hw_chip[0]) {
        fprintf(stderr, "  Inference device      : %s%s%s  [%s]\n",
                p->hw_vendor[0] ? p->hw_vendor : "",
                p->hw_vendor[0] ? " " : "",
                p->hw_chip, kind);
    }
    const char * mem_label = is_gpu ? "VRAM" : "Memory";
    if (p->hw_memory_gb > 0) {
        fprintf(stderr, "  %-22s: %u GB\n", mem_label, p->hw_memory_gb);
    }
    if (p->hw_os[0])    fprintf(stderr, "  OS                    : %s\n", p->hw_os);
    if (p->timestamp[0]) fprintf(stderr, "  Measured              : %s\n", p->timestamp);
    fprintf(stderr, "\n");

    // Bandwidth
    fprintf(stderr, "  Cache-line size       : %u bytes\n", p->cache_line_bytes);

    const bool bw_measured = (p->n_runs > 0);
    const bool bw_nvml     = (strcmp(p->bw_source, "nvml-theoretical") == 0);
    const char * bw_tag    = bw_nvml ? "NVML theoretical" : p->bw_source;

    if (bw_measured) {
        fprintf(stderr, "  Peak BW               : %.1f GB/s  (median of %u runs, CV=%.3f%s)\n",
                p->peak_bw_bytes_per_sec / 1e9, p->n_runs, (double)p->peak_bw_cv,
                p->peak_bw_cv > 0.05f ? "  WARNING: high variance" : "");
    } else {
        fprintf(stderr, "  Peak BW               : %.1f GB/s  (%s)\n",
                p->peak_bw_bytes_per_sec / 1e9, bw_tag[0] ? bw_tag : "source unknown");
    }

    if (is_gpu && p->contention_ratio < 0.005f) {
        fprintf(stderr, "  Contention BW         : %.1f GB/s  (sigma=0, discrete GPU)\n",
                p->obs_bw_bytes_per_sec / 1e9);
    } else if (bw_measured) {
        fprintf(stderr, "  Contention BW         : %.1f GB/s  (median of %u runs, CV=%.3f%s)\n",
                p->obs_bw_bytes_per_sec / 1e9, p->n_runs, (double)p->obs_bw_cv,
                p->obs_bw_cv > 0.05f ? "  WARNING: high variance" : "");
    } else {
        fprintf(stderr, "  Contention BW         : %.1f GB/s\n",
                p->obs_bw_bytes_per_sec / 1e9);
    }

    fprintf(stderr, "  Contention ratio (σ)  : %.3f\n", (double)p->contention_ratio);
    fprintf(stderr, "  Shared-bus procs      : %u\n",   p->shared_bus_procs);
    fprintf(stderr, "  lambda_bw             : %.3e\n", (double)p->lambda_bw);
    fprintf(stderr, "  lambda_mem            : %.3f\n", (double)p->lambda_mem);
    fprintf(stderr, "==============================\n\n");
}
