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
    if (n_runs < 1)   n_runs   = 1;
    if (measure_s <= 0.f) measure_s = 0.5f;
    if (warmup_s  <  0.f) warmup_s  = 0.f;

    // Step 1
    out->cache_line_bytes = detect_cache_line_bytes();

    // Step 2: peak single-agent bandwidth
    fprintf(stderr, "  Measuring peak BW (%d pass%s × %.1f s, %.1f s warmup) ...\n",
            n_runs, n_runs == 1 ? "" : "es", (double)measure_s, (double)warmup_s);
    bw_stats_t peak = stream_benchmark_stats(1, warmup_s, measure_s, n_runs);
    out->peak_bw_bytes_per_sec = peak.median;
    out->peak_bw_cv            = peak.cv;

    // Step 3
    out->shared_bus_procs = count_shared_bus_processors();

    // Step 4: contention bandwidth
    if (sigma_override >= 0.f) {
        out->obs_bw_bytes_per_sec =
            out->peak_bw_bytes_per_sec * (1.0 - (double)sigma_override);
        out->obs_bw_cv = 0.f;
    } else {
        fprintf(stderr, "  Measuring contention BW (%d pass%s × %.1f s, %.1f s warmup) ...\n",
                n_runs, n_runs == 1 ? "" : "es", (double)measure_s, (double)warmup_s);
        bw_stats_t obs = stream_benchmark_stats((int)out->shared_bus_procs,
                                                warmup_s, measure_s, n_runs);
        out->obs_bw_bytes_per_sec = obs.median;
        out->obs_bw_cv            = obs.cv;
    }

    // Step 5: contention ratio
    if (sigma_override >= 0.f) {
        out->contention_ratio = sigma_override;
    } else if (out->peak_bw_bytes_per_sec > 0.0) {
        out->contention_ratio =
            (float)(1.0 - out->obs_bw_bytes_per_sec / out->peak_bw_bytes_per_sec);
    } else {
        out->contention_ratio = 0.f;
    }
    if (out->contention_ratio < 0.f) out->contention_ratio = 0.f;
    if (out->contention_ratio > 1.f) out->contention_ratio = 1.f;

    if (gpu_present_on_shared_bus() && out->shared_bus_procs == 1
            && sigma_override < 0.f) {
        fprintf(stderr,
            "\nWARNING: GPU detected but contention ratio (sigma) was measured\n"
            "         using CPU threads only.  On discrete-GPU or NVLink-C2C\n"
            "         systems (e.g. Grace-Blackwell GB10) the GPU-HBM bus is\n"
            "         independent of CPU LPDDR, so sigma=%.3f may be inaccurate.\n"
            "         Use --sigma <value> to supply a known contention ratio.\n"
            "         Typical values: 0.10-0.25 for shared-BW systems.\n\n",
            (double)out->contention_ratio);
    }

    // Step 6: aligned group sizes for b=4
    out->aligned_group_128 = 2 * out->cache_line_bytes;
    out->aligned_group_256 = 4 * out->cache_line_bytes;

    // Step 7: penalty weights
    out->lambda_bw  = (out->peak_bw_bytes_per_sec > 0.0)
        ? (float)(gamma / out->peak_bw_bytes_per_sec) : 0.f;
    out->lambda_mem = beta * out->contention_ratio;

    // Hardware identification
    detect_hw_vendor_chip(out->hw_vendor, sizeof(out->hw_vendor),
                          out->hw_chip,   sizeof(out->hw_chip));
    detect_os_string(out->hw_os, sizeof(out->hw_os));
    out->hw_memory_gb = detect_hw_memory_gb();

    // Measurement metadata
    out->n_runs    = (uint32_t)n_runs;
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
    out->aligned_group_128     = 128;
    out->aligned_group_256     = 256;
    out->lambda_bw             = (float)(1.0 / out->peak_bw_bytes_per_sec);
    out->lambda_mem            = 1.0f * out->contention_ratio;
    out->n_runs                = 0;
    out->peak_bw_cv            = 0.f;
    out->obs_bw_cv             = 0.f;
    snprintf(out->hw_vendor,  sizeof(out->hw_vendor),  "Unknown");
    snprintf(out->hw_chip,    sizeof(out->hw_chip),    "Unknown");
    snprintf(out->hw_os,      sizeof(out->hw_os),      "Unknown");
    snprintf(out->timestamp,  sizeof(out->timestamp),  "defaults");
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

    if (p->hw_memory_gb > 0) {
        snprintf(buf, len, "blaq-prof_%s_%s_%ugb.json", vendor, chip, p->hw_memory_gb);
    } else {
        snprintf(buf, len, "blaq-prof_%s_%s.json", vendor, chip);
    }
    return buf;
}

// ---------------------------------------------------------------------------
// JSON serialisation
// ---------------------------------------------------------------------------

bool blaq_profile_save_json(const blaq_profile_t * p, const char * path) {
    if (!p || !path) return false;

    json j;
    // Hardware identity first — makes the file self-descriptive at the top
    j["hw_vendor"]             = p->hw_vendor;
    j["hw_chip"]               = p->hw_chip;
    j["hw_os"]                 = p->hw_os;
    j["hw_memory_gb"]          = p->hw_memory_gb;
    j["timestamp"]             = p->timestamp;

    // Core BLAQ parameters
    j["cache_line_bytes"]      = p->cache_line_bytes;
    j["peak_bw_bytes_per_sec"] = p->peak_bw_bytes_per_sec;
    j["obs_bw_bytes_per_sec"]  = p->obs_bw_bytes_per_sec;
    j["contention_ratio"]      = p->contention_ratio;
    j["shared_bus_procs"]      = p->shared_bus_procs;
    j["aligned_group_128"]     = p->aligned_group_128;
    j["aligned_group_256"]     = p->aligned_group_256;
    j["lambda_bw"]             = p->lambda_bw;
    j["lambda_mem"]            = p->lambda_mem;

    // Measurement statistics
    j["n_runs"]                = p->n_runs;
    j["peak_bw_cv"]            = p->peak_bw_cv;
    j["obs_bw_cv"]             = p->obs_bw_cv;

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

    // Required: hardware identity
    if (!req_str("hw_vendor",           out->hw_vendor,  sizeof(out->hw_vendor)))  return false;
    if (!req_str("hw_chip",             out->hw_chip,    sizeof(out->hw_chip)))    return false;

    // Required: core BLAQ parameters
    if (!req_u32("cache_line_bytes",    out->cache_line_bytes))       return false;
    if (!req_dbl("peak_bw_bytes_per_sec", out->peak_bw_bytes_per_sec)) return false;
    if (!req_dbl("obs_bw_bytes_per_sec",  out->obs_bw_bytes_per_sec))  return false;
    if (!req_flt("contention_ratio",    out->contention_ratio))       return false;
    if (!req_u32("shared_bus_procs",    out->shared_bus_procs))       return false;
    if (!req_u32("aligned_group_128",   out->aligned_group_128))      return false;
    if (!req_u32("aligned_group_256",   out->aligned_group_256))      return false;
    if (!req_flt("lambda_bw",           out->lambda_bw))              return false;
    if (!req_flt("lambda_mem",          out->lambda_mem))             return false;

    // Optional: supplementary identity and measurement stats
    opt_str("hw_os",       out->hw_os,      sizeof(out->hw_os));
    opt_u32("hw_memory_gb", out->hw_memory_gb);
    opt_str("timestamp",   out->timestamp,  sizeof(out->timestamp));
    opt_u32("n_runs",      out->n_runs);
    opt_flt("peak_bw_cv",  out->peak_bw_cv);
    opt_flt("obs_bw_cv",   out->obs_bw_cv);

    return true;
}

// ---------------------------------------------------------------------------
// Human-readable print
// ---------------------------------------------------------------------------

void blaq_profile_print(const blaq_profile_t * p) {
    if (!p) return;
    fprintf(stderr, "\n=== BLAQ Hardware Profile ===\n");

    // Hardware identity
    if (p->hw_chip[0]) {
        fprintf(stderr, "  Device                : %s %s\n",
                p->hw_vendor[0] ? p->hw_vendor : "",
                p->hw_chip);
    }
    if (p->hw_os[0]) {
        fprintf(stderr, "  OS                    : %s\n", p->hw_os);
    }
    if (p->hw_memory_gb > 0) {
        fprintf(stderr, "  Memory                : %u GB\n", p->hw_memory_gb);
    }
    if (p->timestamp[0]) {
        fprintf(stderr, "  Measured              : %s\n", p->timestamp);
    }
    fprintf(stderr, "\n");

    // Bandwidth measurements
    fprintf(stderr, "  Cache-line size       : %u bytes\n",  p->cache_line_bytes);

    if (p->n_runs > 0) {
        fprintf(stderr, "  Peak BW               : %.1f GB/s  (median of %u runs, CV=%.3f%s)\n",
                p->peak_bw_bytes_per_sec / 1e9, p->n_runs, (double)p->peak_bw_cv,
                p->peak_bw_cv > 0.05f ? " ⚠ high variance" : "");
    } else {
        fprintf(stderr, "  Peak BW               : %.1f GB/s\n",
                p->peak_bw_bytes_per_sec / 1e9);
    }

    if (p->n_runs > 0) {
        fprintf(stderr, "  Contention BW         : %.1f GB/s  (median of %u runs, CV=%.3f%s)\n",
                p->obs_bw_bytes_per_sec / 1e9, p->n_runs, (double)p->obs_bw_cv,
                p->obs_bw_cv > 0.05f ? " ⚠ high variance" : "");
    } else {
        fprintf(stderr, "  Contention BW         : %.1f GB/s\n",
                p->obs_bw_bytes_per_sec / 1e9);
    }

    const bool gpu_present = gpu_present_on_shared_bus();
    const bool sigma_zero  = (p->contention_ratio < 0.005f);
    fprintf(stderr, "  Contention ratio (σ)  : %.3f%s\n",
            (double)p->contention_ratio,
            (gpu_present && sigma_zero) ? "  [GPU present — CPU-only measurement]" : "");

    fprintf(stderr, "  Shared-bus procs      : %u%s\n",
            p->shared_bus_procs,
            gpu_present ? "  (CPU only; GPU not included)" : "");
    fprintf(stderr, "  Aligned group (4-bit) : %u weights  (%u-byte cache line)\n",
            p->aligned_group_128, p->cache_line_bytes);
    fprintf(stderr, "  Double-line group     : %u weights\n", p->aligned_group_256);
    fprintf(stderr, "  lambda_bw             : %.3e\n",   (double)p->lambda_bw);
    fprintf(stderr, "  lambda_mem            : %.3f\n",   (double)p->lambda_mem);
    fprintf(stderr, "==============================\n\n");
}
