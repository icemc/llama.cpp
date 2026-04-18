#pragma once

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

//
// BLAQ Hardware Profile
//
// Produced by BLAQ-Profile (Algorithm 1 in the BLAQ paper).
// Characterises the host SoC cache geometry and memory bandwidth,
// then derives the penalty weights (lambda_bw, lambda_mem) used in
// the BLAQ objective function:
//
//   L(W, W_hat, pi) = E_rec + lambda_bw * C_bw + lambda_mem * C_mem
//

typedef struct blaq_profile {
    // --- hardware geometry ---
    uint32_t cache_line_bytes;       // L:          cache-line size in bytes (e.g. 64 or 128)
    double   peak_bw_bytes_per_sec;  // BW_peak:    median single-agent sequential read bandwidth
    double   obs_bw_bytes_per_sec;   // BW_obs:     median bandwidth under contention
    float    contention_ratio;       // sigma:      1 - BW_obs / BW_peak  in [0, 1]
    uint32_t shared_bus_procs;       // P_count:    number of processors sharing the bus

    // --- derived alignment ---
    uint32_t aligned_group_128;      // s*(b=4, L):   = 8*L/4  (cache-aligned block size at 4-bit)
    uint32_t aligned_group_256;      // 2 * aligned_group_128 (double-line variant)

    // --- penalty weights ---
    float    lambda_bw;              // gamma / BW_peak
    float    lambda_mem;             // beta  * sigma

    // --- hardware identification (auto-detected) ---
    char     hw_vendor[64];          // e.g. "Apple", "NVIDIA", "Intel", "ARM"
    char     hw_chip[128];           // e.g. "M1", "M2 Pro", "Grace-Blackwell GB10"
    char     hw_os[64];              // e.g. "macOS 15.4", "Ubuntu 22.04 LTS"
    uint32_t hw_memory_gb;           // total system memory in GB

    // --- measurement statistics ---
    uint32_t n_runs;                 // number of timed passes used per measurement
    float    peak_bw_cv;             // CV = stddev/mean across n_runs passes for peak BW
    float    obs_bw_cv;              // CV = stddev/mean across n_runs passes for contention BW
    char     timestamp[32];          // ISO 8601 timestamp of when the profile was generated
} blaq_profile_t;

//
// Measure the hardware profile on the current device.
//
// gamma, beta:       user-tunable scaling constants (safe defaults: 1.0, 1.0)
// sigma_override:    if >= 0.0, skip contention benchmark and use this value
//                    directly for contention_ratio.  Pass -1.0f to auto-measure.
// n_runs:            number of timed measurement passes per BW direction (default 5).
//                    Median across passes is used; higher N reduces variance.
// warmup_s:          warmup duration in seconds before measurement begins (default 1.0).
//                    Warmup results are discarded.
// measure_s:         duration of each timed pass in seconds (default 1.0).
//
// Returns true on success.
//
bool blaq_profile_measure(blaq_profile_t * out, float gamma, float beta,
                          float sigma_override,
                          int n_runs, float warmup_s, float measure_s);

//
// Returns true if a discrete or integrated GPU sharing the DRAM bus was
// detected.  On such systems, CPU-only contention measurement underestimates
// true sigma; pass an explicit sigma_override or use published specs.
//
bool blaq_gpu_on_shared_bus(void);

//
// Load / save a hardware profile as a self-descriptive JSON file.
// Use blaq_profile_save_json() after measuring on a reference machine,
// then blaq_profile_load_json() on the quantization host.
//
bool blaq_profile_load_json(blaq_profile_t * out, const char * path);
bool blaq_profile_save_json(const blaq_profile_t * p, const char * path);

//
// Print a human-readable summary to stderr.
//
void blaq_profile_print(const blaq_profile_t * p);

//
// Fill *out with conservative defaults when no profiling data is available.
// Assumes 64-byte cache lines (universal x86-64 / ARM Neoverse default)
// and a nominal 100 GB/s bandwidth with zero contention.
//
void blaq_profile_defaults(blaq_profile_t * out);

//
// Auto-generate a self-descriptive filename for this profile.
// Format: blaq-prof_<vendor>_<chip>_<ram>gb.json
// Example: blaq-prof_apple_m1_8gb.json
//          blaq-prof_nvidia_gh100_128gb.json
//
// buf:  output buffer
// len:  buffer length (recommend >= 128)
// Returns buf.
//
char * blaq_profile_suggest_filename(const blaq_profile_t * p, char * buf, size_t len);

#ifdef __cplusplus
}
#endif
