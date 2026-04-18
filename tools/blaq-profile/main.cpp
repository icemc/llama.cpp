//
// llama-blaq-profile — BLAQ Hardware Profiler CLI
//
// Runs Algorithm 1 (BLAQ-Probe) on the current machine and writes a JSON
// profile that can be passed to llama-quantize via --blaq-profile.
//
// Usage:
//   llama-blaq-profile [options]
//
//   --out  <path>       output JSON file  (default: auto-named, e.g. blaq-prof_apple_m1_8gb.json)
//   --in   <path>       load existing JSON and print it (skips benchmarking)
//   --runs <N>          number of timed passes per BW direction (default: 5)
//   --warmup-time <s>   warmup duration in seconds before measurement (default: 1.0)
//   --measure-time <s>  duration of each timed pass in seconds (default: 1.0)
//   --gamma <f>         bandwidth penalty scale  (default: 1.0)
//   --beta  <f>         contention penalty scale (default: 1.0)
//   --sigma <f>         override contention ratio in [0,1] instead of measuring
//   --help              show this message
//

#include "blaq-profile.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>

static void print_usage(const char * prog) {
    fprintf(stderr,
        "Usage: %s [options]\n"
        "\n"
        "Options:\n"
        "  --out  <path>       write profile JSON (default: auto-named from hardware)\n"
        "  --in   <path>       load and print an existing profile (no benchmarking)\n"
        "  --runs <N>          timed passes per BW direction (default: 5)\n"
        "  --warmup-time <s>   warmup before measurement, discarded (default: 1.0 s)\n"
        "  --measure-time <s>  duration of each timed pass (default: 1.0 s)\n"
        "  --gamma <f>         bandwidth penalty scale  (default: 1.0)\n"
        "  --beta  <f>         contention penalty scale (default: 1.0)\n"
        "  --sigma <f>         override contention ratio in [0,1] instead of measuring\n"
        "                      (required on GPU systems where CPU measurement gives sigma=0;\n"
        "                       typical values: 0.10-0.25 for shared-BW architectures)\n"
        "  --help              show this message\n"
        "\n"
        "Examples:\n"
        "  %s                                  # measure, auto-name output\n"
        "  %s --out my_machine.json            # measure, custom output name\n"
        "  %s --sigma 0.15 --out gb10.json     # Grace-Blackwell with known sigma\n"
        "  %s --in blaq-prof_apple_m1_8gb.json # inspect saved profile\n"
        "\n"
        "Naming convention for profiles:\n"
        "  blaq-prof_<vendor>_<chip>_<ram>gb.json\n"
        "  e.g.  blaq-prof_apple_m1_8gb.json\n"
        "        blaq-prof_nvidia_gh100_128gb.json\n"
        "        blaq-prof_intel_core-ultra9_32gb.json\n"
        "\n",
        prog, prog, prog, prog, prog);
}

int main(int argc, char ** argv) {
    const char * out_path    = nullptr;   // nullptr = auto-generate after profiling
    const char * in_path     = nullptr;
    float gamma          = 1.0f;
    float beta           = 1.0f;
    float sigma_override = -1.0f;
    int   n_runs         = 5;
    float warmup_s       = 1.0f;
    float measure_s      = 1.0f;

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--out") && i + 1 < argc) {
            out_path = argv[++i];
        } else if (!strcmp(argv[i], "--in") && i + 1 < argc) {
            in_path = argv[++i];
        } else if (!strcmp(argv[i], "--gamma") && i + 1 < argc) {
            gamma = (float)atof(argv[++i]);
        } else if (!strcmp(argv[i], "--beta") && i + 1 < argc) {
            beta  = (float)atof(argv[++i]);
        } else if (!strcmp(argv[i], "--sigma") && i + 1 < argc) {
            sigma_override = (float)atof(argv[++i]);
            if (sigma_override < 0.f || sigma_override > 1.f) {
                fprintf(stderr, "error: --sigma must be in [0.0, 1.0]\n");
                return 1;
            }
        } else if (!strcmp(argv[i], "--runs") && i + 1 < argc) {
            n_runs = atoi(argv[++i]);
            if (n_runs < 1) { fprintf(stderr, "error: --runs must be >= 1\n"); return 1; }
        } else if (!strcmp(argv[i], "--warmup-time") && i + 1 < argc) {
            warmup_s = (float)atof(argv[++i]);
        } else if (!strcmp(argv[i], "--measure-time") && i + 1 < argc) {
            measure_s = (float)atof(argv[++i]);
            if (measure_s <= 0.f) { fprintf(stderr, "error: --measure-time must be > 0\n"); return 1; }
        } else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    blaq_profile_t p = {};

    if (in_path) {
        fprintf(stderr, "Loading profile from: %s\n\n", in_path);
        if (!blaq_profile_load_json(&p, in_path)) {
            fprintf(stderr, "error: failed to load profile from '%s'\n", in_path);
            return 1;
        }
        blaq_profile_print(&p);
        return 0;
    }

    // Measurement mode
    const double total_s = (double)n_runs * (double)measure_s + (double)warmup_s;
    fprintf(stderr, "BLAQ Hardware Profiler\n");
    fprintf(stderr, "  Passes      : %d × %.1f s  (+ %.1f s warmup per direction)\n",
            n_runs, (double)measure_s, (double)warmup_s);
    fprintf(stderr, "  Est. time   : ~%.0f s total\n", total_s * 2);
    fprintf(stderr, "  gamma=%.2f  beta=%.2f\n\n", (double)gamma, (double)beta);

    fprintf(stderr, "Step 1/4  detecting cache-line size ...\n");
    fprintf(stderr, "Step 2/4  measuring peak bandwidth ...\n");
    fprintf(stderr, "Step 3/4  counting shared-bus processors ...\n");
    if (sigma_override >= 0.f) {
        fprintf(stderr, "Step 4/4  contention benchmark skipped"
                " — using sigma=%.3f (user-supplied)\n\n", (double)sigma_override);
    } else {
        fprintf(stderr, "Step 4/4  measuring contention bandwidth ...\n\n");
    }

    if (!blaq_profile_measure(&p, gamma, beta, sigma_override, n_runs, warmup_s, measure_s)) {
        fprintf(stderr, "error: profiling failed\n");
        return 1;
    }

    blaq_profile_print(&p);

    // Determine output path: use user-supplied path or auto-generate
    char suggested[256];
    blaq_profile_suggest_filename(&p, suggested, sizeof(suggested));

    const char * write_path = out_path ? out_path : suggested;

    if (!blaq_profile_save_json(&p, write_path)) {
        fprintf(stderr, "error: could not write profile to '%s'\n", write_path);
        return 1;
    }
    fprintf(stderr, "Profile saved to: %s\n", write_path);

    if (!out_path) {
        fprintf(stderr, "  (auto-named — use --out <path> to specify a custom name)\n");
    }

    // Recommendation
    fprintf(stderr, "\nRecommendation:\n");
    if (p.cache_line_bytes >= 128) {
        fprintf(stderr, "  Cache line = %u B  →  use BLAQ_Q4_256 (256-weight blocks)\n",
                p.cache_line_bytes);
    } else {
        fprintf(stderr, "  Cache line = %u B  →  use BLAQ_Q4_128 (128-weight blocks)\n",
                p.cache_line_bytes);
    }
    fprintf(stderr, "  llama-quantize --blaq-profile %s <model> <output> BLAQ_Q4_%s\n\n",
            write_path, p.cache_line_bytes >= 128 ? "256" : "128");

    if (p.peak_bw_cv > 0.05f || p.obs_bw_cv > 0.05f) {
        fprintf(stderr,
            "NOTE: High CV detected (peak CV=%.3f, obs CV=%.3f).\n"
            "      Consider --runs %d or --measure-time 2.0 for more stable results.\n\n",
            (double)p.peak_bw_cv, (double)p.obs_bw_cv, n_runs * 2);
    }

    return 0;
}
