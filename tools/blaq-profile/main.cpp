//
// llama-blaq-profile — BLAQ Hardware Profiler CLI
//
// Runs Algorithm 1 (BLAQ-Probe) on the current machine and writes a JSON
// profile that can be passed to llama-quantize via --blaq-profile.
//
// Usage:
//   llama-blaq-profile [options]
//
//   --out  <path>   output JSON file  (default: blaq_hw_profile.json)
//   --in   <path>   load existing JSON and print it (skips benchmarking)
//   --gamma <f>     bandwidth penalty scale  (default: 1.0)
//   --beta  <f>     contention penalty scale (default: 1.0)
//   --help          show this help
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
        "  --out  <path>   write profile JSON to this file (default: blaq_hw_profile.json)\n"
        "  --in   <path>   load and print an existing profile JSON (no benchmarking)\n"
        "  --gamma <f>     bandwidth penalty scale  (default: 1.0)\n"
        "  --beta  <f>     contention penalty scale (default: 1.0)\n"
        "  --help          show this message\n"
        "\n"
        "Example (measure current machine):\n"
        "  %s --out my_machine.json\n"
        "\n"
        "Example (inspect saved profile):\n"
        "  %s --in my_machine.json\n"
        "\n",
        prog, prog, prog);
}

int main(int argc, char ** argv) {
    const char * out_path = "blaq_hw_profile.json";
    const char * in_path  = nullptr;
    float gamma = 1.0f;
    float beta  = 1.0f;

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--out") && i + 1 < argc) {
            out_path = argv[++i];
        } else if (!strcmp(argv[i], "--in") && i + 1 < argc) {
            in_path = argv[++i];
        } else if (!strcmp(argv[i], "--gamma") && i + 1 < argc) {
            gamma = (float)atof(argv[++i]);
        } else if (!strcmp(argv[i], "--beta") && i + 1 < argc) {
            beta  = (float)atof(argv[++i]);
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
        // Load-only mode: inspect an existing profile
        fprintf(stderr, "Loading profile from: %s\n", in_path);
        if (!blaq_profile_load_json(&p, in_path)) {
            fprintf(stderr, "error: failed to load profile from '%s'\n", in_path);
            return 1;
        }
        blaq_profile_print(&p);
        return 0;
    }

    // Measurement mode
    fprintf(stderr, "BLAQ Hardware Profiler\n");
    fprintf(stderr, "  gamma = %.2f  beta = %.2f\n\n", (double)gamma, (double)beta);
    fprintf(stderr, "Step 1/4  detecting cache-line size ...\n");
    fprintf(stderr, "Step 2/4  measuring peak bandwidth (single agent, ~0.5 s) ...\n");
    fprintf(stderr, "Step 3/4  counting shared-bus processors ...\n");
    fprintf(stderr, "Step 4/4  measuring contention bandwidth (~0.5 s) ...\n\n");

    if (!blaq_profile_measure(&p, gamma, beta)) {
        fprintf(stderr, "error: profiling failed\n");
        return 1;
    }

    blaq_profile_print(&p);

    if (!blaq_profile_save_json(&p, out_path)) {
        fprintf(stderr, "error: could not write profile to '%s'\n", out_path);
        return 1;
    }
    fprintf(stderr, "Profile saved to: %s\n", out_path);

    // Emit a recommendation
    fprintf(stderr, "\nRecommendation:\n");
    if (p.cache_line_bytes >= 128) {
        fprintf(stderr, "  Cache line = %u B -> use BLAQ_Q4_256 (256-weight blocks)\n",
                p.cache_line_bytes);
        fprintf(stderr, "  llama-quantize <model> <output> BLAQ_Q4_256 --blaq-profile %s\n\n",
                out_path);
    } else {
        fprintf(stderr, "  Cache line = %u B -> use BLAQ_Q4_128 (128-weight blocks)\n",
                p.cache_line_bytes);
        fprintf(stderr, "  llama-quantize <model> <output> BLAQ_Q4_128 --blaq-profile %s\n\n",
                out_path);
    }

    return 0;
}
