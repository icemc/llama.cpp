# BLAQ — Bandwidth- and Layout-Aware Quantization

Build and run guide for BLAQ quantization on **NVIDIA DGX Spark (GB10)** and **Apple Silicon**.

BLAQ adds two new 4-bit quantization types to llama.cpp that align weight groups to hardware
cache-line boundaries, reducing memory-bus contention on unified-memory (UMA) devices:

| Type | Block size | Target cache line | Bytes/block | Primary device |
|------|-----------|-------------------|-------------|----------------|
| `BLAQ_Q4_128` | 128 weights | 64 B (Neoverse V3, x86-64) | 66 B | DGX Spark / GB10 |
| `BLAQ_Q4_256` | 256 weights | 128 B (Apple Avalon) | 130 B | Apple M-series |

> **GPU inference status:** CPU kernels are fully implemented. CUDA (Phase 8) and Metal (Phase 7)
> GPU kernels are not yet implemented. Use `-ngl 0` for all BLAQ inference until those phases land.

---

## Contents

1. [DGX Spark — requirements and build](#1-dgx-spark--requirements-and-build)
2. [Apple Silicon — requirements and build](#2-apple-silicon--requirements-and-build)
3. [Get a model](#3-get-a-model)
4. [Profile your hardware](#4-profile-your-hardware)
5. [Quantize a model](#5-quantize-a-model)
6. [Run inference](#6-run-inference)
7. [Measure perplexity](#7-measure-perplexity)
8. [GPU inference roadmap](#8-gpu-inference-roadmap)

---

## 1. DGX Spark — requirements and build

| Requirement | Details |
|-------------|---------|
| OS | Ubuntu 24.04 LTS (ships pre-installed) |
| CPU | ARM Neoverse V3, 64-byte cache lines → use `BLAQ_Q4_128` |
| Memory | 128 GB LPDDR5X unified memory (CPU and GPU share the same pool) |
| GPU | NVIDIA Blackwell — BLAQ CUDA kernels **not yet implemented** |
| CMake | ≥ 3.17 |
| Compiler | GCC 12+ or Clang 14+ |

### Install dependencies

```bash
sudo apt update && sudo apt install -y \
    git cmake ninja-build build-essential python3-pip
pip3 install huggingface_hub
```

### Clone the branch

```bash
git clone https://github.com/<your-fork>/llama.cpp.git
cd llama.cpp
git checkout feat/blaq-quantization
```

### Build — CPU only (recommended for BLAQ)

```bash
cmake -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=OFF
ninja -C build \
    llama-quantize llama-cli llama-blaq-profile llama-perplexity
```

### Build — with CUDA (for non-BLAQ layers)

If you also want GPU acceleration for standard quantizations (Q4\_K, Q8\_0, etc.),
enable CUDA. Note that BLAQ-quantized models must still be run with `-ngl 0`.

```bash
cmake -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=native
ninja -C build \
    llama-quantize llama-cli llama-blaq-profile llama-perplexity
```

---

## 2. Apple Silicon — requirements and build

| Requirement | Details |
|-------------|---------|
| OS | macOS 13 Ventura or newer |
| CPU | Apple M-series (Avalon cores), 128-byte cache lines → use `BLAQ_Q4_256` |
| Memory | Unified memory — 16 GB minimum, 32 GB+ for 7B+ models |
| GPU | Apple GPU via Metal — BLAQ Metal kernels **not yet implemented** |
| Xcode CLT | ≥ 14 (`xcode-select --install`) |
| Homebrew | for cmake/ninja |

### Install dependencies

```bash
xcode-select --install
brew install cmake ninja python3
pip3 install huggingface_hub
```

### Clone the branch

```bash
git clone https://github.com/<your-fork>/llama.cpp.git
cd llama.cpp
git checkout feat/blaq-quantization
```

### Build — with Metal (Metal used automatically for non-BLAQ types)

```bash
cmake -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_METAL=ON
ninja -C build \
    llama-quantize llama-cli llama-blaq-profile llama-perplexity
```

---

## 3. Get a model

Download a pre-converted F16 GGUF from Hugging Face.
Llama-3.2-1B is convenient for quick experiments; use a larger model for thesis results.

```bash
# Small (1B) — fast, good for smoke tests
huggingface-cli download \
    bartowski/Llama-3.2-1B-Instruct-GGUF \
    Llama-3.2-1B-Instruct-f16.gguf \
    --local-dir ./models

# Medium (8B) — requires ~16 GB RAM
huggingface-cli download \
    bartowski/Meta-Llama-3.1-8B-Instruct-GGUF \
    Meta-Llama-3.1-8B-Instruct-f16.gguf \
    --local-dir ./models
```

Set a variable for convenience:

```bash
MODEL=./models/Llama-3.2-1B-Instruct-f16.gguf
```

---

## 4. Profile your hardware

Run the profiler **on the target inference device**.
The resulting JSON captures cache-line size, peak bandwidth, and contention ratio.
You can then copy it to any machine and pass it to `llama-quantize`.

```bash
./build/bin/llama-blaq-profile --out blaq_hw_profile.json
```

The profiler runs a ~1-second STREAM benchmark — do not run other memory-intensive
workloads alongside it.

**Example output on DGX Spark (Neoverse V3):**
```
=== BLAQ Hardware Profile ===
  Cache-line size       : 64 bytes
  Peak BW               : ~200.0 GB/s
  Contention BW         : ~160.0 GB/s
  Contention ratio (σ)  : 0.200
  Shared-bus procs      : 2
  Aligned group (4-bit) : 128 weights   ← use BLAQ_Q4_128
  lambda_bw             : 5.000e-12
  lambda_mem            : 0.200
==============================
```

**Example output on Apple M3/M4 (Avalon cores):**
```
=== BLAQ Hardware Profile ===
  Cache-line size       : 128 bytes
  Peak BW               : ~300.0 GB/s
  Aligned group (4-bit) : 256 weights   ← use BLAQ_Q4_256
==============================
```

To inspect an existing profile without re-running the benchmark:

```bash
./build/bin/llama-blaq-profile --in blaq_hw_profile.json
```

---

## 5. Quantize a model

### Choose the right type

| Device | Cache line | Recommended type |
|--------|-----------|-----------------|
| DGX Spark / GB10 | 64 B | `BLAQ_Q4_128` |
| Apple M-series | 128 B | `BLAQ_Q4_256` |
| Unknown / generic | 64 B | `BLAQ_Q4_128` (safe default) |

### Dry-run first (estimate output size)

```bash
./build/bin/llama-quantize \
    "$MODEL" /dev/null \
    BLAQ_Q4_128 \
    --dry-run
```

### BLAQ-lite (no imatrix, fast)

```bash
./build/bin/llama-quantize \
    "$MODEL" \
    ./models/model-blaq128.gguf \
    BLAQ_Q4_128 \
    --blaq-profile blaq_hw_profile.json
```

### BLAQ-full (with imatrix, best quality)

Collect calibration statistics once, then reuse across quantization runs.

```bash
# Step 1 — collect imatrix (~10 min for 1B, ~45 min for 8B)
curl -L https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.txt \
    -o wikitext2.txt

./build/bin/llama-imatrix \
    -m "$MODEL" \
    -f wikitext2.txt \
    -o imatrix.gguf \
    --chunks 128

# Step 2 — quantize
./build/bin/llama-quantize \
    "$MODEL" \
    ./models/model-blaq128-im.gguf \
    BLAQ_Q4_128 \
    --imatrix imatrix.gguf \
    --blaq-profile blaq_hw_profile.json
```

### Omitting `--blaq-profile`

If no profile is supplied, safe defaults are used automatically
(64-byte cache line, 100 GB/s peak bandwidth, σ = 0.20).
This is suitable for quick tests but the type selection may not be optimal for your hardware.

```bash
./build/bin/llama-quantize \
    "$MODEL" \
    ./models/model-blaq128-default.gguf \
    BLAQ_Q4_128
```

---

## 6. Run inference

### CPU inference — DGX Spark and Apple Silicon

`-ngl 0` is **required** for BLAQ-quantized models.
GPU kernels (CUDA / Metal) are not yet implemented.

```bash
./build/bin/llama-cli \
    -m ./models/model-blaq128.gguf \
    -ngl 0 \
    -t $(nproc) \
    -p "The main benefit of cache-line-aligned quantization is" \
    -n 100
```

On Apple Silicon replace `$(nproc)` with `$(sysctl -n hw.physicalcpu)`.

### GPU inference — non-BLAQ baseline for comparison

Standard quantizations have CUDA / Metal kernels and can use the GPU normally.
Useful for benchmarking BLAQ CPU throughput against Q4\_K GPU throughput.

```bash
# Quantize to Q4_K_M
./build/bin/llama-quantize \
    "$MODEL" \
    ./models/model-q4km.gguf \
    Q4_K_M

# DGX Spark — CUDA
./build/bin/llama-cli \
    -m ./models/model-q4km.gguf \
    -ngl 99 \
    -p "The main benefit of cache-line-aligned quantization is" \
    -n 100

# Apple Silicon — Metal
./build/bin/llama-cli \
    -m ./models/model-q4km.gguf \
    -ngl 99 \
    -p "The main benefit of cache-line-aligned quantization is" \
    -n 100
```

### Interactive chat

```bash
./build/bin/llama-cli \
    -m ./models/model-blaq128.gguf \
    -ngl 0 \
    -t $(nproc) \
    --conversation \
    --chat-template llama3 \
    -sys "You are a helpful assistant."
```

---

## 7. Measure perplexity

Perplexity on WikiText-2 is the primary quality metric for thesis comparisons.

```bash
# Download test set if not already present
curl -L https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.txt \
    -o wikitext2.txt

# BLAQ-lite (no imatrix)
./build/bin/llama-perplexity \
    -m ./models/model-blaq128.gguf \
    -f wikitext2.txt -ngl 0

# BLAQ-full (with imatrix)
./build/bin/llama-perplexity \
    -m ./models/model-blaq128-im.gguf \
    -f wikitext2.txt -ngl 0

# Q4_K_M baseline (GPU)
./build/bin/llama-perplexity \
    -m ./models/model-q4km.gguf \
    -f wikitext2.txt -ngl 99

# F16 reference
./build/bin/llama-perplexity \
    -m "$MODEL" \
    -f wikitext2.txt -ngl 99
```

Expected results for Llama-3.2-1B on WikiText-2:

| Quantization | PPL (approx) | Notes |
|-------------|-------------|-------|
| F16 | ~7.5 | reference |
| Q4\_K\_M | ~7.9 | imatrix-optimised, GPU |
| BLAQ\_Q4\_128-full | ~7.9 | target: match Q4\_K\_M |
| BLAQ\_Q4\_128-lite | ~8.1 | absmax scales, no imatrix |

---

## 8. GPU inference roadmap

Full UMA benefit — the Blackwell / Apple GPU reading cache-line-aligned weight blocks
directly from the shared memory pool without a separate copy — requires kernel
implementations in the respective GPU backends.

| Platform | Status | Files to implement |
|----------|--------|--------------------|
| Apple Silicon (Metal) | Not implemented — Phase 7 | `ggml-metal.metal` dequant kernels, `ggml-metal-device.cpp` pipeline entries |
| DGX Spark (CUDA / Blackwell) | Not implemented — Phase 8 | `vecdotq.cuh`, `mmvq.cu`, `mmq.cu`, `ggml-cuda.cu` |
| CPU (all platforms) | **Fully working** | — |

Until Phase 7 / 8 land, always run BLAQ models with `-ngl 0`.
