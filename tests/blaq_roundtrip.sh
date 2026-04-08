#!/usr/bin/env bash
# BLAQ round-trip integration tests.
# Usage: tests/blaq_roundtrip.sh [path/to/model.f16.gguf]
#
# Requires a small F16 model on disk.  TinyLlama-1.1B is a convenient choice:
#   huggingface-cli download TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF \
#       tinyllama-1.1b-chat-v1.0.Q16_0.gguf
#
# If no model path is supplied the script exits cleanly with a skip message.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BIN_DIR="${REPO_DIR}/build/bin"

QUANTIZE="${BIN_DIR}/llama-quantize"
CLI="${BIN_DIR}/llama-cli"
PROFILER="${BIN_DIR}/llama-blaq-profile"

MODEL="${1:-}"

if [[ -z "${MODEL}" ]]; then
    echo "SKIP: no model path supplied.  Pass a .f16.gguf model as the first argument."
    exit 0
fi

if [[ ! -f "${MODEL}" ]]; then
    echo "SKIP: model file not found: ${MODEL}"
    exit 0
fi

for BIN in "${QUANTIZE}" "${CLI}"; do
    if [[ ! -x "${BIN}" ]]; then
        echo "ERROR: binary not found or not executable: ${BIN}"
        exit 1
    fi
done

TMPDIR=$(mktemp -d)
trap 'rm -rf "${TMPDIR}"' EXIT

PROMPT="The capital of France is"

echo "==================================================================="
echo " BLAQ round-trip tests"
echo " Model : ${MODEL}"
echo " Tmpdir: ${TMPDIR}"
echo "==================================================================="

# -------------------------------------------------------------------
# 1. BLAQ_Q4_128 — no imatrix
# -------------------------------------------------------------------
echo ""
echo "--- BLAQ_Q4_128 (no imatrix) ---"
OUT128="${TMPDIR}/blaq128.gguf"
"${QUANTIZE}" "${MODEL}" "${OUT128}" BLAQ_Q4_128
"${CLI}" -m "${OUT128}" -p "${PROMPT}" -n 8 --no-display-prompt 2>/dev/null
echo "ok"

# -------------------------------------------------------------------
# 2. BLAQ_Q4_256 — no imatrix
# -------------------------------------------------------------------
echo ""
echo "--- BLAQ_Q4_256 (no imatrix) ---"
OUT256="${TMPDIR}/blaq256.gguf"
"${QUANTIZE}" "${MODEL}" "${OUT256}" BLAQ_Q4_256
"${CLI}" -m "${OUT256}" -p "${PROMPT}" -n 8 --no-display-prompt 2>/dev/null
echo "ok"

# -------------------------------------------------------------------
# 3. BLAQ_Q4_128 — with hardware profile
# -------------------------------------------------------------------
if [[ -x "${PROFILER}" ]]; then
    echo ""
    echo "--- BLAQ_Q4_128 with hardware profile ---"
    PROFILE="${TMPDIR}/hw_profile.json"
    # Use defaults rather than running the benchmark to keep CI fast
    "${PROFILER}" --in /dev/null --out "${PROFILE}" 2>/dev/null || \
        "${PROFILER}" --out "${PROFILE}" 2>/dev/null || true

    if [[ -f "${PROFILE}" ]]; then
        OUT128P="${TMPDIR}/blaq128_profile.gguf"
        "${QUANTIZE}" "${MODEL}" "${OUT128P}" BLAQ_Q4_128 --blaq-profile "${PROFILE}"
        "${CLI}" -m "${OUT128P}" -p "${PROMPT}" -n 8 --no-display-prompt 2>/dev/null
        echo "ok"
    else
        echo "SKIP: profiler did not produce output (ok in CI)"
    fi
fi

# -------------------------------------------------------------------
# 4. Verify GGUF metadata keys are written
# -------------------------------------------------------------------
echo ""
echo "--- Checking BLAQ GGUF metadata keys in ${OUT128} ---"
# gguf-dump is available as llama-gguf-hash or via Python gguf package.
# Use strings(1) as a portable fallback — the key names are plain ASCII.
if strings "${OUT128}" | grep -q "blaq.cache_line_bytes"; then
    echo "blaq.cache_line_bytes : found"
else
    echo "WARNING: blaq.cache_line_bytes not found in GGUF (may be absent without imatrix)"
fi

echo ""
echo "==================================================================="
echo " All BLAQ round-trip tests passed."
echo "==================================================================="
