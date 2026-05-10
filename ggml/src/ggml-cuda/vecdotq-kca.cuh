#pragma once

#include "common.cuh"

// VDR (vectors-per-dispatch-register) for Q4_KCA MMVQ.
//
// Q4_KCA_64: qk=1024, qi=128, vdr=4  → qi/vdr=32 threads per block (1 block/warp)
//   Each vec_dot handles HALF a sub-block pair (4 int32s = 16 bytes).
//   The two halves (half=0 and half=1) together cover one sub-block pair.
//   Only the half=0 thread applies the full Dmin correction via Q8_1 ds.y.
//
// Q4_KCA_128: qk=2048, qi=256, vdr=4  → qi/vdr=64 threads per block (2 warps/block)
//   Each vec_dot handles HALF a sub-block pair (4 int32s = 16 bytes).
//   Only the half=0 thread applies the full Dmin correction via Q8_1 ds.y.
//   With 64 threads/block, half=0 and half=1 threads each fill exactly one warp
//   → no within-warp divergence for the Dmin branch.
#define VDR_Q4_KCA_64_Q8_1_MMVQ  4
#define VDR_Q4_KCA_128_Q8_1_MMVQ 4

// Decode one 6-bit scale and 6-bit min from Q4_K / Q4_KCA 12-byte packed scale block.
// Matches get_scale_min_k4() in ggml-quants.c exactly.
static __device__ __forceinline__ void kca_get_scale_min(
    int j, const uint8_t * __restrict__ q12, uint8_t & sc, uint8_t & m)
{
    if (j < 4) {
        sc = q12[j]   & 63;
        m  = q12[j+4] & 63;
    } else {
        sc = (q12[j+4] & 0x0F) | ((q12[j-4] >> 6) << 4);
        m  = (q12[j+4] >>   4) | ((q12[j  ] >> 6) << 4);
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Q4_KCA_64 vec_dot  (K-quant two-level scales + dp4a)  VDR=4 → 32 threads/block
//
// Block struct layout (576 B = 9×64B):
//   d[4]       FP16 super-scale per group  (g = 0..3)
//   dmin[4]    FP16 super-min  per group
//   scales[48] 4×12B: 8×(6-bit sc + 6-bit m) per group
//   qs[512]    nibbles: 4 groups × 4 sub-block-pairs × 32 bytes
//
// Nibble layout within one group (128 bytes = 32 int32s = 4 sp chunks × 8 int32s):
//   qs[g*128 + sp*32 + b] = lo_nibble(sb0_weight_b) | hi_nibble(sb1_weight_b)
//   where sp = sub-block pair (0..3), sb0 = 2*sp, sb1 = 2*sp+1
//
// VDR=4: iqs is a multiple of 4, selecting 4 int32s = HALF a sub-block pair.
//   g    = iqs / 32               — group (0..3)
//   pos  = iqs % 32               — position within group
//   sp   = pos / 8                — sub-block pair (0..3)
//   half = (pos % 8) / 4          — 0 = first 16 bytes, 1 = second 16 bytes
//
// Dmin handling: each thread computes its own partial activation sum on-the-fly
// via dp4a(0x01010101, acts, …) and applies its share of the Dmin correction.
// After the warp reduction outside this function the partial sums combine into
// the correct per-sub-block totals. This pattern (taken from Q4_K's mmvq impl)
// avoids the within-warp branch on `half` and the redundant ds.y memory loads —
// purely additional dp4a work, which is single-cycle on Ampere/Ada SMs.
// ──────────────────────────────────────────────────────────────────────────────
static __device__ __forceinline__ float vec_dot_q4_KCA_64_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1,
    const int & kbx, const int & iqs)
{
    const block_q4_KCA_64 * blk = (const block_q4_KCA_64 *) vbq + kbx;

    const int g    = iqs >> 5;           // iqs / 32 → group (0..3)
    const int pos  = iqs & 31;           // position within group
    const int sp   = pos >> 3;           // pos / 8 → sub-block pair (0..3)
    const int half = (pos & 7) >> 2;     // (pos % 8) / 4 → 0 or 1

    const int sb0 = sp * 2;
    const int sb1 = sp * 2 + 1;

    const float D    = __half2float(blk->d[g]);
    const float Dmin = __half2float(blk->dmin[g]);

    const uint8_t * scales_12 = blk->scales + g * 12;
    uint8_t sc0, m0, sc1, m1;
    kca_get_scale_min(sb0, scales_12, sc0, m0);
    kca_get_scale_min(sb1, scales_12, sc1, m1);

    // 4 int32s = 16 bytes = half of the 32-byte sub-block pair chunk
    const int * v = (const int *)(blk->qs + iqs * 4);

    // Activation int32s for this half (4 of the 8 int32s per Q8_1 block)
    const int * acts0 = (const int *)bq8_1[g * 8 + sb0].qs + half * 4;
    const int * acts1 = (const int *)bq8_1[g * 8 + sb1].qs + half * 4;

    const float d8_0 = __low2float(bq8_1[g * 8 + sb0].ds);
    const float d8_1 = __low2float(bq8_1[g * 8 + sb1].ds);

    int sumi0 = 0, sumi1 = 0;     // dot products: weights · activations
    int suma0 = 0, suma1 = 0;     // running sums: 1 · activations  (Dmin term)
#pragma unroll
    for (int b = 0; b < 4; ++b) {
        const int lo = (v[b]     ) & 0x0F0F0F0F;  // 4 lo nibbles (sb0 weights)
        const int hi = (v[b] >> 4) & 0x0F0F0F0F;  // 4 hi nibbles (sb1 weights)
        sumi0 = ggml_cuda_dp4a(lo, acts0[b], sumi0);
        sumi1 = ggml_cuda_dp4a(hi, acts1[b], sumi1);
        // dp4a(0x01010101, x, acc) accumulates the four int8 lanes of x.
        // Per-thread partial activation sum; warp-reduced by the caller.
        suma0 = ggml_cuda_dp4a(0x01010101, acts0[b], suma0);
        suma1 = ggml_cuda_dp4a(0x01010101, acts1[b], suma1);
    }

    return D    * (d8_0 * (float)sc0 * (float)sumi0 + d8_1 * (float)sc1 * (float)sumi1)
         - Dmin * (d8_0 * (float)m0  * (float)suma0 + d8_1 * (float)m1  * (float)suma1);
}

// ──────────────────────────────────────────────────────────────────────────────
// Q4_KCA_128 vec_dot  VDR=4 → qi/vdr=64 threads per block (2 warps/block)
// Same algorithm as Q4_KCA_64 with g ∈ {0..7} (8 groups). Uses the same
// branch-free Dmin scheme: every thread accumulates its share of the activation
// sum on-the-fly; warp reduction stitches the partial sums into the per-sub-
// block totals. Eliminates the inter-warp work imbalance the half==0 branch
// previously caused (warp 1 had nothing to do for the Dmin term).
// ──────────────────────────────────────────────────────────────────────────────
static __device__ __forceinline__ float vec_dot_q4_KCA_128_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1,
    const int & kbx, const int & iqs)
{
    const block_q4_KCA_128 * blk = (const block_q4_KCA_128 *) vbq + kbx;

    const int g    = iqs >> 5;           // iqs / 32 → group (0..7)
    const int pos  = iqs & 31;           // position within group
    const int sp   = pos >> 3;           // pos / 8 → sub-block pair (0..3)
    const int half = (pos & 7) >> 2;     // (pos % 8) / 4 → 0 or 1

    const int sb0 = sp * 2;
    const int sb1 = sp * 2 + 1;

    const float D    = __half2float(blk->d[g]);
    const float Dmin = __half2float(blk->dmin[g]);

    const uint8_t * scales_12 = blk->scales + g * 12;
    uint8_t sc0, m0, sc1, m1;
    kca_get_scale_min(sb0, scales_12, sc0, m0);
    kca_get_scale_min(sb1, scales_12, sc1, m1);

    const int * v = (const int *)(blk->qs + iqs * 4);

    const int * acts0 = (const int *)bq8_1[g * 8 + sb0].qs + half * 4;
    const int * acts1 = (const int *)bq8_1[g * 8 + sb1].qs + half * 4;

    const float d8_0 = __low2float(bq8_1[g * 8 + sb0].ds);
    const float d8_1 = __low2float(bq8_1[g * 8 + sb1].ds);

    int sumi0 = 0, sumi1 = 0;
    int suma0 = 0, suma1 = 0;
#pragma unroll
    for (int b = 0; b < 4; ++b) {
        const int lo = (v[b]     ) & 0x0F0F0F0F;
        const int hi = (v[b] >> 4) & 0x0F0F0F0F;
        sumi0 = ggml_cuda_dp4a(lo, acts0[b], sumi0);
        sumi1 = ggml_cuda_dp4a(hi, acts1[b], sumi1);
        suma0 = ggml_cuda_dp4a(0x01010101, acts0[b], suma0);
        suma1 = ggml_cuda_dp4a(0x01010101, acts1[b], suma1);
    }

    return D    * (d8_0 * (float)sc0 * (float)sumi0 + d8_1 * (float)sc1 * (float)sumi1)
         - Dmin * (d8_0 * (float)m0  * (float)suma0 + d8_1 * (float)m1  * (float)suma1);
}
