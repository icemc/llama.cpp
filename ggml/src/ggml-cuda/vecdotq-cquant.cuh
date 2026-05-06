#pragma once

#include "common.cuh"

// NF4 codebook as int8 (×127) for dp4a integer accumulation.
// nf4_int8[k] = round(NF4_float[k] * 127). Max error ≈ 0.4%.
// Result scaling: actual = d_eff * d8 * sumi_int32 / 127.
static __device__ const int8_t nf4_int8_table[16] = {
    -127, -88, -67, -50, -36, -23, -12,   0,
      10,  20,  31,  43,  56,  71,  92, 127,
};

// VDR (vectors-per-dispatch-register) for MMVQ.
// Q4_C_64:  qk=1024, qi=128, vdr=4  → qi/vdr=32 (one warp per super-block with nwarps=4 warps)
// Q4_C_128: qk=2048, qi=256, vdr=8  → qi/vdr=32 (one warp per super-block; needs nwarps≥2)
#define VDR_Q4_C_64_Q8_1_MMVQ  4
#define VDR_Q4_C_128_Q8_1_MMVQ 8

// ──────────────────────────────────────────────────────────────────────
// Q4_C_64 vec_dot  (integer dp4a path via __byte_perm re-interleaving)
//
// kbx  = super-block absolute index (strides by sizeof(block_q4_C_64)).
// iqs  = int32-offset within the nibble region; VDR=4 → always a
//        multiple of 4, selecting one complete group per call.
// bq8_1 = pointer to the first Q8_1 block of this super-block in y[].
//
// Storage format: Q4_C uses Q4_0-style consecutive pairs:
//   qs[g*16 + j/2] = nf4_idx[g*32 + j] | (nf4_idx[g*32 + j+1] << 4)
// get_int_from_table_16 returns even-nibble weights in .x and
// odd-nibble weights in .y.  __byte_perm reshuffles them into
// sequential order {w[0],w[1],w[2],w[3]} for dp4a with sequential acts.
// ──────────────────────────────────────────────────────────────────────
static __device__ __forceinline__ float vec_dot_q4_C_64_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1,
    const int & kbx, const int & iqs)
{
    const block_q4_C_64 * blk = (const block_q4_C_64 *) vbq + kbx;

    // Group index: 4 int32s of nibbles per group (16 bytes, 32 nibbles).
    const int g = iqs >> 2;   // iqs / 4

    const float d_eff = __half2float(blk->d[g]) * (1.0f/127.0f);
    const float d8    = __low2float(bq8_1[g].ds);

    // Nibble bytes for group g: 16 bytes, read as 4 int32s.
    const uint8_t * g_qs = blk->qs + g * 16;
    // Q8_1 activations for this group: 32 int8s = 8 int32s.
    const int * acts = (const int *) bq8_1[g].qs;

    int sumi = 0;
#pragma unroll
    for (int b = 0; b < 4; ++b) {
        // Read 4 nibble bytes (8 weights) at position b within the group.
        const int v32 = get_int_b2(g_qs, b);

        // Table lookup: even nibbles → v.x, odd nibbles → v.y
        const int2 v = get_int_from_table_16(v32, nf4_int8_table);

        // Reshuffle from interleaved to sequential weight order using __byte_perm:
        //   seq_lo = {w[0],w[1],w[2],w[3]}, seq_hi = {w[4],w[5],w[6],w[7]}
        const int seq_lo = __byte_perm(v.x, v.y, 0x5140);
        const int seq_hi = __byte_perm(v.x, v.y, 0x7362);

        // dp4a: seq_lo × acts[b*8..b*8+3], seq_hi × acts[b*8+4..b*8+7]
        sumi = ggml_cuda_dp4a(seq_lo, acts[b * 2    ], sumi);
        sumi = ggml_cuda_dp4a(seq_hi, acts[b * 2 + 1], sumi);
    }

    return d_eff * d8 * (float) sumi;
}

// ──────────────────────────────────────────────────────────────────────
// Q4_C_128 vec_dot
// VDR=8 → iqs is a multiple of 8, covering two consecutive groups.
// Groups: g0 = iqs/4, g1 = g0+1.
// ──────────────────────────────────────────────────────────────────────
static __device__ __forceinline__ float vec_dot_q4_C_128_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1,
    const int & kbx, const int & iqs)
{
    const block_q4_C_128 * blk = (const block_q4_C_128 *) vbq + kbx;

    const int g0 = iqs >> 2;   // first group  (iqs / 4)
    // g1 = g0 + 1  (second group)

    float total = 0.0f;

#pragma unroll
    for (int gi = 0; gi < 2; ++gi) {
        const int g = g0 + gi;

        const float d_eff = __half2float(blk->d[g]) * (1.0f/127.0f);
        const float d8    = __low2float(bq8_1[g].ds);

        const uint8_t * g_qs = blk->qs + g * 16;
        const int * acts = (const int *) bq8_1[g].qs;

        int sumi = 0;
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            const int v32 = get_int_b2(g_qs, b);
            const int2 v  = get_int_from_table_16(v32, nf4_int8_table);
            const int seq_lo = __byte_perm(v.x, v.y, 0x5140);
            const int seq_hi = __byte_perm(v.x, v.y, 0x7362);
            sumi = ggml_cuda_dp4a(seq_lo, acts[b * 2    ], sumi);
            sumi = ggml_cuda_dp4a(seq_hi, acts[b * 2 + 1], sumi);
        }

        total += d_eff * d8 * (float) sumi;
    }

    return total;
}
