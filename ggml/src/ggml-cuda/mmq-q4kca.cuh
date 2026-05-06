#pragma once

// mmq-q4kca.cuh — Standalone GEMM kernel for Q4_KCA prompt processing (PP)
//
// Structure mirrors mmq-q4c.cuh but uses K-quant two-level scale decoding
// instead of NF4 codebook lookup.
//
// K-quant formula per sub-block sb:
//   weight = D_g * sc_sb * unsigned_nibble - Dmin_g * m_sb
//   dot with activations:
//     = D_g*sc_sb * Σ(nibble*act) - Dmin_g*m_sb * Σ(act)
//
// K-groups granularity: QK_KCA_SB = 32 weights per Q8_1 block.
//
// Design (BLOCK_M=32, BLOCK_N=32, NWARPS=4, RPT=8):
//   Grid:  (ceil(ne01/32), ceil(ne11/32))
//   Block: dim3(32, 4) = 128 threads
//   Each thread: 8 weight rows × 1 token col → 8 accumulators
//
//   Shared memory per sub-block:
//     smem_W[32][8]  — nibbles (0x0F0F0F0F masked) per weight row
//     smem_Ws[32]    — D * sc per weight row  (positive scale)
//     smem_Wm[32]    — Dmin * m per weight row (min bias)
//     smem_X[32][9]  — Q8_1 int8 per token col (9 int32s, +1 bank-conflict pad)
//     smem_Xs[32]    — Q8_1 scale d8 per token col

#include "common.cuh"
#include "quantize.cuh"
#include "mmq-q4c.cuh"    // for quantize_q8_1_group_major
#include "vecdotq-kca.cuh"

static constexpr int MMQ_KCA_BLOCK_M = 32;
static constexpr int MMQ_KCA_BLOCK_N = 32;
static constexpr int MMQ_KCA_NWARPS  = 4;
static constexpr int MMQ_KCA_RPT     = MMQ_KCA_BLOCK_M / MMQ_KCA_NWARPS; // 8 rows/thread

// ── GEMM kernel ───────────────────────────────────────────────────────────────
template <ggml_type type>
static __global__ void __launch_bounds__(MMQ_KCA_BLOCK_N * MMQ_KCA_NWARPS, 2)
mul_mat_q4_KCA_f32_kernel(
        const char  * __restrict__ src0_d,     // quantized weight data
        const char  * __restrict__ src1_q8_d,  // Q8_1 activations, group-major layout
        float       * __restrict__ dst,
        const int ne01,                    // M: output features (weight rows)
        const int ne11,                    // N: tokens (original, before pad)
        const int K_groups,                // ne00 / QK_KCA_SB
        const int64_t nb01,                // stride in bytes between weight rows in src0
        const int64_t stride_dst,          // stride in floats between dst columns
        const int ne11_pad                 // padded token count (stride in group-major Q8_1)
) {
    static_assert(type == GGML_TYPE_Q4_KCA_64 || type == GGML_TYPE_Q4_KCA_128,
                  "mul_mat_q4_KCA_f32_kernel: unsupported type");

    // Compile-time constants for the two variants
    constexpr int SB_SIZE    = (type == GGML_TYPE_Q4_KCA_64)
                                ? (int)sizeof(block_q4_KCA_64)
                                : (int)sizeof(block_q4_KCA_128);
    constexpr int QK_SB      = (type == GGML_TYPE_Q4_KCA_64) ? QK_KCA_64  : QK_KCA_128;
    constexpr int N_GROUPS   = (type == GGML_TYPE_Q4_KCA_64) ? N_GROUPS_KCA_64 : N_GROUPS_KCA_128;
    constexpr int SB_PER_ROW = QK_SB / QK_KCA_SB;  // sub-blocks per super-block (32 or 64)
    constexpr int G_BYTES    = QK_KCA_G / 2;         // nibble bytes per group (128 or 256)

    const int warp_id = threadIdx.y;
    const int lane_id = threadIdx.x;   // token col within tile
    const int tid     = warp_id * WARP_SIZE + lane_id;

    const int tile_row = blockIdx.x * MMQ_KCA_BLOCK_M;
    const int tile_col = blockIdx.y * MMQ_KCA_BLOCK_N;

    __shared__ int32_t smem_W[MMQ_KCA_BLOCK_M][8];   // nibbles per weight row
    __shared__ float   smem_Ws[MMQ_KCA_BLOCK_M];     // D * sc  (scale)
    __shared__ float   smem_Wm[MMQ_KCA_BLOCK_M];     // Dmin * m (bias)
    __shared__ int32_t smem_X[MMQ_KCA_BLOCK_N][9];   // Q8_1 activations (+1 pad)
    __shared__ float   smem_Xs[MMQ_KCA_BLOCK_N];     // Q8_1 scale d8

    float acc[MMQ_KCA_RPT] = {};

    for (int sb_abs = 0; sb_abs < K_groups; ++sb_abs) {
        // Decompose sb_abs into super-block and sub-block indices
        const int sb_in_row = sb_abs / SB_PER_ROW;  // super-block index along row
        const int sb_in_sb  = sb_abs % SB_PER_ROW;  // sub-block within super-block
        const int g   = sb_in_sb / N_SB_PER_GROUP;  // group within super-block (0..N_GROUPS-1)
        const int sb  = sb_in_sb % N_SB_PER_GROUP;  // sub-block within group (0..7)
        const int sp  = sb >> 1;                     // sub-block pair (0..3)
        const int hi  = sb & 1;                      // lo(0) or hi(1) nibbles

        // ── Load weight nibbles and scales ────────────────────────────────────
        // 128 threads cover 32 rows × 8 int32s = 256 operations in two passes
        for (int ld = tid; ld < MMQ_KCA_BLOCK_M * 8; ld += MMQ_KCA_BLOCK_M * MMQ_KCA_NWARPS) {
            const int ld_row = ld >> 3;       // ld / 8
            const int ld_b   = ld & 7;        // ld % 8
            const int abs_row = tile_row + ld_row;

            if (abs_row < ne01) {
                const char * sb_ptr = src0_d + (int64_t)abs_row * nb01
                                             + (int64_t)sb_in_row * SB_SIZE;

                const uint8_t * qs_g;
                float D, Dmin;
                uint8_t sc_val, m_val;

                if constexpr (type == GGML_TYPE_Q4_KCA_64) {
                    const auto * blk = (const block_q4_KCA_64 *) sb_ptr;
                    qs_g  = blk->qs + g * G_BYTES + sp * 32;
                    D     = __half2float(blk->d[g]);
                    Dmin  = __half2float(blk->dmin[g]);
                    if (ld_b == 0) {
                        kca_get_scale_min(sb, blk->scales + g * 12, sc_val, m_val);
                        smem_Ws[ld_row] = D * (float)sc_val;
                        smem_Wm[ld_row] = Dmin * (float)m_val;
                    }
                } else {
                    const auto * blk = (const block_q4_KCA_128 *) sb_ptr;
                    qs_g  = blk->qs + g * G_BYTES + sp * 32;
                    D     = __half2float(blk->d[g]);
                    Dmin  = __half2float(blk->dmin[g]);
                    if (ld_b == 0) {
                        kca_get_scale_min(sb, blk->scales + g * 12, sc_val, m_val);
                        smem_Ws[ld_row] = D * (float)sc_val;
                        smem_Wm[ld_row] = Dmin * (float)m_val;
                    }
                }

                const int raw = ((const int *)qs_g)[ld_b];
                smem_W[ld_row][ld_b] = hi ? (raw >> 4) & 0x0F0F0F0F : raw & 0x0F0F0F0F;
            } else {
                smem_W[ld_row][ld_b] = 0;
                if (ld_b == 0) { smem_Ws[ld_row] = 0.f; smem_Wm[ld_row] = 0.f; }
            }
        }

        // ── Load Q8_1 activations (group-major layout → coalesced) ───────────
        // 128 threads cover 32 cols × 8 int32s = 256 operations in two passes
        for (int ld = tid; ld < MMQ_KCA_BLOCK_N * 8; ld += MMQ_KCA_BLOCK_M * MMQ_KCA_NWARPS) {
            const int ld_col = ld >> 3;
            const int ld_k   = ld & 7;
            const int abs_col = tile_col + ld_col;

            const auto * bq8 = (const block_q8_1 *)(src1_q8_d)
                               + (int64_t)sb_abs * ne11_pad + abs_col;

            if (abs_col < ne11) {
                smem_X[ld_col][ld_k] = ((const int32_t *)(bq8->qs))[ld_k];
                if (ld_k == 0) smem_Xs[ld_col] = __low2float(bq8->ds);
            } else {
                smem_X[ld_col][ld_k] = 0;
                if (ld_k == 0) smem_Xs[ld_col] = 0.f;
            }
        }

        __syncthreads();

        // ── dp4a accumulation ─────────────────────────────────────────────────
        const float d8    = smem_Xs[lane_id];
        const int * x_col = smem_X[lane_id];

        // Precompute sum of activations for min correction (column-only, row-independent)
        int sumq = 0;
#pragma unroll
        for (int k = 0; k < 8; ++k)
            sumq = ggml_cuda_dp4a(0x01010101, x_col[k], sumq);
        const float sumq_f = (float)sumq;

#pragma unroll
        for (int r = 0; r < MMQ_KCA_RPT; ++r) {
            const int row_idx = warp_id * MMQ_KCA_RPT + r;
            const int * w32   = smem_W[row_idx];

            int sumi = 0;
#pragma unroll
            for (int k = 0; k < 8; ++k)
                sumi = ggml_cuda_dp4a(w32[k], x_col[k], sumi);

            acc[r] += d8 * (smem_Ws[row_idx] * (float)sumi - smem_Wm[row_idx] * sumq_f);
        }

        __syncthreads();
    }

    // ── Write output ──────────────────────────────────────────────────────────
    const int abs_col = tile_col + lane_id;
#pragma unroll
    for (int r = 0; r < MMQ_KCA_RPT; ++r) {
        const int abs_row = tile_row + warp_id * MMQ_KCA_RPT + r;
        if (abs_row < ne01 && abs_col < ne11) {
            dst[(int64_t)abs_col * stride_dst + abs_row] = acc[r];
        }
    }
}

// ── Host launcher ─────────────────────────────────────────────────────────────
static void ggml_cuda_mul_mat_q4_KCA_f32(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor       * dst) {

    GGML_ASSERT(src0->type == GGML_TYPE_Q4_KCA_64 || src0->type == GGML_TYPE_Q4_KCA_128);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(src1->ne[2] == 1 && src1->ne[3] == 1);

    GGML_TENSOR_BINARY_OP_LOCALS

    cudaStream_t stream = ctx.stream();

    const int ne00_int = (int)ne00;
    const int ne11_int = (int)ne11;
    const int K_groups = ne00_int / QK_KCA_SB;   // sub-blocks = K-dimension
    const int ne11_pad = GGML_PAD(ne11_int, MMQ_KCA_BLOCK_N);

    // Quantize src1 to group-major Q8_1 (QK_KCA_SB = 32 elements per group)
    ggml_cuda_pool_alloc<char> src1_q8(ctx.pool(),
        (size_t)K_groups * ne11_pad * sizeof(block_q8_1));
    {
        const float * src1_d  = (const float *) src1->data;
        const int s01_tok = (int)(nb11 / sizeof(float));

        const dim3 blk(MMQ_KCA_BLOCK_N, 1);
        const dim3 grd((ne11_int + MMQ_KCA_BLOCK_N - 1) / MMQ_KCA_BLOCK_N, K_groups);
        quantize_q8_1_group_major<<<grd, blk, 0, stream>>>(
            src1_d, (block_q8_1 *)(char *)src1_q8.get(),
            ne00_int, ne11_int, ne11_pad, s01_tok);
    }

    const dim3 block(MMQ_KCA_BLOCK_N, MMQ_KCA_NWARPS);
    const dim3 grid(
        ((int)ne01 + MMQ_KCA_BLOCK_M - 1) / MMQ_KCA_BLOCK_M,
        (ne11_int  + MMQ_KCA_BLOCK_N - 1) / MMQ_KCA_BLOCK_N
    );

    const int64_t stride_dst = nb1 / sizeof(float);
    const char  * src0_d     = (const char *) src0->data;
    float       * dst_d      = (float *)       dst->data;

    if (src0->type == GGML_TYPE_Q4_KCA_64) {
        mul_mat_q4_KCA_f32_kernel<GGML_TYPE_Q4_KCA_64><<<grid, block, 0, stream>>>(
            src0_d, src1_q8.get(), dst_d,
            (int)ne01, ne11_int, K_groups, nb01, stride_dst, ne11_pad);
    } else {
        mul_mat_q4_KCA_f32_kernel<GGML_TYPE_Q4_KCA_128><<<grid, block, 0, stream>>>(
            src0_d, src1_q8.get(), dst_d,
            (int)ne01, ne11_int, K_groups, nb01, stride_dst, ne11_pad);
    }

    CUDA_CHECK(cudaGetLastError());
}
