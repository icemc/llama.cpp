#pragma once

// mmq-q4c.cuh — Standalone GEMM kernel for Q4_C prompt processing (PP)
//
// Problem: The MMQ framework uses MMQ_ITER_K=256, but Q4_C super-blocks are
// 1024 (Q4_C_64) or 2048 (Q4_C_128) weights wide → blocks_per_iter = 0 (infinite loop).
//
// Solution: Dedicated tiled GEMM that dequantizes Q4_C nibbles on-the-fly via the
// NF4 int8 table and uses dp4a for integer accumulation.
//
// Performance key: src1 is quantized to Q8_1 in GROUP-MAJOR (transposed) layout:
//   q8_trans[group_abs * ne11_pad + token] rather than q8_std[token * nblocks + group]
// This makes the 32-token tile for each group contiguous → fully coalesced loads.
//
// Design (BLOCK_M=32, BLOCK_N=32, NWARPS=4, RPT=8):
//   Grid:  (ceil(ne01/32), ceil(ne11/32))
//   Block: dim3(32, 4) = 128 threads
//   Each thread: 8 weight rows × 1 token col → 8 accumulators
//
//   Shared memory (~2 KB / block):
//     smem_W[32][8]  — NF4 int8 (8 int32) per weight row per group
//     smem_Ws[32]    — scale d/s_g/127 per weight row
//     smem_X[32][9]  — Q8_1 int8 (8 int32 + 1 pad) per token col per group
//     smem_Xs[32]    — Q8_1 scale d8 per token col

#include "common.cuh"
#include "quantize.cuh"
#include "vecdotq-cquant.cuh"

static constexpr int MMQ_Q4C_BLOCK_M = 32;
static constexpr int MMQ_Q4C_BLOCK_N = 32;  // = WARP_SIZE
static constexpr int MMQ_Q4C_NWARPS  = 4;
static constexpr int MMQ_Q4C_RPT     = MMQ_Q4C_BLOCK_M / MMQ_Q4C_NWARPS; // 8 rows per thread

// ── Transposed Q8_1 quantization ─────────────────────────────────────────────
// Quantizes src1 [F32, ne11 tokens × ne00 elements] to Q8_1 in GROUP-MAJOR layout:
//   output[(g_abs * ne11_pad) + token] for g_abs = element_group, token = token_idx
// Consecutive tokens for the same group are contiguous → coalesced reads in GEMM.

static __global__ void quantize_q8_1_group_major(
        const float * __restrict__ src,   // [ne11 tokens × ne00 elements]
        block_q8_1  * __restrict__ dst,   // [K_groups × ne11_pad] (group-major)
        const int ne00,      // elements per token (K dimension)
        const int ne11,      // number of tokens (N dimension)
        const int ne11_pad,  // ne11 padded to 32 (allocated token stride in dst)
        const int s01        // src stride between tokens, in floats
) {
    const int token = blockIdx.x * blockDim.x + threadIdx.x;
    const int g_abs = blockIdx.y;           // which K group (0..K_groups-1)
    const int e0    = g_abs * QK8_1;        // first element index in this group

    if (token >= ne11) return;

    const float * row = src + (int64_t)token * s01 + e0;

    // Compute amax over 32 elements (zero-pad out-of-bounds)
    float amax = 0.f;
    float v[QK8_1];
#pragma unroll
    for (int j = 0; j < QK8_1; ++j) {
        float x = (e0 + j < ne00) ? row[j] : 0.f;
        v[j]  = x;
        amax  = fmaxf(amax, fabsf(x));
    }

    const float d  = amax / 127.f;
    const float id = (amax > 0.f) ? 127.f / amax : 0.f;

    block_q8_1 * blk = dst + (int64_t)g_abs * ne11_pad + token;
    blk->ds = make_half2(d, 0.f);  // sum field unused by our GEMM

#pragma unroll
    for (int j = 0; j < QK8_1; ++j) {
        blk->qs[j] = (int8_t)roundf(v[j] * id);
    }
}

// (quantize_q8_1_transposed is called inline in ggml_cuda_mul_mat_q4_C_f32)

// ── GEMM kernel ───────────────────────────────────────────────────────────────
template <ggml_type type>
static __global__ void __launch_bounds__(MMQ_Q4C_BLOCK_N * MMQ_Q4C_NWARPS, 2)
mul_mat_q4_C_f32_kernel(
        const char  * __restrict__ src0_d,     // quantized weight data
        const char  * __restrict__ src1_q8_d,  // Q8_1 activations, group-major layout
        float       * __restrict__ dst,
        const int ne01,                    // M: output features (weight rows)
        const int ne11,                    // N: tokens (original, before pad)
        const int K_groups,                // ne00 / QK_UAP_G
        const int64_t nb01,                // stride in bytes between weight rows in src0
        const int64_t stride_dst,          // stride in floats between dst columns
        const int ne11_pad                 // padded token count (stride in group-major Q8_1)
) {
    static_assert(type == GGML_TYPE_Q4_C_64 || type == GGML_TYPE_Q4_C_128,
                  "mul_mat_q4_C_f32_kernel: unsupported type");

    constexpr int SB_SIZE = (type == GGML_TYPE_Q4_C_64)
                             ? (int)sizeof(block_q4_C_64)
                             : (int)sizeof(block_q4_C_128);
    constexpr int QK_SB   = (type == GGML_TYPE_Q4_C_64) ? QK_C_64   : QK_C_128;
    constexpr int NGRP_SB = QK_SB / QK_UAP_G;  // groups per super-block: 32 or 64

    const int warp_id = threadIdx.y;
    const int lane_id = threadIdx.x;   // = token col within tile
    const int tid     = warp_id * WARP_SIZE + lane_id;

    const int tile_row = blockIdx.x * MMQ_Q4C_BLOCK_M;
    const int tile_col = blockIdx.y * MMQ_Q4C_BLOCK_N;

    // Shared memory
    // Weight NF4 int8 values: smem_W[row][k] where k indexes 8 int32 chunks
    __shared__ int32_t smem_W[MMQ_Q4C_BLOCK_M][8];
    // Effective NF4 scale: d / s_g / 127
    __shared__ float   smem_Ws[MMQ_Q4C_BLOCK_M];
    // Q8_1 int8 values: smem_X[col][k], 9 int32s per col (8 data + 1 bank-conflict pad)
    __shared__ int32_t smem_X[MMQ_Q4C_BLOCK_N][9];
    // Q8_1 scale: d8 per col
    __shared__ float   smem_Xs[MMQ_Q4C_BLOCK_N];

    float acc[MMQ_Q4C_RPT] = {};

    for (int g_abs = 0; g_abs < K_groups; ++g_abs) {
        const int sb      = g_abs / NGRP_SB;
        const int g_local = g_abs % NGRP_SB;

        // ── Load weight NF4 ──────────────────────────────────────────────────
        // 128 threads load 32 rows × 4 chunks of 8 nibbles = 128 int32 loads
        // tid: row = tid/4, b = tid%4 (which 4-byte chunk within 16-byte group)
        {
            const int load_row = tid >> 2;
            const int b        = tid &  3;
            const int abs_row  = tile_row + load_row;

            if (abs_row < ne01) {
                const char * sb_ptr = src0_d + (int64_t)abs_row * nb01
                                             + (int64_t)sb * SB_SIZE;
                const uint8_t * g_qs;
                float d_eff;

                if constexpr (type == GGML_TYPE_Q4_C_64) {
                    const auto * blk = (const block_q4_C_64 *) sb_ptr;
                    g_qs  = blk->qs + g_local * 16;
                    d_eff = __half2float(blk->d[g_local]);
                } else {
                    const auto * blk = (const block_q4_C_128 *) sb_ptr;
                    g_qs  = blk->qs + g_local * 16;
                    d_eff = __half2float(blk->d[g_local]);
                }

                const int v32    = get_int_b2(g_qs, b);
                const int2 v     = get_int_from_table_16(v32, nf4_int8_table);
                const int seq_lo = __byte_perm(v.x, v.y, 0x5140);
                const int seq_hi = __byte_perm(v.x, v.y, 0x7362);

                smem_W[load_row][b * 2    ] = seq_lo;
                smem_W[load_row][b * 2 + 1] = seq_hi;

                if (b == 0) {
                    smem_Ws[load_row] = d_eff * (1.0f/127.0f);
                }
            } else {
                smem_W[load_row][b * 2    ] = 0;
                smem_W[load_row][b * 2 + 1] = 0;
                if (b == 0) smem_Ws[load_row] = 0.0f;
            }
        }

        // ── Load Q8_1 activations (group-major layout → coalesced) ───────────
        // 128 threads load 32 tokens × 8 int32s = 256 int32s
        // tid: col = tid%32, chunk = tid/32 (0..3, covers 2 int32s = 8 int8s each)
        // Access: dst[g_abs * ne11_pad + (tile_col + col)] — consecutive cols → coalesced!
        {
            const int load_col = tid & 31;
            const int chunk    = tid >> 5;
            const int abs_col  = tile_col + load_col;

            const auto * bq8 = (const block_q8_1 *)(src1_q8_d)
                               + (int64_t)g_abs * ne11_pad + abs_col;

            if (abs_col < ne11) {
                smem_X[load_col][chunk * 2    ] = ((const int32_t *)(bq8->qs))[chunk * 2    ];
                smem_X[load_col][chunk * 2 + 1] = ((const int32_t *)(bq8->qs))[chunk * 2 + 1];
                if (chunk == 0) smem_Xs[load_col] = __low2float(bq8->ds);
            } else {
                smem_X[load_col][chunk * 2    ] = 0;
                smem_X[load_col][chunk * 2 + 1] = 0;
                if (chunk == 0) smem_Xs[load_col] = 0.0f;
            }
        }

        __syncthreads();

        // ── dp4a accumulation ────────────────────────────────────────────────
        const float   d8   = smem_Xs[lane_id];
        const int32_t * acts = smem_X[lane_id];

#pragma unroll
        for (int r = 0; r < MMQ_Q4C_RPT; ++r) {
            const int   row_idx = warp_id * MMQ_Q4C_RPT + r;
            const float d_eff   = smem_Ws[row_idx];
            const int32_t * w32 = smem_W[row_idx];

            int sumi = 0;
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                sumi = ggml_cuda_dp4a(w32[k], acts[k], sumi);
            }
            acc[r] += d_eff * d8 * (float)sumi;
        }

        __syncthreads();
    }

    // ── Write output ─────────────────────────────────────────────────────────
    const int abs_col = tile_col + lane_id;
#pragma unroll
    for (int r = 0; r < MMQ_Q4C_RPT; ++r) {
        const int abs_row = tile_row + warp_id * MMQ_Q4C_RPT + r;
        if (abs_row < ne01 && abs_col < ne11) {
            dst[(int64_t)abs_col * stride_dst + abs_row] = acc[r];
        }
    }
}

// ── Host launcher ─────────────────────────────────────────────────────────────
static void ggml_cuda_mul_mat_q4_C_f32(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor       * dst) {

    GGML_ASSERT(src0->type == GGML_TYPE_Q4_C_64 || src0->type == GGML_TYPE_Q4_C_128);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(src1->ne[2] == 1 && src1->ne[3] == 1);

    GGML_TENSOR_BINARY_OP_LOCALS   // declares ne00,ne01,ne10,ne11,ne12,ne13,nb00,nb01,...

    cudaStream_t stream = ctx.stream();

    // ne00 must already be a multiple of QK_UAP_G (= QK8_1 = 32) due to zero-padding
    const int ne00_int = (int)ne00;
    const int ne11_int = (int)ne11;
    const int K_groups = ne00_int / QK_UAP_G;
    const int ne11_pad = GGML_PAD(ne11_int, MMQ_Q4C_BLOCK_N);  // multiple of 32

    // Quantize src1 to group-major Q8_1 (consecutive tokens per group → coalesced GEMM)
    ggml_cuda_pool_alloc<char> src1_q8(ctx.pool(),
        (size_t)K_groups * ne11_pad * sizeof(block_q8_1));
    {
        const float * src1_d  = (const float *) src1->data;
        // nb11 from GGML_TENSOR_BINARY_OP_LOCALS = src1->nb[1] = stride between tokens in bytes
        const int s01_tok = (int)(nb11 / sizeof(float));

        const dim3 blk(MMQ_Q4C_BLOCK_N, 1);
        const dim3 grd((ne11_int + MMQ_Q4C_BLOCK_N - 1) / MMQ_Q4C_BLOCK_N, K_groups);
        quantize_q8_1_group_major<<<grd, blk, 0, stream>>>(
            src1_d, (block_q8_1 *)(char *)src1_q8.get(),
            ne00_int, ne11_int, ne11_pad, s01_tok);
    }

    // Launch tiled GEMM
    const dim3 block(MMQ_Q4C_BLOCK_N, MMQ_Q4C_NWARPS);
    const dim3 grid(
        ((int)ne01 + MMQ_Q4C_BLOCK_M - 1) / MMQ_Q4C_BLOCK_M,
        (ne11_int  + MMQ_Q4C_BLOCK_N - 1) / MMQ_Q4C_BLOCK_N
    );

    // nb01 from GGML_TENSOR_BINARY_OP_LOCALS = src0->nb[1] (weight row stride in bytes)
    // nb1  from GGML_TENSOR_BINARY_OP_LOCALS = dst->nb[1]  (dst col stride in bytes)
    const int64_t stride_dst = nb1 / sizeof(float);
    const char  * src0_d     = (const char *) src0->data;
    float       * dst_d      = (float *)       dst->data;

    if (src0->type == GGML_TYPE_Q4_C_64) {
        mul_mat_q4_C_f32_kernel<GGML_TYPE_Q4_C_64><<<grid, block, 0, stream>>>(
            src0_d, src1_q8.get(), dst_d,
            (int)ne01, ne11_int, K_groups, nb01, stride_dst, ne11_pad);
    } else {
        mul_mat_q4_C_f32_kernel<GGML_TYPE_Q4_C_128><<<grid, block, 0, stream>>>(
            src0_d, src1_q8.get(), dst_d,
            (int)ne01, ne11_int, K_groups, nb01, stride_dst, ne11_pad);
    }

    CUDA_CHECK(cudaGetLastError());
}
