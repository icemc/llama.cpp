#include "common.cuh"
#include "mmq.cuh"
#include "quantize.cuh"
#include "mmid.cuh"

template<typename src_block_t, typename dst_block_t, int qk_src, int qk_dst>
static __global__ void convert_blaq_rd_superblock_to_blaq_blocks(
        const src_block_t * __restrict__ src,
        dst_block_t       * __restrict__ dst,
        const int64_t blocks_per_row_src,
        const int64_t s01_src,
        const int64_t s02_src,
        const int64_t s03_src,
        const int64_t ne1,
        const int64_t ne2,
        const int64_t ne3) {
    static_assert(qk_src % qk_dst == 0, "qk_src must be divisible by qk_dst");

    constexpr int payload_src_bytes = qk_src / 2;
    constexpr int payload_dst_bytes = qk_dst / 2;
    constexpr int subblocks_per_superblock = qk_src / qk_dst;

    const int64_t b_src = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (b_src >= blocks_per_row_src) {
        return;
    }

    const int64_t i1 = blockIdx.y;
    const int64_t i0203 = blockIdx.z;
    const int64_t i3 = i0203 / ne2;
    const int64_t i2 = i0203 - i3 * ne2;

    if (i1 >= ne1 || i2 >= ne2 || i3 >= ne3) {
        return;
    }

    const int64_t src_idx = i3 * s03_src + i2 * s02_src + i1 * s01_src + b_src;
    const src_block_t src_block = src[src_idx];

    const int64_t blocks_per_row_dst = blocks_per_row_src * subblocks_per_superblock;
    const int64_t dst_row_base = ((i3 * ne2 + i2) * ne1 + i1) * blocks_per_row_dst;
    const int64_t dst_block_base = dst_row_base + b_src * subblocks_per_superblock;

#pragma unroll
    for (int sb = 0; sb < subblocks_per_superblock; ++sb) {
        dst_block_t out_block;
        out_block.d = src_block.d;

#pragma unroll
        for (int i = 0; i < payload_dst_bytes; ++i) {
            out_block.qs[i] = src_block.qs[sb * payload_dst_bytes + i];
        }

        dst[dst_block_base + sb] = out_block;
    }

    GGML_UNUSED(payload_src_bytes);
}

static void convert_blaq_rd_q4_cl64_to_blaq_q4_128_cuda(
        const void * src,
        void * dst,
        const int64_t ne00,
        const int64_t ne1,
        const int64_t ne2,
        const int64_t ne3,
        const int64_t s01_src,
        const int64_t s02_src,
        const int64_t s03_src,
        cudaStream_t stream) {
    GGML_ASSERT(ne00 % QK_BLAQ_RD_CL64 == 0);

    const int64_t blocks_per_row_src = ne00 / QK_BLAQ_RD_CL64;
    const dim3 num_blocks((blocks_per_row_src + CUDA_QUANTIZE_BLOCK_SIZE - 1) / CUDA_QUANTIZE_BLOCK_SIZE, ne1, ne2 * ne3);
    const dim3 block_size(CUDA_QUANTIZE_BLOCK_SIZE, 1, 1);

    convert_blaq_rd_superblock_to_blaq_blocks<block_blaq_rd_q4_cl64, block_blaq_q4_128, QK_BLAQ_RD_CL64, QK_BLAQ_128>
        <<<num_blocks, block_size, 0, stream>>>(
            (const block_blaq_rd_q4_cl64 *) src,
            (block_blaq_q4_128 *) dst,
            blocks_per_row_src,
            s01_src,
            s02_src,
            s03_src,
            ne1,
            ne2,
            ne3);
}

static void convert_blaq_rd_q4_cl128_to_blaq_q4_256_cuda(
        const void * src,
        void * dst,
        const int64_t ne00,
        const int64_t ne1,
        const int64_t ne2,
        const int64_t ne3,
        const int64_t s01_src,
        const int64_t s02_src,
        const int64_t s03_src,
        cudaStream_t stream) {
    GGML_ASSERT(ne00 % QK_BLAQ_RD_CL128 == 0);

    const int64_t blocks_per_row_src = ne00 / QK_BLAQ_RD_CL128;
    const dim3 num_blocks((blocks_per_row_src + CUDA_QUANTIZE_BLOCK_SIZE - 1) / CUDA_QUANTIZE_BLOCK_SIZE, ne1, ne2 * ne3);
    const dim3 block_size(CUDA_QUANTIZE_BLOCK_SIZE, 1, 1);

    convert_blaq_rd_superblock_to_blaq_blocks<block_blaq_rd_q4_cl128, block_blaq_q4_256, QK_BLAQ_RD_CL128, QK_BLAQ_256>
        <<<num_blocks, block_size, 0, stream>>>(
            (const block_blaq_rd_q4_cl128 *) src,
            (block_blaq_q4_256 *) dst,
            blocks_per_row_src,
            s01_src,
            s02_src,
            s03_src,
            ne1,
            ne2,
            ne3);
}

static void ggml_cuda_mul_mat_q_switch_type(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream) {
    switch (args.type_x) {
        case GGML_TYPE_Q4_0:
            mul_mat_q_case<GGML_TYPE_Q4_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_1:
            mul_mat_q_case<GGML_TYPE_Q4_1>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_0:
            mul_mat_q_case<GGML_TYPE_Q5_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_1:
            mul_mat_q_case<GGML_TYPE_Q5_1>(ctx, args, stream);
            break;
        case GGML_TYPE_Q8_0:
            mul_mat_q_case<GGML_TYPE_Q8_0>(ctx, args, stream);
            break;
        case GGML_TYPE_MXFP4:
            mul_mat_q_case<GGML_TYPE_MXFP4>(ctx, args, stream);
            break;
        case GGML_TYPE_Q2_K:
            mul_mat_q_case<GGML_TYPE_Q2_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q3_K:
            mul_mat_q_case<GGML_TYPE_Q3_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_K:
            mul_mat_q_case<GGML_TYPE_Q4_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_K:
            mul_mat_q_case<GGML_TYPE_Q5_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q6_K:
            mul_mat_q_case<GGML_TYPE_Q6_K>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_XXS:
            mul_mat_q_case<GGML_TYPE_IQ2_XXS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_XS:
            mul_mat_q_case<GGML_TYPE_IQ2_XS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_S:
            mul_mat_q_case<GGML_TYPE_IQ2_S>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ3_XXS:
            mul_mat_q_case<GGML_TYPE_IQ3_XXS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ3_S:
            mul_mat_q_case<GGML_TYPE_IQ3_S>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ1_S:
            mul_mat_q_case<GGML_TYPE_IQ1_S>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ4_XS:
            mul_mat_q_case<GGML_TYPE_IQ4_XS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ4_NL:
            mul_mat_q_case<GGML_TYPE_IQ4_NL>(ctx, args, stream);
            break;
        case GGML_TYPE_BLAQ_Q4_128:
            mul_mat_q_case<GGML_TYPE_BLAQ_Q4_128>(ctx, args, stream);
            break;
        case GGML_TYPE_BLAQ_Q4_256:
            mul_mat_q_case<GGML_TYPE_BLAQ_Q4_256>(ctx, args, stream);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

void ggml_cuda_mul_mat_q(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst) {
    GGML_ASSERT(        src1->type == GGML_TYPE_F32);
    GGML_ASSERT(        dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(!ids || ids->type  == GGML_TYPE_I32); // Optional, used for batched GGML_MUL_MAT_ID.

    GGML_TENSOR_BINARY_OP_LOCALS;

    cudaStream_t stream = ctx.stream();
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    GGML_ASSERT(        nb00       == ts_src0);
    GGML_ASSERT(        nb10       == ts_src1);
    GGML_ASSERT(        nb0        == ts_dst);
    GGML_ASSERT(!ids || ids->nb[0] == ggml_type_size(ids->type));

    const char  * src0_d = (const char  *) src0->data;
    const float * src1_d = (const float *) src1->data;
    float       *  dst_d = (float       *)  dst->data;

    // If src0 is a temporary compute buffer, clear any potential padding.
    if (ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        const size_t size_data  = ggml_nbytes(src0);
        const size_t size_alloc = ggml_backend_buffer_get_alloc_size(src0->buffer, src0);
        if (size_alloc > size_data) {
            GGML_ASSERT(ggml_is_contiguously_allocated(src0));
            GGML_ASSERT(!src0->view_src);
            CUDA_CHECK(cudaMemsetAsync((char *) src0->data + size_data, 0, size_alloc - size_data, stream));
        }
    }

    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);

    const int64_t s01 = src0->nb[1] / ts_src0;
    const int64_t s1  =  dst->nb[1] / ts_dst;
    const int64_t s02 = src0->nb[2] / ts_src0;
    const int64_t s2  =  dst->nb[2] / ts_dst;
    const int64_t s03 = src0->nb[3] / ts_src0;
    const int64_t s3  =  dst->nb[3] / ts_dst;

    ggml_type type_src0_mmq = src0->type;
    const char * src0_mmq_d = src0_d;
    int64_t s01_mmq = s01;
    int64_t s02_mmq = s02;
    int64_t s03_mmq = s03;
    ggml_cuda_pool_alloc<char> src0_mmq_conv(ctx.pool());

    if (src0->type == GGML_TYPE_BLAQ_RD_Q4_CL64) {
        GGML_ASSERT(ne00 % QK_BLAQ_RD_CL64 == 0);

        const int64_t nblocks_per_row_conv = ne00 / QK_BLAQ_128;
        const size_t nbytes_src0_conv = ne03 * ne02 * ne01 * nblocks_per_row_conv * sizeof(block_blaq_q4_128);
        src0_mmq_conv.alloc(nbytes_src0_conv);

        convert_blaq_rd_q4_cl64_to_blaq_q4_128_cuda(src0_d, src0_mmq_conv.get(), ne00, ne01, ne02, ne03, s01, s02, s03, stream);
        CUDA_CHECK(cudaGetLastError());

        src0_mmq_d = src0_mmq_conv.get();
        type_src0_mmq = GGML_TYPE_BLAQ_Q4_128;
        s01_mmq = nblocks_per_row_conv;
        s02_mmq = ne01 * s01_mmq;
        s03_mmq = ne02 * s02_mmq;
    } else if (src0->type == GGML_TYPE_BLAQ_RD_Q4_CL128) {
        GGML_ASSERT(ne00 % QK_BLAQ_RD_CL128 == 0);

        const int64_t nblocks_per_row_conv = ne00 / QK_BLAQ_256;
        const size_t nbytes_src0_conv = ne03 * ne02 * ne01 * nblocks_per_row_conv * sizeof(block_blaq_q4_256);
        src0_mmq_conv.alloc(nbytes_src0_conv);

        convert_blaq_rd_q4_cl128_to_blaq_q4_256_cuda(src0_d, src0_mmq_conv.get(), ne00, ne01, ne02, ne03, s01, s02, s03, stream);
        CUDA_CHECK(cudaGetLastError());

        src0_mmq_d = src0_mmq_conv.get();
        type_src0_mmq = GGML_TYPE_BLAQ_Q4_256;
        s01_mmq = nblocks_per_row_conv;
        s02_mmq = ne01 * s01_mmq;
        s03_mmq = ne02 * s02_mmq;
    }

    const bool use_stream_k = (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA)
                            || GGML_CUDA_CC_IS_CDNA(cc);

    // TODO: tighter pool buffer size vs q8 path
    const bool use_native_mxfp4 = blackwell_mma_available(cc) && src0->type == GGML_TYPE_MXFP4;

    if (!ids) {
        const size_t nbytes_src1_q8_1 = ne13*ne12 * ne11*ne10_padded * sizeof(block_q8_1)/QK8_1 +
            get_mmq_x_max_host(cc)*sizeof(block_q8_1_mmq);
        ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), nbytes_src1_q8_1);

        {
            const int64_t s11 = src1->nb[1] / ts_src1;
            const int64_t s12 = src1->nb[2] / ts_src1;
            const int64_t s13 = src1->nb[3] / ts_src1;
            if (use_native_mxfp4) {
                static_assert(sizeof(block_fp4_mmq) == 4 * sizeof(block_q8_1));
                quantize_mmq_mxfp4_cuda(src1_d, nullptr, src1_q8_1.get(), type_src0_mmq, ne10, s11, s12, s13, ne10_padded,
                                        ne11, ne12, ne13, stream);

            } else {
                quantize_mmq_q8_1_cuda(src1_d, nullptr, src1_q8_1.get(), type_src0_mmq, ne10, s11, s12, s13, ne10_padded,
                                       ne11, ne12, ne13, stream);
            }
            CUDA_CHECK(cudaGetLastError());
        }

        // Stride depends on quantization format
        const int64_t s12 = use_native_mxfp4 ?
                                ne11 * ne10_padded * sizeof(block_fp4_mmq) /
                                    (8 * QK_MXFP4 * sizeof(int))  // block_fp4_mmq holds 256 values (8 blocks of 32)
                                :
                                ne11 * ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
        const int64_t s13 = ne12*s12;

        const mmq_args args = {
            src0_mmq_d, type_src0_mmq, (const int *) src1_q8_1.ptr, nullptr, nullptr, dst_d,
            ne00, ne01, ne1, s01_mmq, ne11, s1,
            ne02, ne12, s02_mmq, s12, s2,
            ne03, ne13, s03_mmq, s13, s3,
            use_stream_k, ne1};
        ggml_cuda_mul_mat_q_switch_type(ctx, args, stream);
        return;
    }

    GGML_ASSERT(ne13 == 1);
    GGML_ASSERT(nb12 % nb11 == 0);
    GGML_ASSERT(nb2  % nb1  == 0);

    const int64_t n_expert_used = ids->ne[0];
    const int64_t ne_get_rows = ne12 * n_expert_used;
    GGML_ASSERT(ne1 == n_expert_used);

    ggml_cuda_pool_alloc<int32_t> ids_src1(ctx.pool(), ne_get_rows);
    ggml_cuda_pool_alloc<int32_t> ids_dst(ctx.pool(), ne_get_rows);
    ggml_cuda_pool_alloc<int32_t> expert_bounds(ctx.pool(), ne02 + 1);

    {
        GGML_ASSERT(ids->nb[0] == ggml_element_size(ids));
        const int si1  = ids->nb[1] / ggml_element_size(ids);
        const int sis1 = nb12 / nb11;

        ggml_cuda_launch_mm_ids_helper((const int32_t *) ids->data, ids_src1.get(), ids_dst.get(), expert_bounds.get(),
            ne02, ne12, n_expert_used, ne11, si1, sis1, stream);
        CUDA_CHECK(cudaGetLastError());
    }

    const size_t nbytes_src1_q8_1 = ne12*n_expert_used*ne10_padded * sizeof(block_q8_1)/QK8_1 +
        get_mmq_x_max_host(cc)*sizeof(block_q8_1_mmq);
    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), nbytes_src1_q8_1);

    const int64_t ne11_flat = ne12*n_expert_used;
    const int64_t ne12_flat = 1;
    const int64_t ne13_flat = 1;

    {
        const int64_t s11 = src1->nb[1] / ts_src1;
        const int64_t s12 = src1->nb[2] / ts_src1;
        const int64_t s13 = src1->nb[3] / ts_src1;

        if (use_native_mxfp4) {
            quantize_mmq_mxfp4_cuda(src1_d, ids_src1.get(), src1_q8_1.get(), type_src0_mmq, ne10, s11, s12, s13,
                                    ne10_padded, ne11_flat, ne12_flat, ne13_flat, stream);
        } else {
            quantize_mmq_q8_1_cuda(src1_d, ids_src1.get(), src1_q8_1.get(), type_src0_mmq, ne10, s11, s12, s13,
                                   ne10_padded, ne11_flat, ne12_flat, ne13_flat, stream);
        }
        CUDA_CHECK(cudaGetLastError());
    }

    const int64_t s12 = use_native_mxfp4 ? ne11 * ne10_padded * sizeof(block_fp4_mmq) / (8 * QK_MXFP4 * sizeof(int)) :
                                           ne11 * ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
    const int64_t s13 = ne12*s12;

    // Note that ne02 is used instead of ne12 because the number of y channels determines the z dimension of the CUDA grid.
    const mmq_args args = {
        src0_mmq_d, type_src0_mmq, (const int *) src1_q8_1.get(), ids_dst.get(), expert_bounds.get(), dst_d,
        ne00, ne01, ne_get_rows, s01_mmq, ne_get_rows, s1,
        ne02, ne02, s02_mmq, s12, s2,
        ne03, ne13, s03_mmq, s13, s3,
        use_stream_k, ne12};

    ggml_cuda_mul_mat_q_switch_type(ctx, args, stream);
}

void ggml_cuda_op_mul_mat_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream) {

    const int64_t ne00 = src0->ne[0];

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    GGML_ASSERT(ne10 % QK8_1 == 0);

    const int64_t ne0 = dst->ne[0];

    const int64_t row_diff = row_high - row_low;
    int64_t stride01 = ne00 / ggml_blck_size(src0->type);

    const int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;

    // the main device has a larger memory buffer to hold the results from all GPUs
    // nrows_dst == nrows of the matrix that the kernel writes into
    const int64_t nrows_dst = id == ctx.device ? ne0 : row_diff;

    // The stream-k decomposition is only faster for recent NVIDIA GPUs.
    // Also its fixup needs to allocate a temporary buffer in the memory pool.
    // There are multiple parallel CUDA streams for src1_ncols != ne11 which would introduce a race condition for this buffer.
    ggml_type type_src0_mmq = src0->type;
    const char * src0_dd_i_mmq = src0_dd_i;
    ggml_cuda_pool_alloc<char> src0_mmq_conv(ctx.pool());

    if (src0->type == GGML_TYPE_BLAQ_RD_Q4_CL64) {
        GGML_ASSERT(ne00 % QK_BLAQ_RD_CL64 == 0);

        const int64_t nblocks_per_row_conv = ne00 / QK_BLAQ_128;
        const size_t nbytes_src0_conv = row_diff * nblocks_per_row_conv * sizeof(block_blaq_q4_128);
        src0_mmq_conv.alloc(nbytes_src0_conv);

        convert_blaq_rd_q4_cl64_to_blaq_q4_128_cuda(src0_dd_i, src0_mmq_conv.get(), ne00, row_diff, 1, 1, stride01, 0, 0, stream);
        CUDA_CHECK(cudaGetLastError());

        src0_dd_i_mmq = src0_mmq_conv.get();
        type_src0_mmq = GGML_TYPE_BLAQ_Q4_128;
        stride01 = nblocks_per_row_conv;
    } else if (src0->type == GGML_TYPE_BLAQ_RD_Q4_CL128) {
        GGML_ASSERT(ne00 % QK_BLAQ_RD_CL128 == 0);

        const int64_t nblocks_per_row_conv = ne00 / QK_BLAQ_256;
        const size_t nbytes_src0_conv = row_diff * nblocks_per_row_conv * sizeof(block_blaq_q4_256);
        src0_mmq_conv.alloc(nbytes_src0_conv);

        convert_blaq_rd_q4_cl128_to_blaq_q4_256_cuda(src0_dd_i, src0_mmq_conv.get(), ne00, row_diff, 1, 1, stride01, 0, 0, stream);
        CUDA_CHECK(cudaGetLastError());

        src0_dd_i_mmq = src0_mmq_conv.get();
        type_src0_mmq = GGML_TYPE_BLAQ_Q4_256;
        stride01 = nblocks_per_row_conv;
    }

    const bool use_stream_k = ((GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA)
                            || GGML_CUDA_CC_IS_CDNA(cc))
                            && src1_ncols == ne11;
    const mmq_args args = {
        src0_dd_i_mmq, type_src0_mmq, (const int *) src1_ddq_i, nullptr, nullptr, dst_dd_i,
        ne00, row_diff, src1_ncols, stride01, ne11, nrows_dst,
        1, 1, 0, 0, 0,
        1, 1, 0, 0, 0,
        use_stream_k, src1_ncols};

    ggml_cuda_mul_mat_q_switch_type(ctx, args, stream);

    GGML_UNUSED_VARS(src1, dst, src1_ddf_i, src1_padded_row_size);
}

bool ggml_cuda_should_use_mmq(enum ggml_type type, int cc, int64_t ne11, int64_t n_experts) {
#ifdef GGML_CUDA_FORCE_CUBLAS
    return false;
#endif // GGML_CUDA_FORCE_CUBLAS

    bool mmq_supported;

    switch (type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_BLAQ_Q4_128:
        case GGML_TYPE_BLAQ_Q4_256:
        case GGML_TYPE_BLAQ_RD_Q4_CL64:
        case GGML_TYPE_BLAQ_RD_Q4_CL128:
            mmq_supported = true;
            break;
        default:
            mmq_supported = false;
            break;
    }

    if (!mmq_supported) {
        return false;
    }

    if (turing_mma_available(cc)) {
        return true;
    }

    if (ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_DP4A) {
        return false;
    }

#ifdef GGML_CUDA_FORCE_MMQ
    return true;
#endif //GGML_CUDA_FORCE_MMQ

    if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
        return !fp16_mma_hardware_available(cc) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
    }

    if (amd_mfma_available(cc)) {
        // As of ROCM 7.0 rocblas/tensile performs very poorly on CDNA3 and hipblaslt (via ROCBLAS_USE_HIPBLASLT)
        // performs better but is currently suffering from a crash on this architecture.
        // TODO: Revisit when hipblaslt is fixed on CDNA3
        if (GGML_CUDA_CC_IS_CDNA3(cc)) {
            return true;
        }
        if (n_experts > 64 || ne11 <= 128) {
            return true;
        }
        if (type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q4_1 || type == GGML_TYPE_Q5_0 || type == GGML_TYPE_Q5_1) {
            return true;
        }
        if (ne11 <= 256 && (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K)) {
            return true;
        }
        return false;
    }

    if (amd_wmma_available(cc)) {
        if (GGML_CUDA_CC_IS_RDNA3(cc)) {
            // High expert counts are almost always better on MMQ due to
            //     the synchronization overhead in the cuBLAS/hipBLAS path:
            // https://github.com/ggml-org/llama.cpp/pull/18202
            if (n_experts >= 64) {
                return true;
            }

            // For some quantization types MMQ can have lower peak TOPS than hipBLAS
            //     so it's only faster for sufficiently small batch sizes:
            switch (type) {
                case GGML_TYPE_Q2_K:
                    return ne11 <= 128;
                case GGML_TYPE_Q6_K:
                    return ne11 <= (GGML_CUDA_CC_IS_RDNA3_0(cc) ? 128 : 256);
                case GGML_TYPE_IQ2_XS:
                case GGML_TYPE_IQ2_S:
                    return GGML_CUDA_CC_IS_RDNA3_5(cc) || ne11 <= 128;
                default:
                    return true;
            }
        }

        // For RDNA4 MMQ is consistently faster than dequantization + hipBLAS:
        // https://github.com/ggml-org/llama.cpp/pull/18537#issuecomment-3706422301
        return true;
    }

    return (!GGML_CUDA_CC_IS_CDNA(cc)) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;

}
