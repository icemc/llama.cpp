#include "common.cuh"

static __device__ __forceinline__ void dequantize_q4_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q4_0 * x = (const block_q4_0 *) vx;

    const float d = x[ib].d;

    const int vui = x[ib].qs[iqs];

    v.x = vui & 0xF;
    v.y = vui >> 4;

    v.x = (v.x - 8.0f) * d;
    v.y = (v.y - 8.0f) * d;
}

static __device__ __forceinline__ void dequantize_q4_1(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q4_1 * x = (const block_q4_1 *) vx;

    const float2 dm = __half22float2(x[ib].dm);

    const int vui = x[ib].qs[iqs];

    v.x = vui & 0xF;
    v.y = vui >> 4;

    v.x = (v.x * dm.x) + dm.y;
    v.y = (v.y * dm.x) + dm.y;
}

static __device__ __forceinline__ void dequantize_q5_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q5_0 * x = (const block_q5_0 *) vx;

    const float d = x[ib].d;

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const int xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y = ((x[ib].qs[iqs] >>  4) | xh_1);

    v.x = (v.x - 16.0f) * d;
    v.y = (v.y - 16.0f) * d;
}

static __device__ __forceinline__ void dequantize_q5_1(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q5_1 * x = (const block_q5_1 *) vx;

    const float2 dm = __half22float2(x[ib].dm);

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const int xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y = ((x[ib].qs[iqs] >>  4) | xh_1);

    v.x = (v.x * dm.x) + dm.y;
    v.y = (v.y * dm.x) + dm.y;
}

static __device__ __forceinline__ void dequantize_blaq_q4_128(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_blaq_q4_128 * x = (const block_blaq_q4_128 *) vx;

    const float2 dm = __half22float2(x[ib].dm);
    const int vui = x[ib].qs[iqs];

    v.x = (vui & 0xF) * dm.x + dm.y;
    v.y = (vui >>  4) * dm.x + dm.y;
}

static __device__ __forceinline__ void dequantize_blaq_q4_256(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_blaq_q4_256 * x = (const block_blaq_q4_256 *) vx;

    const float2 dm = __half22float2(x[ib].dm);
    const int vui = x[ib].qs[iqs];

    v.x = (vui & 0xF) * dm.x + dm.y;
    v.y = (vui >>  4) * dm.x + dm.y;
}

// Get a 6-bit value from a bit-stream buffer at position idx.
static __device__ __forceinline__ float blaq_ska_get_scale_f(const uint8_t * buf, int idx) {
    const int bit_off   = 6 * idx;
    const int byte_idx  = bit_off >> 3;
    const int bit_shift = bit_off & 7;
    unsigned val = (unsigned)buf[byte_idx] >> bit_shift;
    if (bit_shift > 2) val |= (unsigned)buf[byte_idx + 1] << (8 - bit_shift);
    return (float)(val & 0x3Fu);
}

static __device__ __forceinline__ void dequantize_blaq_ska_128(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_blaq_ska_128 * x = (const block_blaq_ska_128 *) vx;
    const float d    = __half2float(x[ib].d);
    const float dmin = __half2float(x[ib].dmin);
    // Each sub-block has 32 weights = 16 bytes; sub-block index from byte offset.
    const int sub    = iqs / 16;
    const float dk   = d    * blaq_ska_get_scale_f(x[ib].scales,     sub);
    const float mk   = dmin * blaq_ska_get_scale_f(x[ib].scales + 3, sub);
    const int vui = x[ib].qs[iqs];

    v.x = (vui & 0xF) * dk - mk;
    v.y = (vui >>  4) * dk - mk;
}

static __device__ __forceinline__ void dequantize_blaq_ska_256(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_blaq_ska_256 * x = (const block_blaq_ska_256 *) vx;
    const float d    = __half2float(x[ib].d);
    const float dmin = __half2float(x[ib].dmin);
    const int sub    = iqs / 16;
    const float dk   = d    * blaq_ska_get_scale_f(x[ib].scales,     sub);
    const float mk   = dmin * blaq_ska_get_scale_f(x[ib].scales + 6, sub);
    const int vui = x[ib].qs[iqs];

    v.x = (vui & 0xF) * dk - mk;
    v.y = (vui >>  4) * dk - mk;
}

static __device__ __forceinline__ void dequantize_q8_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q8_0 * x = (const block_q8_0 *) vx;

    const float d = x[ib].d;

    v.x = x[ib].qs[iqs + 0];
    v.y = x[ib].qs[iqs + 1];

    v.x *= d;
    v.y *= d;
}
