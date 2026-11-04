// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
// Adapted from bertholomus/TensorFold@ddfad35:src/tensorfold/cuda/exl3/decode.cuh (Apache-2.0).
// EXL3 format derives from ExLlamaV3 (MIT); Had128/SUH/SVH follow exllamav3 (MIT). Changes: X3E4 staging and epilogues.
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>

namespace {

__device__ __forceinline__ int a_offset(int lane, int reg, int byte) {
    const int i = reg * 4 + byte;
    const int group = lane >> 2, tid = lane & 3;
    const int row = group + ((i < 4 || (i >= 8 && i < 12)) ? 0 : 8);
    const int k = tid * 4 + (i & 3) + (i >= 8 ? 16 : 0);
    return row * 32 + k;
}

__device__ __forceinline__ int b_offset(int lane, int reg, int byte) {
    const int i = reg * 4 + byte;
    const int group = lane >> 2, tid = lane & 3;
    const int k = tid * 4 + (i & 3) + (i >= 4 ? 16 : 0);
    return k * 8 + group;
}

__device__ __forceinline__ void mma_e4m3(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
#if __CUDA_ARCH__ >= 890
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#else
    __trap();
#endif
}

__device__ __forceinline__ uint8_t round_e4m3(float x) {
    const uint32_t bits = __float_as_uint(x);
    const uint8_t sign = (bits >> 24) & 0x80;
    const uint32_t a = bits & 0x7fffffffu;
    if (a > 0x7f800000u) return sign | 0x7fu;
    if (a >= 0x43e00000u) return sign | 0x7eu;
    if (a >= 0x3c800000u) {
        const uint32_t r = (a + 0x7ffffu + ((a >> 20) & 1u)) & 0xfff00000u;
        return sign | (uint8_t)((((r >> 23) - 120u) << 3) | ((r >> 20) & 7u));
    }
    const uint32_t e = a >> 23;
    if (e == 0 || e < 101u) return sign;
    const uint32_t shift = 141u - e;
    const uint64_t m = (a & 0x7fffffu) | 0x800000u;
    const uint64_t half = 1ull << (shift - 1);
    const uint64_t rem = m & ((1ull << shift) - 1);
    uint64_t n = m >> shift;
    if (rem > half || (rem == half && (n & 1))) ++n;
    return sign | (uint8_t)n;
}

__device__ __forceinline__ float e4m3_value(uint8_t code) {
    const int e = (code >> 3) & 15;
    const int m = code & 7;
    const float v = e == 0 ? m * 0x1p-9f : (8 + m) * ldexpf(1.f, e - 10);
    return (code & 0x80) ? -v : v;
}

__device__ __forceinline__ uint8_t decode_mcg_e4m3(uint16_t state) {
    const uint32_t x = (uint32_t)state * 0xCBAC1FEDu;
    const uint32_t bits = (x & 0x8FFF8FFFu) ^ 0x3B603B60u;
    const half lo = __ushort_as_half((uint16_t)bits);
    const half hi = __ushort_as_half((uint16_t)(bits >> 16));
    const float value = __half2float(__hadd(lo, hi));
    return round_e4m3(value);
}

__device__ __forceinline__ int weight_fragment_index(int row, int k, int k_width) {
    const int kin = k % 32;
    const int a = (row % 64) / 8;
    const int h = (k % 64) / 32;
    for (int lane_lo = 0; lane_lo < 4; ++lane_lo) {
        for (int b = 0; b < 8; ++b) {
            const int p = 16 * (b / 4) + 4 * lane_lo + b % 4;
            const int mapped = 16 * (p / 16) + 2 * ((p % 16) / 4) + p % 2 + 8 * ((p % 4) / 2);
            if (mapped == kin) {
                const int lane = (row % 8) * 4 + lane_lo;
                return (((((row / 64) * (k_width / 64) + k / 64) * 8 + a) * 32 + lane) * 2 + h) * 8 + b;
            }
        }
    }
    return 0;
}

}  // namespace

// One launch decodes a whole EXL3 [K/16,N/16] matrix into WP3 FP8G packed order.
// The launch grid is (tiles_n, tiles_k); one CTA handles each 16x16 trellis tile.
// The accompanying UE8M0 scale is a placeholder unit A scale (127); runtime
// activation scales are produced per member and K32 by tf_glm53_act_mxfp8.
extern "C" __global__ void tf_glm53_x3e4_tile(const uint32_t* __restrict__ trellis, int bits,
                                               int tiles_n, int tiles_k, int k_width,
                                               uint8_t* __restrict__ packed,
                                               uint8_t* __restrict__ scale_ue8m0) {
    const int row0 = (int)blockIdx.x * 16;
    const int k0 = (int)blockIdx.y * 16;
    const int i = (int)threadIdx.x;
    const int tile = ((int)blockIdx.y * tiles_n + (int)blockIdx.x) * (8 * bits);
    if (i < 256) {
        const int end = (i + 1) * bits;
        const int start = end + 256 * bits - 16;
        uint16_t state = 0;
        for (int b = 0; b < 16; ++b) {
            const int pos = start + b;
            const uint32_t word = trellis[tile + (pos / 32) % (8 * bits)];
            state = (uint16_t)((state << 1) | ((word >> (31 - (pos % 32))) & 1u));
        }
        const int lane = i / 8;
        const int j = i % 8;
        const int k = 2 * (lane % 4) + (j & 1) + 8 * ((j >> 1) & 1);
        const int n = lane / 4 + 8 * (j >> 2);
        const int dst = weight_fragment_index(row0 + n, k0 + k, k_width);
        packed[dst] = decode_mcg_e4m3(state);
    }
    if (i == 0 && blockIdx.x == 0 && blockIdx.y == 0 && tiles_n > 0 && tiles_k > 0) {
#pragma unroll
        for (int g = 0; g < 4; ++g) scale_ue8m0[g] = 127;  // unit A scale for the 128-K stage.
    }
}

// Host fixture: row-major 16x32 by 32x8 E4M3 fragment MMA.
extern "C" __global__ void tf_glm53_x3e4_fragment_mma(const uint8_t* __restrict__ a,
                                                       const uint8_t* __restrict__ b,
                                                       float* __restrict__ out) {
    const int lane = (int)threadIdx.x;
    uint32_t af[4] = {}, bf[2] = {};
#pragma unroll
    for (int reg = 0; reg < 4; ++reg)
#pragma unroll
        for (int byte = 0; byte < 4; ++byte)
            af[reg] |= static_cast<uint32_t>(a[a_offset(lane, reg, byte)]) << (8 * byte);
#pragma unroll
    for (int reg = 0; reg < 2; ++reg)
#pragma unroll
        for (int byte = 0; byte < 4; ++byte)
            bf[reg] |= static_cast<uint32_t>(b[b_offset(lane, reg, byte)]) << (8 * byte);
    float accum[4] = {0.f, 0.f, 0.f, 0.f};
    mma_e4m3(accum, af, bf);
    const int group = lane >> 2, tid = lane & 3;
    out[group * 8 + tid * 2] = accum[0];
    out[group * 8 + tid * 2 + 1] = accum[1];
    out[(group + 8) * 8 + tid * 2] = accum[2];
    out[(group + 8) * 8 + tid * 2 + 1] = accum[3];
}

// WP4 routed ACT_BF16 gate/up epilogue. GLM-5.3 has no SwiGLU limit clamp.
// SiLU uses expf plus IEEE division, then rounds to bf16 before multiplication.
extern "C" __global__ void tf_glm53_x3e4_swiglu(const __nv_bfloat16* __restrict__ gate,
                                                const __nv_bfloat16* __restrict__ up,
                                                __nv_bfloat16* __restrict__ out, int count) {
    const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < count) {
        const float g = __bfloat162float(gate[i]);
        const float u = __bfloat162float(up[i]);
        const float silu = __bfloat162float(__float2bfloat16_rn(g / (1.f + expf(-g))));
        out[i] = __float2bfloat16_rn(silu * u);
    }
}

// Deterministic final routed reduction: eight routed slots in WP8 slot order, fp32 output.
extern "C" __global__ void tf_glm53_x3e4_route_sum(const float* __restrict__ slots,
                                                   const float* __restrict__ weights,
                                                   float* __restrict__ routed,
                                                   int rows, int slot_count,
                                                   int slot_stride, int hidden) {
    const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < rows * hidden) {
        const int row = i / hidden;
        const int col = i - row * hidden;
        float acc = 0.f;
        for (int slot = 0; slot < slot_count; ++slot) {
            const float y = slots[((size_t)row * slot_stride + slot) * hidden + col];
            const float w = weights[(size_t)row * slot_stride + slot];
            acc = fmaf(w, y, acc);
        }
        routed[i] = acc;
    }
}
