// SPDX-License-Identifier: Apache-2.0
// Modified 2026 by Kristian Aaron (Krishando). Original: ashhart/TensorFold@d1018b318 (Apache-2.0).
// After ExLlamaV3, MIT, Copyright (c) 2025 Turboderp; DeepSeek integration by @jayleaton.
// Adapted from bertholomus/TensorFold@ddfad35:src/tensorfold/cuda/exl3/decode.cuh (Apache-2.0). Changes: CB 3 E4M3FN round-trip.

// EXL3 tiles: lane L decodes values 8L..8L+7 straight into its two mma B fragments; port of tensorfold's decode.cuh.
#pragma once

#include <cstdint>
#include <cuda_fp16.h>

namespace tf_exl3 {

enum Codebook : int { CB_3INST = 0, CB_MCG = 1, CB_MUL1 = 2, CB_MCG_E4M3 = 3 };

// Round an fp32 value to OCP E4M3FN (nearest-even, finite saturation), then widen it.
// This implements the mcg_e4m3 decode contract.
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
    if (e == 15 && m == 7) return __int_as_float(0x7fffffffu);
    float v = e == 0 ? m * 0x1p-9f : (8 + m) * ldexpf(1.f, e - 10);
    return (code & 0x80) ? -v : v;
}

// 32-bit words of a tile: 256 values of K2 / 2 bits.
template <int K2>
__host__ __device__ constexpr int tile_words() {
    return 4 * K2;
}

// E(p): one past the last stream bit of value p's 16-bit window (exl3_format.zig's streamEnds).
template <int K2>
__host__ __device__ constexpr int stream_end(int p) {
    return (K2 & 1) ? ((p + 1) * K2 - ((p + 1) & 1)) / 2 : (p + 1) * (K2 / 2);
}

// Words a lane reads for its 8 values: the most any lane's windows span (2, or 3 for 3.5 bits and 5 to 8 bits).
template <int K2>
__host__ __device__ constexpr int lane_words() {
    int most = 0;
    for (int l = 0; l < 32; ++l) {
        const int first = (stream_end<K2>(8 * l) - 16 + 128 * K2) % 32;
        const int need = first + stream_end<K2>(8 * l + 7) - stream_end<K2>(8 * l) + 16;
        most = need > most ? need : most;
    }
    return (most + 31) / 32;
}

// The index of the first word lane `lane` reads, and the bit offset of its first window in that word.
template <int K2>
__device__ __forceinline__ void lane_start(int lane, int& word, int& offset) {
    const int first = 4 * lane * K2 + K2 / 2 - 16 + 128 * K2;   // E(8 lane) - 16, made non-negative
    word = (first >> 5) % tile_words<K2>();
    offset = first & 31;
}

// This lane's words of a tile (tile: the tile's first 32-bit word; global or shared memory).
template <int K2>
__device__ __forceinline__ void load_lane_words(const uint32_t* tile, int lane, uint32_t (&w)[lane_words<K2>()]) {
    int word, offset;
    lane_start<K2>(lane, word, offset);
#pragma unroll
    for (int i = 0; i < lane_words<K2>(); ++i) w[i] = tile[(word + i) % tile_words<K2>()];
}

// The same from global memory through the read-only cache.
template <int K2>
__device__ __forceinline__ void ldg_lane_words(const uint32_t* tile, int lane, uint32_t (&w)[lane_words<K2>()]) {
    int word, offset;
    lane_start<K2>(lane, word, offset);
#pragma unroll
    for (int i = 0; i < lane_words<K2>(); ++i) w[i] = __ldg(tile + (word + i) % tile_words<K2>());
}

// The weight stream is read once, so do not allocate L1 for it: same value as __ldg, cache policy only.
// Ported from ashhart/TensorFold PR #1534 (Apache-2.0).
__device__ __forceinline__ uint32_t ldg_stream_u32(const uint32_t* p) {
    uint32_t v;
    asm volatile("ld.global.nc.L1::no_allocate.u32 %0, [%1];" : "=r"(v) : "l"(p));
    return v;
}

template <int K2>
__device__ __forceinline__ void ldg_lane_words_noalloc(const uint32_t* tile, int lane, uint32_t (&w)[lane_words<K2>()]) {
    int word, offset;
    lane_start<K2>(lane, word, offset);
#pragma unroll
    for (int i = 0; i < lane_words<K2>(); ++i) w[i] = ldg_stream_u32(tile + (word + i) % tile_words<K2>());
}

// Bits [d, d + 16) of the 96-bit big-endian stream hi:mid:lo (d a compile-time constant after unrolling).
__device__ __forceinline__ uint32_t window16(uint32_t hi, uint32_t mid, uint32_t lo, int d) {
    if (d <= 16) return (hi >> (16 - d)) & 0xffffu;
    if (d < 32) return __funnelshift_l(mid, hi, d) >> 16;
    if (d <= 48) return (mid >> (48 - d)) & 0xffffu;
    return __funnelshift_l(lo, mid, d - 32) >> 16;
}

// The 16-bit states of this lane's 8 values (8 lane + j, j = 0..7) from its words.
template <int K2>
__device__ __forceinline__ void lane_states(const uint32_t (&w)[lane_words<K2>()], int lane, uint32_t (&s)[8]) {
    int word, offset;
    lane_start<K2>(lane, word, offset);
    const uint32_t c = lane_words<K2>() > 2 ? w[lane_words<K2>() > 2 ? 2 : 0] : 0u;
    const uint32_t hi = __funnelshift_l(w[1], w[0], offset);   // the stream from the lane's first window on
    const uint32_t mid = __funnelshift_l(c, w[1], offset);
    const uint32_t lo = c << offset;
#pragma unroll
    for (int j = 0; j < 8; ++j) s[j] = window16(hi, mid, lo, stream_end<K2>(j) - stream_end<K2>(0));
}

// Two codebook values from two 16-bit states, as a half2 in a uint32 (the first state in the low half).
template <int CB>
__device__ __forceinline__ uint32_t decode2(uint32_t s0, uint32_t s1) {
    if constexpr (CB == CB_MUL1) {
        const uint32_t x0 = s0 * 0x83DCD12Du, x1 = s1 * 0x83DCD12Du;
        const uint32_t h0 = __dp4a(x0, 0x01010101u, 0x6400u);    // fp16 bits of 1024 + the byte sum
        const uint32_t h1 = __dp4a(x1, 0x01010101u, 0x6400u);
        const uint32_t hh = (h0 & 0xffffu) | (h1 << 16);
        const half2 r = __hfma2(*reinterpret_cast<const half2*>(&hh), __half2half2(__ushort_as_half(0x1eee)),
                                __half2half2(__ushort_as_half(0xc931)));
        return *reinterpret_cast<const uint32_t*>(&r);
    } else {
        uint32_t x0, x1;
        if constexpr (CB == CB_MCG || CB == CB_MCG_E4M3) {
            x0 = s0 * 0xCBAC1FEDu;
            x1 = s1 * 0xCBAC1FEDu;
        } else {
            x0 = s0 * 89226354u + 64248484u;
            x1 = s1 * 89226354u + 64248484u;
        }
        x0 = (x0 & 0x8FFF8FFFu) ^ 0x3B603B60u;
        x1 = (x1 & 0x8FFF8FFFu) ^ 0x3B603B60u;
        const uint32_t lo = __byte_perm(x0, x1, 0x5410);
        const uint32_t hi = __byte_perm(x0, x1, 0x7632);
        const half2 r = __hadd2(*reinterpret_cast<const half2*>(&lo), *reinterpret_cast<const half2*>(&hi));
        const uint32_t out = *reinterpret_cast<const uint32_t*>(&r);
        if constexpr (CB == CB_MCG_E4M3) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
            // The paired conversion sequence is also used by gr8 (MIT); see THIRD_PARTY_NOTICES.md.
            uint16_t packed;
            uint32_t restored;
            asm volatile("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;\n" : "=h"(packed) : "r"(out));
            asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;\n" : "=r"(restored) : "h"(packed));
            return restored;
#else
            const half2 h = *reinterpret_cast<const half2*>(&out);
            const float a = __half2float(__low2half(h));
            const float b = __half2float(__high2half(h));
            const half2 q = __halves2half2(__float2half_rn(e4m3_value(round_e4m3(a))),
                                           __float2half_rn(e4m3_value(round_e4m3(b))));
            return *reinterpret_cast<const uint32_t*>(&q);
#endif
        }
        return out;
    }
}

// The two e4m3 codes decode2<CB_MCG_E4M3> rounds to, low byte = s0, high byte = s1.
// Stops before the f16 widen (cvt.rn.f16x2.e4m3x2, or e4m3_value + float2half on the fallback).
// The mcg pair above that round is the same arithmetic as decode2.
__device__ __forceinline__ uint32_t decode2_e4m3_codes(uint32_t s0, uint32_t s1) {
    uint32_t x0 = s0 * 0xCBAC1FEDu;
    uint32_t x1 = s1 * 0xCBAC1FEDu;
    x0 = (x0 & 0x8FFF8FFFu) ^ 0x3B603B60u;
    x1 = (x1 & 0x8FFF8FFFu) ^ 0x3B603B60u;
    const uint32_t lo = __byte_perm(x0, x1, 0x5410);
    const uint32_t hi = __byte_perm(x0, x1, 0x7632);
    const half2 summed = __hadd2(*reinterpret_cast<const half2*>(&lo), *reinterpret_cast<const half2*>(&hi));
    const uint32_t out = *reinterpret_cast<const uint32_t*>(&summed);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    uint16_t codes;
    asm volatile("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;\n" : "=h"(codes) : "r"(out));
    return codes;
#else
    const half2 h = *reinterpret_cast<const half2*>(&out);
    const float a = __half2float(__low2half(h));
    const float b = __half2float(__high2half(h));
    return (uint32_t)round_e4m3(a) | ((uint32_t)round_e4m3(b) << 8);
#endif
}

// This lane's 8 values of a tile as the B fragments of the tile's two n8 halves (columns 0-7 in b0, 8-15 in b1).
template <int K2, int CB>
__device__ __forceinline__ void decode_lane(const uint32_t (&w)[lane_words<K2>()], int lane, uint32_t (&b0)[2],
                                            uint32_t (&b1)[2]) {
    uint32_t s[8];
    lane_states<K2>(w, lane, s);
    b0[0] = decode2<CB>(s[0], s[1]);
    b0[1] = decode2<CB>(s[2], s[3]);
    b1[0] = decode2<CB>(s[4], s[5]);
    b1[1] = decode2<CB>(s[6], s[7]);
}

// Row and column in the 16x16 tile (row = k, column = n) of this lane's value j (0..7).
__host__ __device__ constexpr int value_row(int lane, int j) { return 2 * (lane & 3) + (j & 1) + 8 * ((j >> 1) & 1); }
__host__ __device__ constexpr int value_col(int lane, int j) { return (lane >> 2) + 8 * (j >> 2); }

// mma.m16n8k16, fp16 inputs, fp32 accumulators: d += a @ b.
__device__ __forceinline__ void mma16816(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// Unscaled Walsh-Hadamard transform of 128 fp32 values, 4 a lane, in a fixed butterfly order (multiply by 1/sqrt(128)).
__device__ __forceinline__ void fwht128(float (&v)[4], int lane) {
    const float a = v[0] + v[1], b = v[0] - v[1], c = v[2] + v[3], d = v[2] - v[3];
    v[0] = a + c;
    v[1] = b + d;
    v[2] = a - c;
    v[3] = b - d;
#pragma unroll
    for (int m = 1; m < 32; m <<= 1) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float o = __shfl_xor_sync(0xffffffffu, v[j], m);
            v[j] = (lane & m) ? o - v[j] : v[j] + o;
        }
    }
}

// Two independent 128-value transforms, interleaved stage by stage. Each list takes exactly fwht128's
// operations in the same order, so both results are fwht128's bits; only issue slots are shared.
__device__ __forceinline__ void fwht128x2(float (&v0)[4], float (&v1)[4], int lane) {
    const float a0 = v0[0] + v0[1], b0 = v0[0] - v0[1], c0 = v0[2] + v0[3], d0 = v0[2] - v0[3];
    const float a1 = v1[0] + v1[1], b1 = v1[0] - v1[1], c1 = v1[2] + v1[3], d1 = v1[2] - v1[3];
    v0[0] = a0 + c0; v1[0] = a1 + c1;
    v0[1] = b0 + d0; v1[1] = b1 + d1;
    v0[2] = a0 - c0; v1[2] = a1 - c1;
    v0[3] = b0 - d0; v1[3] = b1 - d1;
#pragma unroll
    for (int m = 1; m < 32; m <<= 1) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float o0 = __shfl_xor_sync(0xffffffffu, v0[j], m);
            const float o1 = __shfl_xor_sync(0xffffffffu, v1[j], m);
            v0[j] = (lane & m) ? o0 - v0[j] : v0[j] + o0;
            v1[j] = (lane & m) ? o1 - v1[j] : v1[j] + o1;
        }
    }
}

constexpr float HAD_SCALE = 0.08838834764831845f;   // 1 / sqrt(128)

}  // namespace tf_exl3
