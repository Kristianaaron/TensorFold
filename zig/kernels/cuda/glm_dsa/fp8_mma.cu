// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
#include "../tf_vblock.cuh"
// Adapted from ashhart/TensorFold@d1018b31:zig/kernels/cuda/ck_lane.cu (Apache-2.0).
// The MMA register operation follows ashhart/TensorFold@d1018b31:zig/kernels/cuda/mma4.cuh (Apache-2.0).
// Changes: per-row activation scales, FP32 block scaling and deterministic sm_121 MMA.
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <math_constants.h>
#include <stdint.h>

__device__ __forceinline__ int tf_glm53_xoff(int row, int k, int K) {
    const int local = k & 15;
    const int q = (local & 7) / 2;
    const int j = (local & 1) + 2 * (local / 8);
    return row * K + (k / 16) * 16 + 4 * q + j;
}

__device__ __forceinline__ int tf_glm53_woff(int row, int k, int K) {
    const int local = k & 31;
    const int a = (row & 63) / 8;
    const int h = (k & 63) / 32;
    for (int lane_lo = 0; lane_lo < 4; ++lane_lo) {
        for (int b = 0; b < 8; ++b) {
            const int p = 16 * (b / 4) + 4 * lane_lo + b % 4;
            const int kin = 16 * (p / 16) + 2 * ((p % 16) / 4) + p % 2 + 8 * ((p % 4) / 2);
            if (kin == local) {
                const int lane = (row & 7) * 4 + lane_lo;
                return (((((row / 64) * (K / 64) + k / 64) * 8 + a) * 32 + lane) * 2 + h) * 8 + b;
            }
        }
    }
    return 0;
}

__device__ __forceinline__ int tf_glm53_woff_tile(int row, int k) {
    // Inverse of FP8G's packed-byte map, so four fragment bytes can be gathered directly.
    const int kin = k & 31, a = (row & 63) / 8, h = (k & 63) / 32;
    const int lane_lo = (kin & 7) >> 1;
    const int b = (kin >> 4) * 4 + (kin & 1) + ((kin & 8) >> 2);
    return (((a * 32 + (row & 7) * 4 + lane_lo) * 2 + h) * 8 + b);
}

__device__ __forceinline__ void tf_glm53_cp16(unsigned char* dst, const unsigned char* src, bool valid) {
    const unsigned smem = static_cast<unsigned>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(smem), "l"(src), "r"(valid ? 16 : 0));
}

__device__ __forceinline__ void tf_glm53_cp_commit() { asm volatile("cp.async.commit_group;"); }
__device__ __forceinline__ void tf_glm53_cp_wait() { asm volatile("cp.async.wait_group 0;"); }

__device__ __forceinline__ uint32_t tf_glm53_pack_a(const unsigned char* a, int row, int k, int tig, int half) {
    uint32_t v = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int logical = k + half * 16 + 4 * tig + i;
        const int q = (logical & 7) / 2;
        const int j = (logical & 1) + 2 * ((logical & 15) / 8);
        v |= static_cast<uint32_t>(a[row * 128 + (logical / 16) * 16 + 4 * q + j]) << (8 * i);
    }
    return v;
}

__device__ __forceinline__ uint32_t tf_glm53_pack_b(const unsigned char* b, int row, int k, int tig, int half) {
    uint32_t v = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int logical = k + half * 16 + 4 * tig + i;
        const int block = (row / 64) * 2 + (logical / 64);
        v |= static_cast<uint32_t>(b[block * 4096 + tf_glm53_woff_tile(row, logical)]) << (8 * i);
    }
    return v;
}

__device__ __forceinline__ void tf_glm53_mma(float (&c)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
#if __CUDA_ARCH__ >= 890
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#else
    __trap();
#endif
}

extern "C" __global__ void __launch_bounds__(256)
tf_glm53_peak_bf16(int iterations, float* __restrict__ sink) {
    TF_VBLOCK_ENTER();
    uint32_t a[4] = { 0x3f803f80u, 0x3f803f80u, 0x3f803f80u, 0x3f803f80u };
    uint32_t b[2] = { 0x3f803f80u, 0x3f803f80u };
    float c0[4] = { 0, 0, 0, 0 }, c1[4] = { 0, 0, 0, 0 };
    float c2[4] = { 0, 0, 0, 0 }, c3[4] = { 0, 0, 0, 0 };
#pragma unroll 1
    for (int i = 0; i < iterations; ++i) {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                     : "+f"(c0[0]), "+f"(c0[1]), "+f"(c0[2]), "+f"(c0[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                     : "+f"(c1[0]), "+f"(c1[1]), "+f"(c1[2]), "+f"(c1[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                     : "+f"(c2[0]), "+f"(c2[1]), "+f"(c2[2]), "+f"(c2[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                     : "+f"(c3[0]), "+f"(c3[1]), "+f"(c3[2]), "+f"(c3[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    }
    sink[static_cast<size_t>(TF_VBID.x) * 256 + threadIdx.x] = c0[0] + c1[0] + c2[0] + c3[0];
}

extern "C" __global__ void __launch_bounds__(256)
tf_glm53_peak_fp8(int iterations, float* __restrict__ sink) {
    TF_VBLOCK_ENTER();
    uint32_t a[4] = { 0x38383838u, 0x38383838u, 0x38383838u, 0x38383838u };
    uint32_t b[2] = { 0x38383838u, 0x38383838u };
    float c0[4] = { 0, 0, 0, 0 }, c1[4] = { 0, 0, 0, 0 };
    float c2[4] = { 0, 0, 0, 0 }, c3[4] = { 0, 0, 0, 0 };
    for (int i = 0; i < iterations; ++i) {
        tf_glm53_mma(c0, a, b);
        tf_glm53_mma(c1, a, b);
        tf_glm53_mma(c2, a, b);
        tf_glm53_mma(c3, a, b);
    }
    sink[static_cast<size_t>(TF_VBID.x) * 256 + threadIdx.x] = c0[0] + c1[0] + c2[0] + c3[0];
}

// Raw accumulator probe: unlike tf_glm53_fp8_mma, C is fed back for every
// k32 instruction without a per-128 promotion. Each block tests one FP8 pair.
extern "C" __global__ void __launch_bounds__(32)
tf_glm53_raw_fp8_acc_probe(const uint32_t* __restrict__ a_words,
                           const uint32_t* __restrict__ b_words,
                           const float* __restrict__ seeds, int iterations,
                           float* __restrict__ out) {
    TF_VBLOCK_ENTER();
    const int probe = TF_VBID.x;
    const int lane = threadIdx.x;
    uint32_t a[4] = {0, 0, 0, 0};
    uint32_t b[2] = {0, 0};
    float c[4] = {0, 0, 0, 0};
    if (lane == 0) {
        a[0] = a_words[probe];
        b[0] = b_words[probe];
        c[0] = seeds[probe];
    }
    for (int i = 0; i < iterations; ++i) tf_glm53_mma(c, a, b);
    if (lane == 0) out[probe] = c[0];
}

// Same C-feedback measurement for bf16 m16n8k16, included as a control.
extern "C" __global__ void __launch_bounds__(32)
tf_glm53_raw_bf16_acc_probe(const uint32_t* __restrict__ a_words,
                            const uint32_t* __restrict__ b_words,
                            const float* __restrict__ seeds, int iterations,
                            float* __restrict__ out) {
    TF_VBLOCK_ENTER();
    const int probe = TF_VBID.x;
    const int lane = threadIdx.x;
    uint32_t a[4] = {0, 0, 0, 0};
    uint32_t b[2] = {0, 0};
    float c[4] = {0, 0, 0, 0};
    if (lane == 0) {
        a[0] = a_words[probe];
        b[0] = b_words[probe];
        c[0] = seeds[probe];
    }
    for (int i = 0; i < iterations; ++i) {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                     "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                     : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    }
    if (lane == 0) out[probe] = c[0];
}

__device__ __forceinline__ void tf_glm53_load_stage(unsigned char* smem, int stage, int m0, int n0, int K, int M,
                                                     int N, int kg, const uint8_t* xq, const uint8_t* w) {
    constexpr int TILE_BYTES = 128 * 128;
    constexpr int STAGE_BYTES = 2 * TILE_BYTES;
    const int tid = threadIdx.x;
    unsigned char* dst = smem + stage * STAGE_BYTES;
    const int kb = kg * 128;
    for (int i = tid; i < TILE_BYTES / 16; i += TF_VBDIM.x) {
        const int row = i / 8, chunk = i % 8;
        const bool live = m0 + row < M;
        const uint8_t* src = xq + static_cast<size_t>(live ? m0 + row : 0) * K + kb + chunk * 16;
        tf_glm53_cp16(dst + row * 128 + chunk * 16, src, live);
    }
    unsigned char* wb = dst + TILE_BYTES;
    for (int i = tid; i < 4 * (4096 / 16); i += TF_VBDIM.x) {
        const int block = i / (4096 / 16), chunk = i % (4096 / 16);
        const int nr = (block / 2) * 64, kc = (block % 2) * 64;
        const bool live = n0 + nr < ((N + 63) / 64) * 64;
        const size_t at = static_cast<size_t>(tf_glm53_woff(n0 + nr, kb + kc, K)) + chunk * 16;
        tf_glm53_cp16(wb + block * 4096 + chunk * 16, w + (live ? at : 0), live);
    }
    tf_glm53_cp_commit();
}

// Fragment order matches ck_lane's A8 64-byte chunks. CUDA implementation is
// deliberately independent of any FP8G checkpoint qmm mode.
__device__ __forceinline__ void tf_glm53_act_q8_task(const __nv_bfloat16* __restrict__ x, long long ldx, int M, int K,
                 uint8_t* __restrict__ xq, float* __restrict__ xs) {
    const int row = TF_VBID.x, group = TF_VBID.y, tid = threadIdx.x;
    const int base = group * 128;
    __shared__ float vals[128];
    float v = 0.0f;
    for (int i = tid; i < 128; i += TF_VBDIM.x) {
        v = __bfloat162float(x[static_cast<long long>(row) * ldx + base + i]);
        vals[i] = fabsf(v);
    }
    TF_TASK_SYNC();
    for (int d = 64; d; d >>= 1) {
        if (tid < d) vals[tid] = fmaxf(vals[tid], vals[tid + d]);
        TF_TASK_SYNC();
    }
    const float scale = fmaxf(vals[0], 1.0e-10f) / 448.0f;
    if (tid == 0) xs[static_cast<long long>(row) * (K / 128) + group] = scale;
    for (int i = tid; i < 128; i += TF_VBDIM.x) {
        const float a = __bfloat162float(x[static_cast<long long>(row) * ldx + base + i]) / scale;
        const int k = base + i;
        xq[tf_glm53_xoff(row, k, K)] = __nv_cvt_float_to_fp8(a, __NV_SATFINITE, __NV_E4M3);
    }
}

extern "C" __global__ void __launch_bounds__(128)
tf_glm53_act_q8(const __nv_bfloat16* __restrict__ x, long long ldx, int M, int K, uint8_t* __restrict__ xq, float* __restrict__ xs) {
    TF_VBLOCK_ENTER();
    tf_glm53_act_q8_task(x, ldx, M, K, xq, xs);
}

// MXFP8's 1x32 UE8M0 scales stay explicit and do not alter the default K2 path.
__device__ __forceinline__ void tf_glm53_act_mxfp8_task(const __nv_bfloat16* __restrict__ x, long long ldx, int M, int K,
                    uint8_t* __restrict__ xq, uint8_t* __restrict__ xs) {
    const int row = TF_VBID.x, group = TF_VBID.y, tid = threadIdx.x;
    const int base = group * 32;
    __shared__ float vals[32];
    if (tid < 32) vals[tid] = fabsf(__bfloat162float(x[static_cast<long long>(row) * ldx + base + tid]));
    TF_TASK_SYNC();
    for (int d = 16; d; d >>= 1) { if (tid < d) vals[tid] = fmaxf(vals[tid], vals[tid + d]); TF_TASK_SYNC(); }
    int e = 0;
    if (vals[0] != 0.0f) {
        int exp2 = 0;
        const float frac = frexpf(vals[0] / 448.0f, &exp2);
        e = exp2 - (frac == 0.5f ? 1 : 0);
    }
    e = max(-127, min(127, e));
    const float scale = ldexpf(1.0f, e);
    if (tid == 0) xs[static_cast<long long>(row) * (K / 32) + group] = static_cast<uint8_t>(e + 127);
    if (tid < 32) xq[tf_glm53_xoff(row, base + tid, K)] = __nv_cvt_float_to_fp8(__bfloat162float(x[static_cast<long long>(row) * ldx + base + tid]) / scale, __NV_SATFINITE, __NV_E4M3);
}

extern "C" __global__ void __launch_bounds__(128)
tf_glm53_act_mxfp8(const __nv_bfloat16* __restrict__ x, long long ldx, int M, int K, uint8_t* __restrict__ xq, uint8_t* __restrict__ xs) {
    TF_VBLOCK_ENTER();
    tf_glm53_act_mxfp8_task(x, ldx, M, K, xq, xs);
}

// Scalar oracle entry point. It retains the exact FP32 128-K partial/scaling
// order for comparison against the sm_121 tensor-core implementation.
// The reduction tree is independent of M, so it is also a row-invariance oracle.
__device__ __forceinline__ void tf_glm53_fp8_mma_scalar_task(const uint8_t* __restrict__ xq, const float* __restrict__ xs,
                  const uint8_t* __restrict__ w, const float* __restrict__ ws,
                  void* __restrict__ out, int M, int N, int K, int output_f32) {
    const long long index = static_cast<long long>(TF_VBID.x) * TF_VBDIM.x + threadIdx.x;
    const long long total = static_cast<long long>(M) * N;
    if (index >= total) return;
    const int row = static_cast<int>(index / N), col = static_cast<int>(index % N);
    float acc = 0.0f;
    for (int g = 0; g < K / 128; ++g) {
        float partial = 0.0f;
        for (int t = 0; t < 128; ++t) {
            const int k = g * 128 + t;
            const float av = __half2float(__half(__nv_cvt_fp8_to_halfraw(xq[tf_glm53_xoff(row, k, K)], __NV_E4M3)));
            const float bv = __half2float(__half(__nv_cvt_fp8_to_halfraw(w[tf_glm53_woff(col, k, K)], __NV_E4M3)));
            partial = partial + av * bv;
        }
        const float scale = ws[static_cast<long long>(col / 128) * (K / 128) + g] * xs[static_cast<long long>(row) * (K / 128) + g];
        acc = fmaf(partial, scale, acc);
    }
    if (output_f32) static_cast<float*>(out)[index] = acc;
    else static_cast<__nv_bfloat16*>(out)[index] = __float2bfloat16_rn(acc);
}

extern "C" __global__ void __launch_bounds__(256)
tf_glm53_fp8_mma_scalar(const uint8_t* xq, const float* xs, const uint8_t* w, const float* ws,
                        void* out, int M, int N, int K, int output_f32) {
    TF_VBLOCK_ENTER();
    tf_glm53_fp8_mma_scalar_task(xq, xs, w, ws, out, M, N, K, output_f32);
}

// Eight warps share one 128x128 output tile. Each warp owns one 16x8 MMA
// microtile at a time and walks the other microtiles in fixed order. Four 32-K
// MMA instructions make each 128-K partial; only after that partial is complete
// do we apply ws*xs with one FMA. Two 128-K input stages occupy 64 KiB dynamic smem.
__device__ __forceinline__ void tf_glm53_fp8_mma_task(const uint8_t* __restrict__ xq, const float* __restrict__ xs,
                  const uint8_t* __restrict__ w, const float* __restrict__ ws,
                  void* __restrict__ out, int M, int N, int K, int output_f32, unsigned char* smem) {
    constexpr int TILE_BYTES = 128 * 128;
    constexpr int STAGE_BYTES = 2 * TILE_BYTES;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int m0 = TF_VBID.y * 128, n0 = TF_VBID.x * 128;
    const int num_k = K / 128;
    if (num_k == 0) return;
    tf_glm53_load_stage(smem, 0, m0, n0, K, M, N, 0, xq, w);
    tf_glm53_cp_wait();
    TF_TASK_SYNC();
    float acc[16][4];
#pragma unroll
    for (int t = 0; t < 16; ++t) {
#pragma unroll
        for (int c = 0; c < 4; ++c) acc[t][c] = 0.0f;
    }
    for (int kg = 0; kg < num_k; ++kg) {
        const int stage = kg & 1;
        if (kg + 1 < num_k) tf_glm53_load_stage(smem, stage ^ 1, m0, n0, K, M, N, kg + 1, xq, w);
        unsigned char* base = smem + stage * STAGE_BYTES;
        const unsigned char* as = base;
        const unsigned char* bs = base + TILE_BYTES;
        const int group = lane >> 2, tig = lane & 3;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            const int t = warp + 8 * i;
            const int mt = t / 16, nt = t % 16;
            float partial[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
            for (int h = 0; h < 4; ++h) {
                uint32_t a[4], b[2];
                const int row = mt * 16 + group, col = nt * 8 + group;
                a[0] = tf_glm53_pack_a(as, row, h * 32, tig, 0);
                a[1] = tf_glm53_pack_a(as, row + 8, h * 32, tig, 0);
                a[2] = tf_glm53_pack_a(as, row, h * 32, tig, 1);
                a[3] = tf_glm53_pack_a(as, row + 8, h * 32, tig, 1);
                b[0] = tf_glm53_pack_b(bs, col, h * 32, tig, 0);
                b[1] = tf_glm53_pack_b(bs, col, h * 32, tig, 1);
                tf_glm53_mma(partial, a, b);
            }
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                const int row = m0 + mt * 16 + group + (c >= 2 ? 8 : 0);
                const int col = n0 + nt * 8 + 2 * tig + (c & 1);
                if (row < M && col < N) {
                    const float scale = ws[static_cast<long long>(col / 128) * (K / 128) + kg] *
                                        xs[static_cast<long long>(row) * (K / 128) + kg];
                    acc[i][c] = fmaf(partial[c], scale, acc[i][c]);
                }
            }
        }
        TF_TASK_SYNC();
        if (kg + 1 < num_k) {
            tf_glm53_cp_wait();
            TF_TASK_SYNC();
        }
    }
    const int group = lane >> 2, tig = lane & 3;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        const int t = warp + 8 * i;
        const int mt = t / 16, nt = t % 16;
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            const int row = m0 + mt * 16 + group + (c >= 2 ? 8 : 0);
            const int col = n0 + nt * 8 + 2 * tig + (c & 1);
            if (row >= M || col >= N) continue;
            const size_t at = static_cast<size_t>(row) * N + col;
            if (output_f32) static_cast<float*>(out)[at] = acc[i][c];
            else static_cast<__nv_bfloat16*>(out)[at] = __float2bfloat16_rn(acc[i][c]);
        }
    }
}

extern "C" __global__ void __launch_bounds__(256)
tf_glm53_fp8_mma(const uint8_t* xq, const float* xs, const uint8_t* w, const float* ws,
                 void* out, int M, int N, int K, int output_f32) {
    TF_VBLOCK_ENTER();
    extern __shared__ __align__(16) unsigned char smem[];
    tf_glm53_fp8_mma_task(xq, xs, w, ws, out, M, N, K, output_f32, smem);
}

// Optional MXFP8 activation variant: UE8M0 activation scales cover 32 inputs,
// while the weight scale still covers a 128x128 block.
__device__ __forceinline__ void tf_glm53_fp8_mma_mx_task(const uint8_t* __restrict__ xq, const uint8_t* __restrict__ xs,
                     const uint8_t* __restrict__ w, const float* __restrict__ ws,
                     void* __restrict__ out, int M, int N, int K, int output_f32) {
    const long long index = static_cast<long long>(TF_VBID.x) * TF_VBDIM.x + threadIdx.x;
    const long long total = static_cast<long long>(M) * N;
    if (index >= total) return;
    const int row = static_cast<int>(index / N), col = static_cast<int>(index % N);
    float acc = 0.0f;
    for (int g = 0; g < K / 32; ++g) {
        float partial = 0.0f;
        for (int t = 0; t < 32; ++t) {
            const int k = g * 32 + t;
            const float av = __half2float(__half(__nv_cvt_fp8_to_halfraw(xq[tf_glm53_xoff(row, k, K)], __NV_E4M3)));
            const float bv = __half2float(__half(__nv_cvt_fp8_to_halfraw(w[tf_glm53_woff(col, k, K)], __NV_E4M3)));
            partial = partial + av * bv;
        }
        const int exponent = static_cast<int>(xs[static_cast<long long>(row) * (K / 32) + g]) - 127;
        const float xscale = ldexpf(1.0f, exponent);
        const float wscale = ws[static_cast<long long>(col / 128) * (K / 128) + g / 4];
        acc = fmaf(partial, xscale * wscale, acc);
    }
    if (output_f32) static_cast<float*>(out)[index] = acc;
    else static_cast<__nv_bfloat16*>(out)[index] = __float2bfloat16_rn(acc);
}

extern "C" __global__ void __launch_bounds__(256)
tf_glm53_fp8_mma_mx(const uint8_t* xq, const uint8_t* xs, const uint8_t* w, const float* ws,
                    void* out, int M, int N, int K, int output_f32) {
    TF_VBLOCK_ENTER();
    tf_glm53_fp8_mma_mx_task(xq, xs, w, ws, out, M, N, K, output_f32);
}
