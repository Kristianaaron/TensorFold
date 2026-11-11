// SPDX-License-Identifier: Apache-2.0
// Modified 2026 by Kristian Aaron (Krishando). Original: ashhart/TensorFold@d1018b318 (Apache-2.0).
// Bench-only (qmmf2): schedule variants of the FP8B qmmf kernel, same bits. Every variant is the release body
// tf_qmmf::qmmf_task with other template arguments (row tile, warps a CTA, ring depth), so the arithmetic is the
// release code and each output element keeps its one accumulator chain per K slice:
//   release : BM 64, 4 warps a CTA, 4 stages, 49,152 B, 2 CTAs an SM (8 warps an SM)
//   s2      : BM 64, 4 warps, 2 stages, 24,576 B, 4 CTAs an SM (16 warps an SM)
//   w2      : BM 64, 8 warps (32 rows a warp), 4 stages, 49,152 B, 2 CTAs an SM (16 warps an SM)
//   w2s2    : BM 64, 8 warps, 2 stages, 24,576 B, 4 CTAs an SM (32 warps an SM)
//   b128    : BM 128, 8 warps (64 rows a warp), 2 stages, 40,960 B, 2 CTAs an SM (16 warps an SM, half the weight loads)
// Entries per variant: plain (_p_, split-K 1) and fused (_f_, split-K > 1), bf16 and f32 output, and the fused group.
#include "qmmf.cu"

namespace tf_qmmf {

static_assert(Tile<FP8B, 64, 64, 1, 4, 2>::SMEM == 24576, "s2 shared bytes");
static_assert(Tile<FP8B, 64, 64, 2, 4, 4>::SMEM == 49152, "w2 shared bytes");
static_assert(Tile<FP8B, 64, 64, 2, 4, 2>::SMEM == 24576, "w2s2 shared bytes");
static_assert(Tile<FP8B, 128, 64, 2, 4, 2>::SMEM == 40960, "b128 shared bytes");

template <int BM, int WM, int ST>
__device__ __forceinline__ void qmmf2_group(const GroupArgs& a, unsigned char* buf) {
    int s = 0, local = static_cast<int>(blockIdx.x);
    while (s + 1 < a.count && local >= a.seg[s].blocks) {
        local -= a.seg[s].blocks;
        ++s;
    }
    const GroupSeg& g = a.seg[s];
    tf_vblock_set(make_uint3(local, 0, blockIdx.z), make_uint3(g.blocks, 1, gridDim.z), blockDim);
    qmmf_task<FP8B, BM, 64, WM, 4, ST, false, false, true>(a.x, g.w, g.bs, g.scale, g.out, nullptr, a.m, g.n, a.k, g.sk,
                                                           g.npad, a.ldx, a.band, buf);
    if (a.prefetch_bytes > 0) {
        const long long blocks = static_cast<long long>(gridDim.x) * gridDim.z;
        const long long id = static_cast<long long>(blockIdx.z) * gridDim.x + blockIdx.x;
        const long long lines = (a.prefetch_bytes + 127) / 128, per = (lines + blocks - 1) / blocks;
        const long long lo = id * per, hi = lines < lo + per ? lines : lo + per;
        for (long long l = lo + threadIdx.x; l < hi; l += blockDim.x)
            asm volatile("prefetch.global.L2::evict_last [%0];" : : "l"(a.prefetch + l * 128));
    }
}

#define QMMF2_ENTRY(NAME, BM, WM, ST, CTAS, F32_, FUSE_) \
extern "C" __global__ void __launch_bounds__(WM * 128, CTAS) NAME( \
        const __nv_bfloat16* __restrict__ x, const unsigned char* __restrict__ w, const uint8_t* __restrict__ bs, \
        float scale, void* __restrict__ out, float* __restrict__ part, int M, int N, int K, int SK, int npad, int ldx, \
        int group) { \
    TF_VBLOCK_ENTER(); \
    extern __shared__ __align__(128) unsigned char buf[]; \
    qmmf_task<FP8B, BM, 64, WM, 4, ST, F32_, false, FUSE_>(x, w, bs, scale, out, part, M, N, K, SK, npad, ldx, group, \
                                                           buf); \
}
#define QMMF2_VARIANT(V, BM, WM, ST, CTAS) \
QMMF2_ENTRY(tf_qmmf2_##V##_p_b16, BM, WM, ST, CTAS, false, false) \
QMMF2_ENTRY(tf_qmmf2_##V##_p_f32, BM, WM, ST, CTAS, true, false) \
QMMF2_ENTRY(tf_qmmf2_##V##_f_b16, BM, WM, ST, CTAS, false, true) \
QMMF2_ENTRY(tf_qmmf2_##V##_f_f32, BM, WM, ST, CTAS, true, true) \
extern "C" __global__ void __launch_bounds__(WM * 128, CTAS) tf_qmmf2_group_##V(const GroupArgs a) { \
    extern __shared__ __align__(128) unsigned char buf[]; \
    qmmf2_group<BM, WM, ST>(a, buf); \
}

QMMF2_VARIANT(s2, 64, 1, 2, 4)
QMMF2_VARIANT(w2, 64, 2, 4, 2)
QMMF2_VARIANT(w2s2, 64, 2, 2, 4)
QMMF2_VARIANT(b128, 128, 2, 2, 2)

}
