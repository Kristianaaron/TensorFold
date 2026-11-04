// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
#include "../tf_vblock.cuh"
// Adapted from ashhart/TensorFold@d1018b31:zig/kernels/cuda/sample.cu (Apache-2.0).
// Changes: fp32 GLM-5.3 head reductions, deterministic local top-k and gate probe gathering.
#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>
#include <stdint.h>

namespace glm53_head {

__device__ __forceinline__ bool better(float a, unsigned ai, float b, unsigned bi) {
    return !isnan(a) && (isnan(b) || a > b || (a == b && ai < bi));
}

// One row per block. The first maximum wins; NaNs never win.
__device__ __forceinline__ void tf_glm53_top1_f32_task(const float* x, unsigned vocab, unsigned offset,
                                               float* value, unsigned* id) {
    const unsigned row = TF_VBID.x, t = threadIdx.x;
    __shared__ float sv[256];
    __shared__ unsigned si[256];
    float v = -INFINITY;
    unsigned ix = 0xffffffffu;
    for (unsigned i = t; i < vocab; i += TF_VBDIM.x) {
        const float z = x[(size_t)row * vocab + i];
        if (better(z, i, v, ix)) v = z, ix = i;
    }
    sv[t] = v; si[t] = ix;
    TF_TASK_SYNC();
    for (unsigned n = TF_VBDIM.x >> 1; n; n >>= 1) {
        if (t < n && better(sv[t + n], si[t + n], sv[t], si[t])) sv[t] = sv[t + n], si[t] = si[t + n];
        TF_TASK_SYNC();
    }
    if (t == 0) { value[row] = sv[0]; id[row] = si[0] == 0xffffffffu ? si[0] : si[0] + offset; }
}

extern "C" __global__ void tf_glm53_top1_f32(const float* x, unsigned vocab, unsigned offset,
                                               float* value, unsigned* id) {
    TF_VBLOCK_ENTER();
    tf_glm53_top1_f32_task(x, vocab, offset, value, id);
}

__device__ __forceinline__ unsigned ordered_float(float v) {
    unsigned bits = __float_as_uint(v);
    if ((bits & 0x7fffffffu) == 0) bits = 0; // Python equality treats both signed zeros as a tie.
    return (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
}

__device__ __forceinline__ unsigned long long candidate_key(float v, unsigned id) {
    const unsigned high = ~ordered_float(v); // ascending unsigned key means value descending
    return (static_cast<unsigned long long>(high) << 32) | id;
}

__device__ __forceinline__ float candidate_value(unsigned long long key) {
    const unsigned ordered = ~static_cast<unsigned>(key >> 32);
    const unsigned bits = (ordered & 0x80000000u) ? (ordered ^ 0x80000000u) : ~ordered;
    return __uint_as_float(bits);
}

// Fill every requested output slot before the merge. Non-NaN candidates are
// overwritten by their exact rank; short lists retain Python's sort sentinel.
__device__ __forceinline__ void tf_glm53_cand_fill_f32_task(float* values, unsigned* ids, unsigned count) {
    const unsigned i = TF_VBID.x * TF_VBDIM.x + threadIdx.x;
    if (i < count) values[i] = -CUDART_INF_F, ids[i] = 0xffffffffu;
}

extern "C" __global__ void tf_glm53_cand_fill_f32(float* values, unsigned* ids, unsigned count) {
    TF_VBLOCK_ENTER();
    tf_glm53_cand_fill_f32_task(values, ids, count);
}

// Stage one: each CTA selects up to k entries from its shard. A global top-k can
// contain all k entries from one shard, so each local list must retain k keys.
__device__ __forceinline__ void tf_glm53_cand_part_f32_task(const float* x, unsigned vocab, unsigned offset, unsigned k,
                                                   unsigned long long* scratch, char* smem) {
    const unsigned row = TF_VBID.x, part = TF_VBID.y, t = threadIdx.x;
    if (k > 2048 || k == 0) return;
    const unsigned begin = (unsigned long long)vocab * part / 32;
    const unsigned end = (unsigned long long)vocab * (part + 1) / 32;
    const unsigned local_want = min(k, (unsigned)(end - begin));
    unsigned* hist = reinterpret_cast<unsigned*>(smem);
    unsigned* scan = hist + 256;
    unsigned* tmp = scan + 256;
    unsigned long long* selected = reinterpret_cast<unsigned long long*>(tmp + 256);
    unsigned long long* scalars = selected + 2048;
    unsigned long long& prefix = scalars[0];
    unsigned* state = reinterpret_cast<unsigned*>(scalars + 1);
    unsigned& target = state[0];
    unsigned& want = state[1];
    unsigned& chosen = state[2];
    unsigned& count = state[3];
    unsigned& valid = state[4];
    for (unsigned i = t; i < 2048; i += TF_VBDIM.x) selected[i] = ~0ull;
    if (t == 0) { prefix = 0; target = local_want - 1; want = local_want; valid = 1; }
    TF_TASK_SYNC();
    for (unsigned pass = 0; pass < 8; ++pass) {
        const unsigned long long mask = pass == 0 ? 0ull : (~0ull << (64 - pass * 8));
        if (t < 256) hist[t] = 0;
        TF_TASK_SYNC();
        const unsigned shift = 56 - pass * 8;
        for (unsigned i = begin + t; i < end; i += TF_VBDIM.x) {
            const float v = x[(size_t)row * vocab + i];
            if (isnan(v)) continue;
            const unsigned long long key = candidate_key(v, offset + i);
            if ((key & mask) == prefix) atomicAdd(&hist[(key >> shift) & 255], 1u);
        }
        TF_TASK_SYNC();
        if (t < 256) scan[t] = hist[t];
        TF_TASK_SYNC();
        for (unsigned step = 1; step < 256; step <<= 1) {
            if (t < 256) tmp[t] = scan[t] + (t >= step ? scan[t - step] : 0);
            TF_TASK_SYNC();
            if (t < 256) scan[t] = tmp[t];
            TF_TASK_SYNC();
        }
        if (pass == 0 && t == 0) { want = min(local_want, scan[255]); target = want ? want - 1 : 0; valid = want != 0; }
        TF_TASK_SYNC();
        if (valid) {
            if (t == 0) chosen = 256;
            TF_TASK_SYNC();
            if (t < 256 && scan[t] > target && (t == 0 || scan[t - 1] <= target)) atomicMin(&chosen, t);
            TF_TASK_SYNC();
            if (t == 0) {
                const unsigned before = chosen == 0 ? 0 : scan[chosen - 1];
                target -= before;
                prefix |= (unsigned long long)chosen << shift;
            }
            TF_TASK_SYNC();
        }
    }
    if (t == 0) count = 0;
    TF_TASK_SYNC();
    for (unsigned i = begin + t; i < end; i += TF_VBDIM.x) {
        const float v = x[(size_t)row * vocab + i];
        if (isnan(v)) continue;
        const unsigned long long key = candidate_key(v, offset + i);
        if (key <= prefix) {
            const unsigned slot = atomicAdd(&count, 1u);
            if (slot < local_want) selected[slot] = key;
        }
    }
    TF_TASK_SYNC();
    for (unsigned width = 2; width <= 2048; width <<= 1) {
        for (unsigned stride = width >> 1; stride; stride >>= 1) {
            for (unsigned i = t; i < 2048; i += TF_VBDIM.x) {
                const unsigned partner = i ^ stride;
                if (partner > i) {
                    const bool ascending = (i & width) == 0;
                    const unsigned long long a = selected[i], b = selected[partner];
                    if ((ascending && a > b) || (!ascending && a < b)) selected[i] = b, selected[partner] = a;
                }
            }
            TF_TASK_SYNC();
        }
    }
    for (unsigned i = t; i < local_want; i += TF_VBDIM.x) scratch[((size_t)row * 32 + part) * k + i] = selected[i];
}

extern "C" __global__ void tf_glm53_cand_part_f32(const float* x, unsigned vocab, unsigned offset, unsigned k,
                                                   unsigned long long* scratch) {
    TF_VBLOCK_ENTER();
    extern __shared__ __align__(16) char smem[];
    tf_glm53_cand_part_f32_task(x, vocab, offset, k, scratch, smem);
}

// Stage two: rank each retained local key by counting smaller keys in the 32
// sorted lists. Unique token ids make ranks unique; output ranks below k are exact.
__device__ __forceinline__ void tf_glm53_cand_merge_f32_task(const unsigned long long* scratch, unsigned vocab, unsigned k,
                                                    float* values, unsigned* ids, char* smem) {
    const unsigned row = TF_VBID.x, part = TF_VBID.y, t = threadIdx.x;
    if (k > 2048 || k == 0) return;
    const unsigned long long* lists = scratch + (size_t)row * 32 * k;
    const unsigned long long part_len = (unsigned long long)vocab * (part + 1) / 32 - (unsigned long long)vocab * part / 32;
    const unsigned local_count = min(k, (unsigned)part_len);
    unsigned long long* staged = reinterpret_cast<unsigned long long*>(smem);
    unsigned long long keys[8];
    unsigned ranks[8];
    bool active[8];
#pragma unroll
    for (unsigned j = 0; j < 8; ++j) {
        const unsigned i = t + j * TF_VBDIM.x;
        keys[j] = i < local_count ? lists[(size_t)part * k + i] : ~0ull;
        ranks[j] = i;
        active[j] = i < local_count && keys[j] != ~0ull;
    }
    for (unsigned q = 0; q < 32; ++q) {
        if (q == part) continue;
        const unsigned long long q_len = (unsigned long long)vocab * (q + 1) / 32 - (unsigned long long)vocab * q / 32;
        const unsigned q_count = min(k, (unsigned)q_len);
        const unsigned long long* list = lists + (size_t)q * k;
        for (unsigned j = t; j < q_count; j += TF_VBDIM.x) staged[j] = list[j];
        TF_TASK_SYNC();
#pragma unroll
        for (unsigned j = 0; j < 8; ++j) {
            if (!active[j]) continue;
            unsigned lo = 0, hi = q_count;
            while (lo < hi) {
                const unsigned mid = lo + (hi - lo) / 2;
                if (staged[mid] < keys[j]) lo = mid + 1;
                else hi = mid;
            }
            ranks[j] += lo;
            active[j] = ranks[j] < k;
        }
        TF_TASK_SYNC();
    }
#pragma unroll
    for (unsigned j = 0; j < 8; ++j) if (active[j] && ranks[j] < k) {
        values[(size_t)row * k + ranks[j]] = candidate_value(keys[j]);
        ids[(size_t)row * k + ranks[j]] = static_cast<unsigned>(keys[j]);
    }
}

extern "C" __global__ void tf_glm53_cand_merge_f32(const unsigned long long* scratch, unsigned vocab, unsigned k,
                                                    float* values, unsigned* ids) {
    TF_VBLOCK_ENTER();
    extern __shared__ __align__(16) char smem[];
    tf_glm53_cand_merge_f32_task(scratch, vocab, k, values, ids, smem);
}

// Stable fp32 local LSE partial [max,sum], merged across ranks in sampler.zig.
__device__ __forceinline__ void tf_glm53_lse_f32_task(const float* x, unsigned vocab, float* partial) {
    const unsigned row = TF_VBID.x, t = threadIdx.x;
    __shared__ float sv[256];
    float m = -INFINITY;
    for (unsigned i = t; i < vocab; i += TF_VBDIM.x) m = fmaxf(m, x[(size_t)row * vocab + i]);
    sv[t] = m;
    TF_TASK_SYNC();
    for (unsigned n = TF_VBDIM.x >> 1; n; n >>= 1) {
        if (t < n) sv[t] = fmaxf(sv[t], sv[t + n]);
        TF_TASK_SYNC();
    }
    const float mx = sv[0];
    if (mx == -CUDART_INF_F) {
        if (t == 0) partial[(size_t)row * 2] = mx, partial[(size_t)row * 2 + 1] = 0.f;
        return;
    }
    float z = 0.f;
    for (unsigned i = t; i < vocab; i += TF_VBDIM.x) z += expf(x[(size_t)row * vocab + i] - mx);
    sv[t] = z;
    TF_TASK_SYNC();
    for (unsigned n = TF_VBDIM.x >> 1; n; n >>= 1) {
        if (t < n) sv[t] += sv[t + n];
        TF_TASK_SYNC();
    }
    if (t == 0) partial[(size_t)row * 2] = mx, partial[(size_t)row * 2 + 1] = sv[0];
}

extern "C" __global__ void tf_glm53_lse_f32(const float* x, unsigned vocab, float* partial) {
    TF_VBLOCK_ENTER();
    tf_glm53_lse_f32_task(x, vocab, partial);
}

// Structured output: a row's tokens outside its grammar bits become -inf before top-1, LSE or candidates.
// bits: rows x words u32 over the whole vocabulary; offset: this rank's first token.
__device__ __forceinline__ void tf_glm53_mask_f32_task(float* x, unsigned vocab, unsigned offset, const unsigned* bits, unsigned words) {
    const unsigned row = TF_VBID.x;
    const unsigned* b = bits + (size_t)row * words;
    for (unsigned i = threadIdx.x; i < vocab; i += TF_VBDIM.x) {
        const unsigned t = offset + i, w = t >> 5;
        if (w >= words || ((b[w] >> (t & 31u)) & 1u) == 0u) x[(size_t)row * vocab + i] = -INFINITY;
    }
}
extern "C" __global__ void tf_glm53_mask_f32(float* x, unsigned vocab, unsigned offset, const unsigned* bits, unsigned words) {
    TF_VBLOCK_ENTER();
    tf_glm53_mask_f32_task(x, vocab, offset, bits, words);
}

// Gather probe logits for a gate row from this rank's vocabulary shard.
__device__ __forceinline__ void tf_glm53_gate_gather_f32_task(const float* logits, const unsigned* ids,
                                                       unsigned vocab, unsigned offset, float* gathered,
                                                       unsigned probes, unsigned rows) {
    const unsigned i = TF_VBID.x * TF_VBDIM.x + threadIdx.x;
    if (i >= rows * probes) return;
    const unsigned id = ids[i], local = id - offset;
    gathered[i] = (id >= offset && local < vocab) ? logits[(size_t)(i / probes) * vocab + local] : 0.f;
}

extern "C" __global__ void tf_glm53_gate_gather_f32(const float* logits, const unsigned* ids,
                                                      unsigned vocab, unsigned offset, float* gathered,
                                                      unsigned probes, unsigned rows) {
    TF_VBLOCK_ENTER();
    tf_glm53_gate_gather_f32_task(logits, ids, vocab, offset, gathered, probes, rows);
}

// Python forward.py gate mass is clamp(1 - sum(exp(logp)), 0).
__device__ __forceinline__ void tf_glm53_gate_mass_f32_task(const float* probe_logits, const float* lse,
                                                    float* logp, float* mass_outside,
                                                    unsigned probes, unsigned rows) {
    const unsigned row = TF_VBID.x;
    if (row >= rows || threadIdx.x != 0) return;
    float sum = 0.f;
    for (unsigned j = 0; j < probes; ++j) {
        const float lp = probe_logits[(size_t)row * probes + j] - lse[row];
        logp[(size_t)row * probes + j] = lp;
        sum += expf(lp);
    }
    mass_outside[row] = fmaxf(0.f, 1.f - sum);
}

extern "C" __global__ void tf_glm53_gate_mass_f32(const float* probe_logits, const float* lse,
                                                    float* logp, float* mass_outside,
                                                    unsigned probes, unsigned rows) {
    TF_VBLOCK_ENTER();
    tf_glm53_gate_mass_f32_task(probe_logits, lse, logp, mass_outside, probes, rows);
}

} // namespace glm53_head
