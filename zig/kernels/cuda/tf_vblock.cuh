// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
#pragma once
#include <cuda_runtime.h>
#include <assert.h>

// Virtual block coordinates: ordinary kernels mirror blockIdx/gridDim; the megakernel sets them per task.
__device__ __forceinline__ uint3& tf_vbid_ref() {
    __shared__ uint3 v;
    return v;
}
__device__ __forceinline__ uint3& tf_vnblk_ref() {
    __shared__ uint3 v;
    return v;
}
__device__ __forceinline__ uint3& tf_vbdim_ref() {
    __shared__ uint3 v;
    return v;
}
#define TF_VBID (tf_vbid_ref())
#define TF_VNBLK (tf_vnblk_ref())
#define TF_VBDIM (tf_vbdim_ref())
// Every thread stores the same value, then reads its own store: no barrier needed in ordinary kernels.
#define TF_VBLOCK_ENTER()                                                                                              \
    do {                                                                                                               \
        tf_vbid_ref() = blockIdx;                                                                                      \
        tf_vnblk_ref() = gridDim;                                                                                      \
        tf_vbdim_ref() = blockDim;                                                                                     \
    } while (0)
__device__ __forceinline__ unsigned tf_vthreads() {
    const uint3 d = TF_VBDIM;
    return d.x * d.y * d.z;
}
__device__ __forceinline__ unsigned tf_vthread() {
    const uint3 d = TF_VBDIM;
    return threadIdx.x + d.x * (threadIdx.y + d.y * threadIdx.z);
}
#ifndef NDEBUG
#define TF_VBLOCK_ASSERT_WARP(count) assert(((count) & 31u) == 0u)
#else
#define TF_VBLOCK_ASSERT_WARP(count) ((void)0)
#endif
#define TF_TASK_SYNC()                                                                                                 \
    do {                                                                                                               \
        const unsigned tf_task_threads = tf_vthreads();                                                                \
        TF_VBLOCK_ASSERT_WARP(tf_task_threads);                                                                        \
        if (tf_vthread() < tf_task_threads)                                                                            \
            asm volatile("bar.sync 1, %0;" :: "r"(tf_task_threads) : "memory");                                       \
    } while (0)
__device__ __forceinline__ void tf_vblock_set(uint3 b, uint3 n, uint3 d) {
    tf_vbid_ref() = b;
    tf_vnblk_ref() = n;
    tf_vbdim_ref() = d;
}
