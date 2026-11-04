// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
//! Force analysis of the Spark-only RoCE all-reduce benchmark on a host without CUDA.
const std = @import("std");
const benchmark = @import("roce_allreduce_benchmark");

test {
    std.testing.refAllDecls(benchmark);
}
