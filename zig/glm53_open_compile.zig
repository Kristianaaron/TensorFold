// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
//! Compile-time root that forces the GLM-5.3 native open declarations to resolve.
const glm53 = @import("glm53");

comptime {
    _ = glm53.open;
    _ = glm53.snap_grid_fn;
    _ = glm53.model_type;
}
