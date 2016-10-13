// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include "phantasy_engine/level/StaticScene.hpp"
#include "phantasy_engine/ray_tracer_common/BVH.hpp"

namespace phe {

BVH buildStaticBVH(RawMesh& mesh) noexcept;

void buildStaticBVH(StaticScene& scene) noexcept;

BVH buildStaticFrom(const DynArray<TriangleVertices>& triangles,
                    const DynArray<TriangleData>& triangleDatas) noexcept;

// Fixes a few potential issues with the BVH, including:
// * (Potentially) improves cache locality
// * Ensures last triangle in a leaf has the necessary end marker in the padding
// * Ensures all leaf indices are bitwise negated (~)
void sanitizeBVH(BVH& bvh) noexcept;

} // namespace phe
