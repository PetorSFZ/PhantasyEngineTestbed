// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include "phantasy_engine/level/StaticScene.hpp"
#include "phantasy_engine/ray_tracer_common/BVH.hpp"

namespace phe {

void buildStaticBVH(StaticScene& scene) noexcept;

BVH buildStaticFrom(const DynArray<TriangleVertices>& triangles,
                    const DynArray<TriangleData>& triangleDatas) noexcept;

} // namespace phe
