// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include <cstdint>

#include <sfz/containers/DynArray.hpp>
#include <sfz/geometry/AABB.hpp>
#include <sfz/math/Vector.hpp>

#include "phantasy_engine/level/StaticScene.hpp"

#include "phantasy_engine/ray_tracer_common/BVHNode.hpp"
#include "phantasy_engine/ray_tracer_common/Triangle.hpp"

#pragma once

namespace phe {

using sfz::DynArray;
using sfz::vec3;
using sfz::vec3i;

// C++ container
// ------------------------------------------------------------------------------------------------

struct BVH final {
	DynArray<BVHNode> nodes;

	// These arrays are supposed to be the same size, an index is valid in both lists
	DynArray<TriangleVertices> triangles;
	DynArray<TriangleData> triangleDatas;

	// The distance from the (inclusive) root node to the "deepest" leaf node
	uint32_t maxDepth = ~0u;
};

BVH buildStaticFrom(const StaticScene& scene) noexcept;
BVH buildStaticFrom(const DynArray<TriangleVertices>& triangles,
                    const DynArray<TriangleData>& triangleDatas) noexcept;

} // namespace phe
