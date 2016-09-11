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

// C++ container
// ------------------------------------------------------------------------------------------------

class BVH final {
public:

	// Members
	// --------------------------------------------------------------------------------------------

	DynArray<BVHNode> nodes;

	// These arrays are supposed to be the same size, an index is valid in both lists
	DynArray<TriangleVertices> triangles;
	DynArray<TriangleData> triangleDatas;

	// Methods
	// --------------------------------------------------------------------------------------------

	void buildStaticFrom(const StaticScene& scene) noexcept;
	void buildStaticFrom(const DynArray<TriangleVertices>& triangles) noexcept;

private:

	// Private methods
	// --------------------------------------------------------------------------------------------

	void fillStaticNode(
		uint32_t nodeInd,
		uint32_t depth,
		const DynArray<uint32_t>& triangleInds,
		const DynArray<TriangleVertices>& inTriangles,
		const DynArray<sfz::AABB>& inTriangleAabbs) noexcept;

};

} // namespace phe
