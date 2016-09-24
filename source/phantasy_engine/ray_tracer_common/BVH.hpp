// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include <sfz/containers/DynArray.hpp>

#include "phantasy_engine/ray_tracer_common/BVHNode.hpp"
#include "phantasy_engine/ray_tracer_common/Triangle.hpp"

#pragma once

namespace phe {

using sfz::DynArray;

// C++ container
// ------------------------------------------------------------------------------------------------

struct BVH final {
	// The first node (index 0) is the root node for the whole BVH
	DynArray<BVHNode> nodes;

	// These arrays are supposed to be the same size, an index is valid in both lists
	DynArray<TriangleVertices> triangleVerts;
	DynArray<TriangleData> triangleDatas;
};

} // namespace phe
