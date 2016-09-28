// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include <sfz/math/Vector.hpp>
#include <sfz/math/Matrix.hpp>

#include "phantasy_engine/ray_tracer_common/BVH.hpp"
#include "phantasy_engine/rendering/RawMesh.hpp"

namespace phe {

struct OuterBVHNode {
	bool leftIsLeaf, rightIsLeaf;
	uint32_t leftIndex, rightIndex;
	sfz::vec3 leftMin, leftMax, rightMin, rightMax;
};

struct OuterBVH {
	sfz::DynArray<OuterBVHNode> nodes;
	sfz::DynArray<BVH> bvhs;
};

BVH createDynamicBvh(const RawMesh& meshes, const sfz::mat4& transforms);
OuterBVH createOuterBvh(DynArray<BVH>& bvhs);

}
