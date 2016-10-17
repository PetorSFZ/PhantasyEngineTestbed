// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include <sfz/math/Vector.hpp>
#include <sfz/math/Matrix.hpp>

#include "phantasy_engine/ray_tracer_common/BVH.hpp"
#include "phantasy_engine/rendering/RawMesh.hpp"
#include "phantasy_engine/level/DynObject.hpp"

namespace phe {

struct LeafData {
	uint32_t bvhIndex;
	vec3 translation;
};

struct DynNode {
	vec3 min, max;
	bool isLeaf = false;
	uint32_t leftChildIndex, rightChildIndex;
};

struct DynamicBVH {
	DynArray<DynNode> nodes;
	DynArray<LeafData> leaves;
};

DynamicBVH createDynamicBvh(const DynArray<BVH>& bvhs, const DynArray<DynObject>& dynObjects);

}
