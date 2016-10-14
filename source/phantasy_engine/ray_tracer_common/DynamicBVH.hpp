// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include <sfz/math/Vector.hpp>
#include <sfz/math/Matrix.hpp>

#include "phantasy_engine/ray_tracer_common/BVH.hpp"
#include "phantasy_engine/rendering/RawMesh.hpp"
#include "phantasy_engine/level/DynObject.hpp"

namespace phe {

struct SphereNode {
	vec3 center;
	float radius;
	bool isLeaf;
	uint32_t leftChildIndex, rightChildIndex;
	vec3 translation;
};

struct DynamicBVH {
	DynArray<SphereNode> nodes;
	DynArray<uint32_t> leaves;
};

DynamicBVH createDynamicBvh(DynArray<BVH>& bvhs, DynArray<DynObject>& dynObjects);

}
