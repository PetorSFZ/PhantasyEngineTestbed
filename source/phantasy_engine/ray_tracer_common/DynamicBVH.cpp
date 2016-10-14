// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#include "phantasy_engine/ray_tracer_common/DynamicBVH.hpp"

#include "phantasy_engine/ray_tracer_common/BVHNode.hpp"

#include <sfz/math/MatrixSupport.hpp>

#include "phantasy_engine/ray_tracer_common/StaticBVHBuilder.hpp"

namespace phe {

using namespace sfz;

DynamicBVH createDynamicBvh(DynArray<BVH>& bvhs, DynArray<DynObject>& dynObjects)
{
	int nLeaves = dynObjects.size();

	DynArray<SphereNode> leaves;
	leaves.setCapacity(nLeaves);

	for (DynObject& obj : dynObjects) {
		uint32_t bvhIndex = obj.meshIndex;
		BVH& bvh = bvhs[bvhIndex];
		BVHNode& root = bvh.nodes[0];

		vec3 halfDiagonal = (root.rightChildAABBMax() - root.leftChildAABBMin()) / 2.0f;

		SphereNode leaf;
		leaf.radius = length(halfDiagonal);
		leaf.center = root.rightChildAABBMin + halfDiagonal;
		leaf.isLeaf = true;
		leaf.translation = obj.transform.columnAt(3).xyz;
		leaf.leftChildIndex = bvhIndex;

		leaves.add(leaf);
	}

	// Recursively construct tree

	return DynamicBVH();
}

}
