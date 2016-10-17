// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#include "phantasy_engine/ray_tracer_common/DynamicBVH.hpp"

#include "phantasy_engine/ray_tracer_common/BVHNode.hpp"

#include <sfz/math/MatrixSupport.hpp>
#include <sfz/geometry/AABB.hpp>

#include "phantasy_engine/ray_tracer_common/StaticBVHBuilder.hpp"

namespace phe {

using namespace sfz;

struct NodeContainer {
	vec3 min, max;
	int nodeIndex;
};

static void fillNode(DynamicBVH& bvh, AABB& aabb, DynArray<NodeContainer>& leaves, const DynArray<LeafData>& leafDatas)
{
	int index = bvh.nodes.size();
	if (leaves.size() <= 2) {
		const NodeContainer& leaf1 = leaves.first();
		const NodeContainer& leaf2 = leaves.last();

		BVHNode node;
		node.setLeftChildAABB(leaf1.min, leaf1.max);
		node.setRightChildAABB(leaf2.min, leaf2.max);
		node.setLeftChildLeaf(leaf1.nodeIndex, 1);
		node.setRightChildLeaf(leaf2.nodeIndex, 1);
		bvh.nodes.add(node);
		return;
	}

	bvh.nodes.add(BVHNode());
	BVHNode& root = bvh.nodes[bvh.nodes.size()-1];

	const int nLeaves = leaves.size();

	const vec3 diagonal = aabb.extents();
	const int splitAxis = diagonal.y > diagonal.x ? (diagonal.z > diagonal.y ? 2 : 1) : 0;

	// Sort along split axis
	for (int i = 0; i < nLeaves; i++) {
		int currIndex = i;
		while (currIndex > 0 && leaves[currIndex].min[splitAxis] < leaves[currIndex-1].min[splitAxis]) {
			std::swap(leaves[currIndex], leaves[--currIndex]);
		}
	}

	// Decide which leaves to put in which child
	const int nRightLeaves = nLeaves / 2;
	const int nLeftLeaves = nLeaves - nRightLeaves;

	DynArray<NodeContainer> rightLeaves(0, nRightLeaves);
	DynArray<NodeContainer> leftLeaves(0, nLeftLeaves);
	
	for (int i = 0; i < nRightLeaves; i++) {
		rightLeaves.add(std::move(leaves[i]));
	}
	for (int i = nRightLeaves; i < nLeaves; i++) {
		leftLeaves.add(std::move(leaves[i]));
	}

	if (nRightLeaves == 1) {
		const NodeContainer& child = rightLeaves.first();
		root.setRightChildAABB(child.min, child.max);
		root.setRightChildLeaf(child.nodeIndex, 1);
	}
	else {
		AABB childAabb;
		childAabb.min = vec3{ INFINITY };
		childAabb.max = vec3{ -INFINITY };
		for (const NodeContainer& leaf : rightLeaves) {
			childAabb.min = min(childAabb.min, leaf.min);
			childAabb.max = max(childAabb.max, leaf.max);
		}

		root.setRightChildAABB(childAabb.min, childAabb.max);
		root.setRightChildInner(bvh.nodes.size());
		fillNode(bvh, childAabb, rightLeaves, leafDatas);
	}

	if (nLeftLeaves == 1) {
		const NodeContainer& child = leftLeaves.first();
		root.setLeftChildAABB(child.min, child.max);
		root.setLeftChildLeaf(child.nodeIndex, 1);
	}
	else {
		AABB childAabb;
		childAabb.min = vec3{ INFINITY };
		childAabb.max = vec3{ -INFINITY };
		for (const NodeContainer& leaf : leftLeaves) {
			childAabb.min = min(childAabb.min, leaf.min);
			childAabb.max = max(childAabb.max, leaf.max);
		}

		root.setLeftChildAABB(childAabb.min, childAabb.max);
		root.setLeftChildInner(bvh.nodes.size());
		fillNode(bvh, childAabb, leftLeaves, leafDatas);
	}
}

DynamicBVH createDynamicBvh(const DynArray<BVH>& bvhs, const DynArray<DynObject>& dynObjects)
{
	int nLeaves = dynObjects.size();

	DynArray<LeafData> leafDatas(0, nLeaves);
	DynArray<BVHNode> leafNodes(0, nLeaves);
	DynArray<NodeContainer> leaves(0, nLeaves);
	
	AABB rootAabb;
	rootAabb.min = vec3{ INFINITY };
	rootAabb.max = vec3{ -INFINITY };

	for (int i = 0; i < nLeaves; i++) {
		const DynObject& obj = dynObjects[i];
		
		uint32_t bvhIndex = obj.meshIndex;
		const BVH& bvh = bvhs[bvhIndex];
		const BVHNode& root = bvh.nodes[0];

		LeafData leaf;
		leaf.translation = obj.transform.columnAt(3).xyz;
		leaf.bvhIndex = bvhIndex;
		leafDatas.add(leaf);

		NodeContainer container;
		container.nodeIndex = i;
		container.min = min(root.leftChildAABBMin(), root.rightChildAABBMin()) + leafDatas[i].translation;
		container.max = max(root.leftChildAABBMax(), root.rightChildAABBMax()) + leafDatas[i].translation;
		leaves.add(container);

		rootAabb.min = min(container.min, rootAabb.min);
		rootAabb.max = max(container.max, rootAabb.max);
	}

	// Recursively construct tree
	DynamicBVH bvh;
	bvh.leaves = std::move(leafDatas);
	bvh.nodes.setCapacity(nLeaves);
	fillNode(bvh, rootAabb, leaves, leafDatas);
	return bvh;
}

}
