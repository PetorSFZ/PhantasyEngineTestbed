// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#include "phantasy_engine/ray_tracer_common/DynamicBVH.hpp"

#include "phantasy_engine/ray_tracer_common/BVHNode.hpp"

#include <sfz/math/MatrixSupport.hpp>

#include "phantasy_engine/ray_tracer_common/StaticBVHBuilder.hpp"

namespace phe {

using namespace sfz;

static void fillNode(DynamicBVH& bvh, DynArray<DynNode>& leafNodes, const DynArray<LeafData>& leafDatas)
{
	if (leafNodes.size() == 1) {
		bvh.nodes.add(std::move(leafNodes[0]));
		return;
	}

	bvh.nodes.add(DynNode());
	DynNode& root = bvh.nodes[bvh.nodes.size()-1];

	// Expand AABB to precisely contain children
	for (const DynNode& leaf : leafNodes) {
		root.min = min(root.min, leaf.min);
		root.max = max(root.max, leaf.max);
	}

	const int nLeaves = leafNodes.size();

	const vec3 diagonal = root.max - root.min;
	const int splitAxis = diagonal.y > diagonal.x ? (diagonal.z > diagonal.y ? 2 : 1) : 0;

	// Sort along split axis
	for (int i = 0; i < nLeaves; i++) {
		int currIndex = i;
		while (currIndex > 0 && leafNodes[currIndex].min[splitAxis] < leafNodes[currIndex-1].min[splitAxis]) {
			std::swap(leafNodes[currIndex], leafNodes[--currIndex]);
		}
	}

	// Decide which leaves to put in which child
	const int nRightLeaves = nLeaves / 2;
	const int nLeftLeaves = nLeaves - nRightLeaves;

	DynArray<DynNode> rightLeaves;
	DynArray<DynNode> leftLeaves;
	rightLeaves.setCapacity(nRightLeaves);
	leftLeaves.setCapacity(nLeftLeaves);

	for (int i = 0; i < nRightLeaves; i++) {
		rightLeaves.add(std::move(leafNodes[i]));
	}
	for (int i = nRightLeaves; i < nLeaves; i++) {
		leftLeaves.add(std::move(leafNodes[i]));
	}

	// Recursively fill children
	root.rightChildIndex = bvh.nodes.size();
	fillNode(bvh, rightLeaves, leafDatas);
	root.leftChildIndex = bvh.nodes.size();
	fillNode(bvh, leftLeaves, leafDatas);
}

DynamicBVH createDynamicBvh(const DynArray<BVH>& bvhs, const DynArray<DynObject>& dynObjects)
{
	int nLeaves = dynObjects.size();

	DynArray<LeafData> leafDatas;
	DynArray<DynNode> leafNodes;
	leafDatas.setCapacity(nLeaves);
	leafNodes.setCapacity(nLeaves);

	for (int i = 0; i < nLeaves; i++) {
		const DynObject& obj = dynObjects[i];
		
		uint32_t bvhIndex = obj.meshIndex;
		const BVH& bvh = bvhs[bvhIndex];
		const BVHNode& root = bvh.nodes[0];

		LeafData leaf;
		leaf.translation = obj.transform.columnAt(3).xyz;
		leaf.bvhIndex = bvhIndex;
		leafDatas.add(leaf);

		DynNode node;
		node.isLeaf = true;
		node.leftChildIndex = i;
		node.min = min(root.leftChildAABBMin(), root.rightChildAABBMin()) + leafDatas[i].translation;
		node.max = max(root.leftChildAABBMax(), root.rightChildAABBMax()) + leafDatas[i].translation;
		leafNodes.add(node);
	}

	// Recursively construct tree
	DynamicBVH bvh;
	bvh.leaves = std::move(leafDatas);
	bvh.nodes.setCapacity(2*nLeaves-1);
	fillNode(bvh, leafNodes, leafDatas);
	return bvh;
}

}
