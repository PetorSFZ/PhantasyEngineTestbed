// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#include <sfz/containers/DynArray.hpp>

#include "phantasy_engine/ray_tracer_common/BVHSanitizer.hpp"

namespace phe {

// Statics
// ------------------------------------------------------------------------------------------------

static int32_t sanitizeInternal(BVH& bvh, int32_t oldNodeIndex, DynArray<BVHNode>& newNodes) noexcept
{
	int32_t newNodeIndex = int32_t(newNodes.size());
	newNodes.add(bvh.nodes[oldNodeIndex]);
	BVHNode& newNode = newNodes[newNodeIndex];

	// Process left child
	if (newNode.leftChildIsLeaf()) {

		// Makes sure triangle index is bitwise negated
		if (newNode.leftChildIndex() >= 0) {
			newNode.setLeftChildLeaf(~newNode.leftChildIndex(), newNode.leftChildNumTriangles());
		}
		
		// Sets padding to 0 in all non-last triangles in leaf
		int32_t firstTriIndex = ~newNode.leftChildIndex();
		int32_t lastTriIndex = firstTriIndex + newNode.leftChildNumTriangles() - 1;
		for (int32_t i = firstTriIndex; i < lastTriIndex; i++) {
			bvh.triangles[i].v0.w = 0.0f;
			bvh.triangles[i].v1.w = 0.0f;
			bvh.triangles[i].v2.w = 0.0f;
		}
		
		// Sets padding to -1.0f in last triangle in leaf (end marker)
		bvh.triangles[lastTriIndex].v0.w = -1.0f;
		bvh.triangles[lastTriIndex].v1.w = -1.0f;
		bvh.triangles[lastTriIndex].v2.w = -1.0f;
	}
	else {
		newNode.setLeftChildInner(sanitizeInternal(bvh, newNode.leftChildIndex(), newNodes));
	}

	// Process right child
	if (newNode.rightChildIsLeaf()) {
		
		// Makes sure triangle index is bitwise negated
		if (newNode.rightChildIndex() >= 0) {
			newNode.setRightChildLeaf(~newNode.rightChildIndex(), newNode.rightChildNumTriangles());
		}
		
		// Sets padding to 0 in all non-last triangles in leaf
		int32_t firstTriIndex = ~newNode.rightChildIndex();
		int32_t lastTriIndex = firstTriIndex + newNode.rightChildNumTriangles() - 1;
		for (int32_t i = firstTriIndex; i < lastTriIndex; i++) {
			bvh.triangles[i].v0.w = 0.0f;
			bvh.triangles[i].v1.w = 0.0f;
			bvh.triangles[i].v2.w = 0.0f;
		}
		
		// Sets padding to -1.0f in last triangle in leaf (end marker)
		bvh.triangles[lastTriIndex].v0.w = -1.0f;
		bvh.triangles[lastTriIndex].v1.w = -1.0f;
		bvh.triangles[lastTriIndex].v2.w = -1.0f;
	}
	else {
		newNode.setRightChildInner(sanitizeInternal(bvh, newNode.rightChildIndex(), newNodes));
	}

	return newNodeIndex;
}

void sanitizeBVH(BVH& bvh) noexcept
{
	DynArray<BVHNode> newNodes;
	newNodes.ensureCapacity(bvh.nodes.capacity() + 32u);

	sanitizeInternal(bvh, 0, newNodes);
	bvh.nodes = std::move(newNodes);
}

} // namespace phe
