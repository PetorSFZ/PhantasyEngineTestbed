// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#include <sfz/containers/DynArray.hpp>

#include "phantasy_engine/ray_tracer_common/BVHCacheOptimizer.hpp"

namespace phe {

// Statics
// ------------------------------------------------------------------------------------------------

static uint32_t optimizeInternal(const DynArray<BVHNode>& oldNodes, uint32_t oldNodeIndex,
                                 DynArray<BVHNode>& newNodes) noexcept
{
	const BVHNode& oldNode = oldNodes[oldNodeIndex];

	if (oldNode.isLeaf()) {
		newNodes.add(oldNode);
		return newNodes.size() - 1u;
	}

	uint32_t newNodeIndex = newNodes.size();
	newNodes.add(oldNode);
	BVHNode& newNode = newNodes[newNodeIndex];

	uint32_t leftChildIndex = optimizeInternal(oldNodes, oldNode.leftChildIndex(), newNodes);
	uint32_t rightChildIndex = optimizeInternal(oldNodes, oldNode.rightChildIndex(), newNodes);
	
	newNode.setInner(leftChildIndex, rightChildIndex);
	return newNodeIndex;
}

void optimizeBVHCacheLocality(BVH& bvh) noexcept
{
	DynArray<BVHNode> newNodes;
	newNodes.ensureCapacity(bvh.nodes.capacity());

	optimizeInternal(bvh.nodes, 0u, newNodes);
	bvh.nodes = std::move(newNodes);
}

} // namespace phe
