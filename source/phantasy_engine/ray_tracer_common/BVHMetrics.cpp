// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#include "phantasy_engine/ray_tracer_common/BVHMetrics.hpp"
#include "phantasy_engine/ray_tracer_common/Intersection.hpp"

namespace phe {

BVHMetrics computeBVHMetrics(const BVH& bvh)
{
	BVHMetrics result;

	// Initialize result values

	result.nodeCount = bvh.nodes.size();
	result.triangleCount = bvh.triangles.size() / 3;
	result.leafCount = 0;

	result.minDepth = UINT32_MAX;
	result.maxDepth = 0;
	result.averageLeafDepth = 0.0f;
	result.medianLeafDepth = 0.0f;
	result.leafDepthDeviation = 0.0f;

	result.minLeavesInNodes = UINT32_MAX;
	result.maxLeavesInNodes = 0;
	result.averageLeavesInNodes = 0.0f;
	result.medianLeavesInNodes = 0.0f;
	result.leavesInNodesDeviation = 0.0f;

	result.averageChildOverlap = 0.0f;
	result.leftVolumeProportion = 0.0f;
	result.rightVolumeProportion = 0.0f;

	// Internal intermediate variables

	uint64_t totalLeafDepth = 0;
	uint64_t totalTrianglesInNodes = 0;
	uint32_t depth = 1;

	// Initialize stack
	DynArray<uint32_t> stack;
	stack.setCapacity(bvh.maxDepth);
	stack.add(0);

	// Traverse the BVH
	while (stack.size() > 0) {
		// Pop node off the stack
		uint32_t nodeIndex = stack.last();
		stack.remove(stack.size() - 1);

		const BVHNode& node = bvh.nodes[nodeIndex];
		if (node.isLeaf()) {
			totalLeafDepth += depth;
			totalTrianglesInNodes += node.numTriangles();
			result.minDepth = std::min(result.minDepth, depth);
			result.leafCount++;
		} else {
			depth++;
			sfz_assert_debug(node.leftChildIndex() < bvh.nodes.size());
			sfz_assert_debug(node.rightChildIndex() < bvh.nodes.size());
			stack.add(node.rightChildIndex());
			stack.add(node.leftChildIndex());
		}
	}

	return result;
}

void printBVHMetrics(const BVH& bvh)
{
	static const char* METRICS_FORMAT = 
R"(nodeCount: %u
triangleCount: %u
numNodes: %u
)";

	BVHMetrics metrics = computeBVHMetrics(bvh);
	printf(METRICS_FORMAT,
		metrics.nodeCount,
		metrics.leafCount,
		metrics.triangleCount
	);
}

} // namespace phe
