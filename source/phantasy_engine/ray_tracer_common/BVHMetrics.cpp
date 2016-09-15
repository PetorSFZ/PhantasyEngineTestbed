// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#include "phantasy_engine/ray_tracer_common/BVHMetrics.hpp"

namespace phe {

struct InternalBVHMetrics
{
	uint64_t totalLeafDepth;
	uint64_t totalTrianglesInNodes;
	uint32_t depth;
};

static void processNode(const BVH& bvh, uint32_t nodeIndex, uint32_t depth, BVHMetrics& metrics, InternalBVHMetrics& internalMetrics)
{
	const BVHNode& node = bvh.nodes[nodeIndex];
	if (node.isLeaf()) {
		internalMetrics.totalLeafDepth += depth;
		internalMetrics.totalTrianglesInNodes += node.numTriangles();
		metrics.minLeafDepth = std::min(metrics.minLeafDepth, depth);
		metrics.maxLeafDepth = std::max(metrics.maxLeafDepth, depth);
		metrics.leafCount++;
	}
	else {
		processNode(bvh, node.leftChildIndex(), depth + 1, metrics, internalMetrics);
		processNode(bvh, node.rightChildIndex(), depth + 1, metrics, internalMetrics);
	}
}

BVHMetrics computeBVHMetrics(const BVH& bvh)
{
	BVHMetrics metrics;
	InternalBVHMetrics internalMetrics;

	// Initialize result members

	metrics.nodeCount = bvh.nodes.size();
	metrics.triangleCount = bvh.triangles.size() / 3;
	metrics.leafCount = 0;

	metrics.minLeafDepth = UINT32_MAX;
	metrics.maxLeafDepth = 0;
	metrics.averageLeafDepth = 0.0f;
	metrics.medianLeafDepth = 0.0f;
	metrics.leafDepthDeviation = 0.0f;

	metrics.minLeavesInNodes = UINT32_MAX;
	metrics.maxLeavesInNodes = 0;
	metrics.averageLeavesInNodes = 0.0f;
	metrics.medianLeavesInNodes = 0.0f;
	metrics.leavesInNodesDeviation = 0.0f;

	metrics.averageChildOverlap = 0.0f;
	metrics.leftVolumeProportion = 0.0f;
	metrics.rightVolumeProportion = 0.0f;

	// Initialize internal members
	internalMetrics.totalLeafDepth = 0;
	internalMetrics.totalTrianglesInNodes = 0;

	// Traverse tree starting at root
	processNode(bvh, 0, 1, metrics, internalMetrics);

	metrics.averageLeafDepth = float(internalMetrics.totalLeafDepth) / float(metrics.leafCount);
	metrics.averageLeavesInNodes = float(internalMetrics.totalTrianglesInNodes) / float(metrics.leafCount);

	return metrics;
}

void printBVHMetrics(const BVH& bvh)
{
	static const char* METRICS_FORMAT = 
R"(nodeCount: %u
leafCount: %u
triangleCount: %u
maxLeafDepth: %u
minLeafDepth: %u
averageLeafDepth: %f
averageLeavesInNodes: %f
)";

	BVHMetrics metrics = computeBVHMetrics(bvh);
	printf(METRICS_FORMAT,
		metrics.nodeCount,
		metrics.leafCount,
		metrics.triangleCount,
		metrics.maxLeafDepth,
		metrics.minLeafDepth,
		metrics.averageLeafDepth,
		metrics.averageLeavesInNodes
	);
}

} // namespace phe
