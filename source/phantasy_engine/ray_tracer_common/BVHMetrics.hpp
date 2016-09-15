// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include "phantasy_engine/ray_tracer_common/BVH.hpp"

namespace phe {

struct BVHMetrics {
	uint32_t nodeCount;
	uint32_t leafCount;
	uint32_t triangleCount;

	uint32_t maxLeafDepth;
	uint32_t minLeafDepth;
	float averageLeafDepth;
	float medianLeafDepth; // Not implemented
	float leafDepthDeviation; // Not implemented

	uint32_t minLeavesInNodes; // Not implemented
	uint32_t maxLeavesInNodes; // Not implemented
	float averageLeavesInNodes;
	float medianLeavesInNodes; // Not implemented
	float leavesInNodesDeviation; // Not implemented

	float averageChildOverlap; // Not implemented
	float leftVolumeProportion; // Not implemented
	float rightVolumeProportion; // Not implemented
};

BVHMetrics computeBVHMetrics(const BVH& bvh);

void printBVHMetrics(const BVH& bvh);

} // namespace phe
