// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include "phantasy_engine/ray_tracer_common/BVH.hpp"

namespace phe {

// TraversalMetrics struct
// ------------------------------------------------------------------------------------------------

struct TraversalMetrics
{
	uint32_t maxVisitedNodes;
	uint32_t maxTriangleIntersectionTests;
	uint32_t maxTrianglesIntersected;
	uint32_t maxAABBIntersectionTests;
	uint32_t maxAABBIntersections;

	float averageVisitedNodes;
	float averageTriangleIntersectionTests;
	float averageTrianglesIntersected;
	float averageAABBIntersectionTests;
	float averageAABBIntersections;
};

// BVHMetrics struct
// ------------------------------------------------------------------------------------------------

struct BVHMetrics {
	uint32_t nodeCount;
	uint32_t leafCount;
	uint32_t triangleCount;

	uint32_t maxLeafDepth;
	uint32_t minLeafDepth;
	float averageLeafDepth;
	float medianLeafDepth; // Not implemented
	float leafDepthDeviation; // Not implemented

	uint32_t minTrianglesPerLeaf; // Not implemented
	uint32_t maxTrianglesPerLeaf; // Not implemented
	float averageTrianglesPerLeaf;
	float medianTrianglesPerLeaf; // Not implemented
	float trianglesPerLeafDeviation; // Not implemented

	float averageChildVolumeOverlap; // Not implemented
	float averageLeftVolumeProportion; // Not implemented
	float averageRightVolumeProportion; // Not implemented

	TraversalMetrics traversalMetrics;
};

// Functions
// ------------------------------------------------------------------------------------------------

BVHMetrics computeBVHMetrics(const BVH& bvh);

void printBVHMetrics(const BVH& bvh);

} // namespace phe
