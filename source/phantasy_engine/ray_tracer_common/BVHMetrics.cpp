// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#include "phantasy_engine/ray_tracer_common/BVHMetrics.hpp"

#include <sfz/geometry/AABB.hpp>
#include <sfz/math/MathHelpers.hpp>

#include "phantasy_engine/ray_tracer_common/BVHTraversal.hpp"
#include "phantasy_engine/ray_tracer_common/Intersection.hpp"

namespace phe {

// Internal structs
// ------------------------------------------------------------------------------------------------

struct InternalBVHMetrics
{
	uint64_t totalLeafDepth;
	uint64_t totalTrianglesPerLeaf;
	double totalLeftSAProportion;
	double totalRightSAProportion;
	double totalOverlapVolumeProportion;
};

// Forward declarations
// ------------------------------------------------------------------------------------------------

static void processNode(const BVH& bvh, BVHMetrics& metrics, InternalBVHMetrics& internalMetrics,
                        uint32_t nodeIndex, uint32_t depth, const sfz::AABB& thisAABB) noexcept;

static void processLeaf(const BVH& bvh, BVHMetrics& metrics, InternalBVHMetrics& internalMetrics,
                        uint32_t numTriangles, uint32_t depth) noexcept;

static void computeTraversalMetrics(BVHMetrics& metrics, const BVH& bvh) noexcept;

static float surfaceArea(const sfz::AABB& aabb) noexcept;

static float surfaceArea(const vec3& boxSize) noexcept;

static float volume(const sfz::AABB& aabb) noexcept;

static float volume(const vec3& boxSize) noexcept;

static float overlapVolume(const sfz::AABB& box1, const sfz::AABB& box2) noexcept;

template<typename T>
static void updateMin(T& currentMin, const T& newCandidate) noexcept;

template<typename T>
static void updateMax(T& currentMin, const T& newCandidate) noexcept;

// Functions
// ------------------------------------------------------------------------------------------------

BVHMetrics computeBVHMetrics(const BVH& bvh)
{
	BVHMetrics metrics;
	InternalBVHMetrics internalMetrics;

	// Initialize result members

	metrics.nodeCount = bvh.nodes.size();
	metrics.leafCount = 0;
	metrics.triangleCount = bvh.triangles.size();

	metrics.minLeafDepth = UINT32_MAX;
	metrics.maxLeafDepth = 0;
	metrics.averageLeafDepth = 0.0f;
	metrics.medianLeafDepth = 0.0f;
	metrics.leafDepthDeviation = 0.0f;

	metrics.minTrianglesPerLeaf = UINT32_MAX;
	metrics.maxTrianglesPerLeaf = 0;
	metrics.averageTrianglesPerLeaf = 0.0f;
	metrics.medianTrianglesPerLeaf = 0.0f;
	metrics.trianglesPerLeafDeviation = 0.0f;

	// Initialize internal members
	internalMetrics.totalLeafDepth = 0;
	internalMetrics.totalTrianglesPerLeaf = 0;
	internalMetrics.totalOverlapVolumeProportion = 0.0;
	internalMetrics.totalLeftSAProportion = 0.0;
	internalMetrics.totalRightSAProportion = 0.0;

	// Traverse tree starting at root
	const BVHNode& root = bvh.nodes[0];
	sfz::AABB rootAABB = sfz::AABB(sfz::min(root.leftChildAABBMin(), root.rightChildAABBMin()),
	                               sfz::max(root.leftChildAABBMax(), root.rightChildAABBMax()));
	processNode(bvh, metrics, internalMetrics, 0, 1, rootAABB);

	// Compute averages
	metrics.averageLeafDepth = float(internalMetrics.totalLeafDepth) / float(metrics.leafCount);
	metrics.averageTrianglesPerLeaf = float(internalMetrics.totalTrianglesPerLeaf) / float(metrics.leafCount);
	metrics.averageChildOverlapVolumeProportion = float(internalMetrics.totalOverlapVolumeProportion / double(metrics.nodeCount));
	metrics.averageLeftSAProportion = float(internalMetrics.totalLeftSAProportion / double(metrics.nodeCount));
	metrics.averageRightSAProportion = float(internalMetrics.totalRightSAProportion / double(metrics.nodeCount));

	computeTraversalMetrics(metrics, bvh);

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
averageTrianglesPerLeaf: %f
averageChildOverlapVolumeProportion: %f
averageLeftSAProportion: %f
averageRightSAProportion: %f

Traversal metrics:
    maxNodesVisited: %u
    maxTriangleIntersectionTests: %u
    maxTrianglesIntersected: %u
    maxAABBIntersectionTests: %u
    maxAABBIntersections: %u

    averageNodesVisited: %f
    averageTriangleIntersectionTests: %f
    averageTrianglesIntersected: %f
    averageAABBIntersectionTests: %f
    averageAABBIntersections: %f
)";

	BVHMetrics metrics = computeBVHMetrics(bvh);

	printf(METRICS_FORMAT,
	       metrics.nodeCount,
	       metrics.leafCount,
	       metrics.triangleCount,
	       metrics.maxLeafDepth,
	       metrics.minLeafDepth,
	       metrics.averageLeafDepth,
	       metrics.averageTrianglesPerLeaf,
	       metrics.averageChildOverlapVolumeProportion,
	       metrics.averageLeftSAProportion,
	       metrics.averageRightSAProportion,
	       metrics.traversalMetrics.maxNodesVisited,
	       metrics.traversalMetrics.maxTriangleIntersectionTests,
	       metrics.traversalMetrics.maxTrianglesIntersected,
	       metrics.traversalMetrics.maxAABBIntersectionTests,
	       metrics.traversalMetrics.maxAABBIntersections,
	       metrics.traversalMetrics.averageNodesVisited,
	       metrics.traversalMetrics.averageTriangleIntersectionTests,
	       metrics.traversalMetrics.averageTrianglesIntersected,
	       metrics.traversalMetrics.averageAABBIntersectionTests,
	       metrics.traversalMetrics.averageAABBIntersections);
}

// Static functions
// ------------------------------------------------------------------------------------------------

static void processNode(const BVH& bvh, BVHMetrics& metrics, InternalBVHMetrics& internalMetrics,
                        uint32_t nodeIndex, uint32_t depth, const sfz::AABB& thisAABB) noexcept
{
	const BVHNode& node = bvh.nodes[nodeIndex];

	const sfz::AABB leftAABB = sfz::AABB(node.leftChildAABBMin(), node.leftChildAABBMax());
	const sfz::AABB rightAABB = sfz::AABB(node.rightChildAABBMin(), node.rightChildAABBMax());

	float parentSurfaceArea = surfaceArea(thisAABB);
	float parentVolume = volume(thisAABB);
	float leftSurfaceArea = surfaceArea(leftAABB);
	float rightSurfaceArea = surfaceArea(rightAABB);
	float childOverlapVolume = overlapVolume(leftAABB, rightAABB);

	internalMetrics.totalLeftSAProportion += leftSurfaceArea / parentSurfaceArea;
	internalMetrics.totalRightSAProportion += rightSurfaceArea / parentSurfaceArea;
	if (!sfz::approxEqual(0.0f, parentVolume)) {
		internalMetrics.totalOverlapVolumeProportion += childOverlapVolume / parentVolume;
	}

	if (node.leftChildIsLeaf()) {
		processLeaf(bvh, metrics, internalMetrics, node.leftChildNumTriangles(), depth + 1);
	}
	else {
		processNode(bvh, metrics, internalMetrics, node.leftChildIndex(), depth + 1, leftAABB);
	}
	if (node.rightChildIsLeaf()) {
		processLeaf(bvh, metrics, internalMetrics, node.rightChildNumTriangles(), depth + 1);
	}
	else {
		processNode(bvh, metrics, internalMetrics, node.rightChildIndex(), depth + 1, rightAABB);
	}
}

static void processLeaf(const BVH& bvh, BVHMetrics& metrics, InternalBVHMetrics& internalMetrics,
                        uint32_t numTriangles, uint32_t depth) noexcept
{
	internalMetrics.totalLeafDepth += depth;
	internalMetrics.totalTrianglesPerLeaf += numTriangles;
	metrics.leafCount++;
	updateMin(metrics.minLeafDepth, depth);
	updateMax(metrics.maxLeafDepth, depth);
}

/// Compute traversal metrics by sampling the scene with a number of sample rays
static void computeTraversalMetrics(BVHMetrics& metrics, const BVH& bvh) noexcept
{
	TraversalMetrics& traversalMetrics = metrics.traversalMetrics;

	// Initialize result members

	traversalMetrics.maxNodesVisited = 0;
	traversalMetrics.maxTriangleIntersectionTests = 0;
	traversalMetrics.maxTrianglesIntersected = 0;
	traversalMetrics.maxAABBIntersectionTests = 0;
	traversalMetrics.maxAABBIntersections = 0;

	static const Ray SAMPLE_RAYS[]{
		// Rays directed somewhat uniformly from the middle of the scene, assuming it's centered
		// around the origin
		Ray(vec3(0.0f, 2.0f, 0.0f), vec3(1.0f, 0.0f, 0.0f)),
		Ray(vec3(0.0f, 2.0f, 0.0f), vec3(0.0f, 1.0f, 0.0f)),
		Ray(vec3(0.0f, 2.0f, 0.0f), vec3(0.0f, 0.0f, 1.0f)),
		Ray(vec3(0.0f, 2.0f, 0.0f), vec3(-1.0f, 0.0f, 0.0f)),
		Ray(vec3(0.0f, 2.0f, 0.0f), vec3(0.0f, -1.0f, 0.0f)),
		Ray(vec3(0.0f, 2.0f, 0.0f), vec3(0.0f, 0.0f, -1.0f)),
		Ray(vec3(0.0f, 2.0f, 0.0f), normalize(vec3(1.0f, 1.0f, 1.0f))),
		Ray(vec3(0.0f, 2.0f, 0.0f), normalize(-vec3(1.0f, 1.0f, 1.0f))),

		// Special ray directed toward flower vase in sponza
		Ray(vec3(6.23f, 6.2f, -11.3f), normalize(vec3(0.1f, -1.0f, 0.0f)))
	};
	static const size_t sampleRayCount = sizeof(SAMPLE_RAYS) / sizeof(Ray);

	uint64_t totalNodesVisited = 0;
	uint64_t totalTriangleIntersectionTests = 0;
	uint64_t totalTrianglesIntersected = 0;
	uint64_t totalAABBIntersectionTests = 0;
	uint64_t totalAABBIntersections = 0;

	for (const auto& ray : SAMPLE_RAYS) {
		DebugRayCastData debugData;
		RayCastResult rayResult = castDebugRay(bvh.nodes.data(), bvh.triangles.data(), ray, &debugData);

		updateMax(traversalMetrics.maxNodesVisited, debugData.nodesVisited);
		updateMax(traversalMetrics.maxTriangleIntersectionTests, debugData.triangleIntersectionTests);
		updateMax(traversalMetrics.maxTrianglesIntersected, debugData.trianglesIntersected);
		updateMax(traversalMetrics.maxAABBIntersectionTests, debugData.aabbIntersectionTests);
		updateMax(traversalMetrics.maxAABBIntersections, debugData.aabbIntersections);

		totalNodesVisited += debugData.nodesVisited;
		totalTriangleIntersectionTests += debugData.triangleIntersectionTests;
		totalTrianglesIntersected += debugData.trianglesIntersected;
		totalAABBIntersectionTests += debugData.aabbIntersectionTests;
		totalAABBIntersections += debugData.aabbIntersections;
	}

	traversalMetrics.averageNodesVisited = float(totalNodesVisited) / float(sampleRayCount);
	traversalMetrics.averageTriangleIntersectionTests = float(totalTriangleIntersectionTests) / float(sampleRayCount);
	traversalMetrics.averageTrianglesIntersected = float(totalTrianglesIntersected) / float(sampleRayCount);
	traversalMetrics.averageAABBIntersectionTests = float(totalAABBIntersectionTests) / float(sampleRayCount);
	traversalMetrics.averageAABBIntersections = float(totalAABBIntersections) / float(sampleRayCount);
}

static float surfaceArea(const sfz::AABB& aabb) noexcept
{
	vec3 boxSize = aabb.extents();
	return surfaceArea(boxSize);
}

static float surfaceArea(const vec3& boxSize) noexcept
{
	return 2.0f * (
		boxSize.x * boxSize.y +
		boxSize.x * boxSize.z +
		boxSize.y * boxSize.z
	);
}

static float volume(const sfz::AABB& aabb) noexcept
{
	vec3 boxSize = aabb.extents();
	return volume(boxSize);
}

static float volume(const vec3& boxSize) noexcept
{
	return boxSize.x * boxSize.y * boxSize.z;
}

static float overlapVolume(const sfz::AABB& box1, const sfz::AABB& box2) noexcept
{
	vec3 overlapSize = sfz::max(vec3(0.0f), sfz::min(box1.max, box2.max) - sfz::max(box1.min, box1.min));
	return volume(overlapSize);
}

template<typename T>
static void updateMin(T& currentMin, const T& newCandidate) noexcept
{
	currentMin = std::min(currentMin, newCandidate);
}

template<typename T>
static void updateMax(T& currentMax, const T& newCandidate) noexcept
{
	currentMax = std::max(currentMax, newCandidate);
}

} // namespace phe
