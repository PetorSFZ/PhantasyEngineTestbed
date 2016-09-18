// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#include "phantasy_engine/ray_tracer_common/BVHMetrics.hpp"

#include <sfz/math/MathHelpers.hpp>

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

struct TestRayCastResult
{
	float t = FLT_MAX;

	// Store additional data for computing metrics
	uint32_t visitedNodes = 0;
	uint32_t trianglesIntersected = 0;
	uint32_t triangleIntersectionTests = 0;
	uint32_t aabbIntersectionTests = 0;
	uint32_t aabbIntersections = 0;
};

// Forward declarations
// ------------------------------------------------------------------------------------------------

static void processNode(const BVH& bvh, uint32_t nodeIndex, uint32_t depth, BVHMetrics& metrics,
                        InternalBVHMetrics& internalMetrics) noexcept;

static void computeTraversalMetrics(BVHMetrics& metrics, const BVH& bvh) noexcept;

static TestRayCastResult castTestRay(const BVH& bvh, const Ray& ray,
                                     float tMin = 0.0001f, float tMax = FLT_MAX) noexcept;

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
	processNode(bvh, 0, 1, metrics, internalMetrics);

	uint32_t innerCount = metrics.nodeCount - metrics.leafCount;

	// Compute averages
	metrics.averageLeafDepth = float(internalMetrics.totalLeafDepth) / float(metrics.leafCount);
	metrics.averageTrianglesPerLeaf = float(internalMetrics.totalTrianglesPerLeaf) / float(metrics.leafCount);
	metrics.averageChildOverlapVolumeProportion = float(internalMetrics.totalOverlapVolumeProportion / double(innerCount));
	metrics.averageLeftSAProportion = float(internalMetrics.totalLeftSAProportion / double(innerCount));
	metrics.averageRightSAProportion = float(internalMetrics.totalRightSAProportion / double(innerCount));

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
    maxVisitedNodes: %u
    maxTriangleIntersectionTests: %u
    maxTrianglesIntersected: %u
    maxAABBIntersectionTests: %u
    maxAABBIntersections: %u

    averageVisitedNodes: %f
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
	       metrics.traversalMetrics.maxVisitedNodes,
	       metrics.traversalMetrics.maxTriangleIntersectionTests,
	       metrics.traversalMetrics.maxTrianglesIntersected,
	       metrics.traversalMetrics.maxAABBIntersectionTests,
	       metrics.traversalMetrics.maxAABBIntersections,
	       metrics.traversalMetrics.averageVisitedNodes,
	       metrics.traversalMetrics.averageTriangleIntersectionTests,
	       metrics.traversalMetrics.averageTrianglesIntersected,
	       metrics.traversalMetrics.averageAABBIntersectionTests,
	       metrics.traversalMetrics.averageAABBIntersections);
}

// Static functions
// ------------------------------------------------------------------------------------------------

static void processNode(const BVH& bvh, uint32_t nodeIndex, uint32_t depth, BVHMetrics& metrics,
                        InternalBVHMetrics& internalMetrics) noexcept
{
	const BVHNode& node = bvh.nodes[nodeIndex];
	if (node.isLeaf()) {
		internalMetrics.totalLeafDepth += depth;
		internalMetrics.totalTrianglesPerLeaf += node.numTriangles();
		metrics.leafCount++;
		updateMin(metrics.minLeafDepth, depth);
		updateMax(metrics.maxLeafDepth, depth);
	}
	else {
		const BVHNode& leftNode = bvh.nodes[node.leftChildIndex()];
		const BVHNode& rightNode = bvh.nodes[node.rightChildIndex()];

		const sfz::AABB parentAABB = sfz::AABB(node.min, node.max);
		const sfz::AABB leftAABB = sfz::AABB(leftNode.min, leftNode.max);
		const sfz::AABB rightAABB = sfz::AABB(rightNode.min, rightNode.max);

		float parentSurfaceArea = surfaceArea(parentAABB);
		float parentVolume = volume(parentAABB);
		float leftSurfaceArea = surfaceArea(leftAABB);
		float rightSurfaceArea = surfaceArea(rightAABB);
		float childOverlapVolume = overlapVolume(leftAABB, rightAABB);

		internalMetrics.totalLeftSAProportion += leftSurfaceArea / parentSurfaceArea;
		internalMetrics.totalRightSAProportion += rightSurfaceArea / parentSurfaceArea;
		if (!sfz::approxEqual(0.0f, parentVolume)) {
			internalMetrics.totalOverlapVolumeProportion += childOverlapVolume / parentVolume;
		}

		processNode(bvh, node.leftChildIndex(), depth + 1, metrics, internalMetrics);
		processNode(bvh, node.rightChildIndex(), depth + 1, metrics, internalMetrics);
	}
}

/// Compute traversal metrics by sampling the scene with a number of sample rays
static void computeTraversalMetrics(BVHMetrics& metrics, const BVH& bvh) noexcept
{
	TraversalMetrics& traversalMetrics = metrics.traversalMetrics;

	// Initialize result members

	traversalMetrics.maxVisitedNodes = 0;
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

	uint64_t totalVisitedNodes = 0;
	uint64_t totalTriangleIntersectionTests = 0;
	uint64_t totalTrianglesIntersected = 0;
	uint64_t totalAABBIntersectionTests = 0;
	uint64_t totalAABBIntersections = 0;

	for (const auto& ray : SAMPLE_RAYS) {
		TestRayCastResult rayResult = castTestRay(bvh, ray);

		updateMax(traversalMetrics.maxVisitedNodes, rayResult.visitedNodes);
		updateMax(traversalMetrics.maxTriangleIntersectionTests, rayResult.triangleIntersectionTests);
		updateMax(traversalMetrics.maxTrianglesIntersected, rayResult.trianglesIntersected);
		updateMax(traversalMetrics.maxAABBIntersectionTests, rayResult.aabbIntersectionTests);
		updateMax(traversalMetrics.maxAABBIntersections, rayResult.aabbIntersections);

		totalVisitedNodes += rayResult.visitedNodes;
		totalTriangleIntersectionTests += rayResult.triangleIntersectionTests;
		totalTrianglesIntersected += rayResult.trianglesIntersected;
		totalAABBIntersectionTests += rayResult.aabbIntersectionTests;
		totalAABBIntersections += rayResult.aabbIntersections;
	}

	traversalMetrics.averageVisitedNodes = float(totalVisitedNodes) / float(sampleRayCount);
	traversalMetrics.averageTriangleIntersectionTests = float(totalTriangleIntersectionTests) / float(sampleRayCount);
	traversalMetrics.averageTrianglesIntersected = float(totalTrianglesIntersected) / float(sampleRayCount);
	traversalMetrics.averageAABBIntersectionTests = float(totalAABBIntersectionTests) / float(sampleRayCount);
	traversalMetrics.averageAABBIntersections = float(totalAABBIntersections) / float(sampleRayCount);
}

/// Version of castRay that stores additional data about the traversal
static TestRayCastResult castTestRay(const BVH& bvh, const Ray& ray,
                                     float tMin, float tMax) noexcept
{
	// Create local stack
	DynArray<uint32_t> stack;
	stack.setCapacity(bvh.maxDepth);

	// Place initial node on stack
	stack.add(0);

	// Traverse through the tree
	TestRayCastResult result;
	while (stack.size() > 0) {

		// Retrieve node on top of stack
		const BVHNode& node = bvh.nodes[stack.last()];
		stack.remove(stack.size() - 1);
		result.visitedNodes++;

		// Process inner node
		if (!node.isLeaf()) {
			float tCurrMax = std::min(tMax, result.t);
			AABBHit hit = intersects(ray, node.min, node.max);
			result.aabbIntersectionTests++;

			if (hit.hit &&
				hit.tOut > tMin &&
				hit.tIn < tCurrMax) {

				result.aabbIntersections++;

				stack.add(node.rightChildIndex());
				stack.add(node.leftChildIndex());
			}
		}
		// Process leaf node
		else {
			uint32_t triCount = node.numTriangles();
			const TriangleVertices* triList = bvh.triangles.data() + node.triangleListIndex();

			for (uint32_t i = 0; i < triCount; i++) {
				const TriangleVertices& tri = triList[i];
				TriangleHit hit = intersects(tri, ray.origin, ray.dir);
				result.triangleIntersectionTests++;

				if (hit.hit && hit.t < result.t && tMin <= hit.t && hit.t <= tMax) {
					result.trianglesIntersected++;
					result.t = hit.t;
				}
			}
		}
	}

	return result;
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
