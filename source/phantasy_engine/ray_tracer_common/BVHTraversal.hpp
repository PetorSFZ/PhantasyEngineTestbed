// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include <cfloat>

#include <sfz/CUDACallable.hpp>

#include "phantasy_engine/ray_tracer_common/BVHNode.hpp"
#include "phantasy_engine/ray_tracer_common/Intersection.hpp"
#include "phantasy_engine/ray_tracer_common/Triangle.hpp"

namespace phe {

// Traversing BVH
// ------------------------------------------------------------------------------------------------

struct RayCastResult final {
	uint32_t triangleIndex = ~0u;
	
	// Amount to go in ray direction
	float t = FLT_MAX;

	// Hit position on triangle
	float u = FLT_MAX;
	float v = FLT_MAX;
};

struct DebugRayCastData final {
	uint32_t nodesVisited = 0;
	uint32_t trianglesIntersected = 0;
	uint32_t triangleIntersectionTests = 0;
	uint32_t aabbIntersectionTests = 0;
	uint32_t aabbIntersections = 0;
};

template<size_t STACK_MAX_SIZE = 144>
SFZ_CUDA_CALLABLE RayCastResult castRay(const BVHNode* nodes, const TriangleVertices* triangles,
                                        const Ray& ray, float tMin = 0.0001f, float tMax = FLT_MAX) noexcept
{
	// Create local stack
	uint32_t stack[STACK_MAX_SIZE];
#ifndef SFZ_NO_DEBUG
	for (uint32_t& s : stack) s = ~0u;
#endif

	// Place initial node on stack
	stack[0] = 0u;
	uint32_t stackSize = 1u;

	// Traverse through the tree
	RayCastResult closest;
	while (stackSize > 0u) {
		
		// Retrieve node on top of stack
		stackSize--;
		const BVHNode& node = nodes[stack[stackSize]];

		// Node is a not leaf
		if (!node.isLeaf()) {

			float tCurrMax = std::min(tMax, closest.t);
			AABBHit hit = intersects(ray, node.min, node.max);
			if (hit.hit &&
			    hit.tOut > tMin &&
			    hit.tIn < tCurrMax) {

				stack[stackSize] = node.rightChildIndex();
				stack[stackSize + 1] = node.leftChildIndex();
				stackSize += 2;
			}
		}

		// Node is a leaf
		else {
			uint32_t triCount = node.numTriangles();
			const TriangleVertices* triList = triangles + node.triangleListIndex();

			for (uint32_t i = 0; i < triCount; i++) {
				const TriangleVertices& tri = triList[i];
				TriangleHit hit = intersects(tri, ray.origin, ray.dir);

				if (hit.hit && hit.t < closest.t && tMin <= hit.t && hit.t <= tMax) {
					closest.triangleIndex = (triList - triangles) + i;
					closest.t = hit.t;
					closest.u = hit.u;
					closest.v = hit.v;

					// Possible early exit
					// if (hit.t == tMin) return closest;
				}
			}
		}
	}

	return closest;
}

/// Version of castRay that stores additional data about the traversal
template <size_t STACK_MAX_SIZE = 144>
SFZ_CUDA_CALLABLE RayCastResult castDebugRay(const BVHNode* nodes, const TriangleVertices* triangles,
                                             const Ray& ray, DebugRayCastData* debugData,
                                             float tMin = 0.0001f, float tMax = FLT_MAX) noexcept
{
	sfz_assert_debug(debugData != nullptr);

	// Create local stack
	uint32_t stack[STACK_MAX_SIZE];
#ifndef SFZ_NO_DEBUG
	for (uint32_t& s : stack) s = ~0u;
#endif

	// Place initial node on stack
	stack[0] = 0u;
	uint32_t stackSize = 1u;

	// Traverse through the tree
	RayCastResult closest;
	while (stackSize > 0u) {

		// Retrieve node on top of stack
		stackSize--;
		const BVHNode& node = nodes[stack[stackSize]];

		debugData->nodesVisited++;

		// Node is a not leaf
		if (!node.isLeaf()) {

			float tCurrMax = std::min(tMax, closest.t);
			AABBHit hit = intersects(ray, node.min, node.max);
			debugData->aabbIntersectionTests++;
			if (hit.hit &&
				hit.tOut > tMin &&
				hit.tIn < tCurrMax) {
				debugData->aabbIntersections++;

				stack[stackSize] = node.rightChildIndex();
				stack[stackSize + 1] = node.leftChildIndex();
				stackSize += 2;
			}
		}
		// Node is a leaf
		else {
			uint32_t triCount = node.numTriangles();
			const TriangleVertices* triList = triangles + node.triangleListIndex();

			for (uint32_t i = 0; i < triCount; i++) {
				const TriangleVertices& tri = triList[i];
				TriangleHit hit = intersects(tri, ray.origin, ray.dir);
				debugData->triangleIntersectionTests++;

				if (hit.hit && hit.t < closest.t && tMin <= hit.t && hit.t <= tMax) {
					debugData->trianglesIntersected++;

					closest.triangleIndex = (triList - triangles) + i;
					closest.t = hit.t;
					closest.u = hit.u;
					closest.v = hit.v;

					// Possible early exit
					// if (hit.t == tMin) return closest;
				}
			}
		}
	}

	return closest;
}

// Interpreting result
// ------------------------------------------------------------------------------------------------

struct HitInfo final {
	vec3 pos;
	vec3 normal;
	vec2 uv;
	uint32_t materialIndex;
};

SFZ_CUDA_CALLABLE HitInfo interpretHit(const TriangleData* triDatas, const RayCastResult& result,
                                       const Ray& ray) noexcept
{
	const TriangleData& data = triDatas[result.triangleIndex];
	float u = result.u;
	float v = result.v;

	// Retrieving position
	HitInfo info;
	info.pos = ray.origin + result.t * ray.dir;
	
	// Interpolating normal
	vec3 n0 = data.n0;
	vec3 n1 = data.n1;
	vec3 n2 = data.n2;
	info.normal = normalize(n0 + (n1 - n0) * u + (n2 - n0) * v); // TODO: Wrong

	// Interpolating uv coordinate
	vec2 uv0 = data.uv0;
	vec2 uv1 = data.uv1;
	vec2 uv2 = data.uv2;
	info.uv = uv0 + (uv1 - uv0) * u + (uv2 - uv0) * v; // TODO: Wrong

	// Material index
	info.materialIndex = data.materialIndex;

	return info;
}

} // namespace phe
