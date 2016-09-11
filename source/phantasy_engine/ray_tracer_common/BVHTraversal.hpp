// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include <sfz/CUDACallable.hpp>

#include "phantasy_engine/ray_tracer_common/BVHNode.hpp"
#include "phantasy_engine/ray_tracer_common/Intersection.hpp"
#include "phantasy_engine/ray_tracer_common/Triangle.hpp"

namespace phe {

struct RayCastResult final {
	uint32_t triangleIndex = ~0u;
	
	// Amount to go in ray direction
	float t = FLT_MAX;

	// Hit position on triangle
	float u = FLT_MAX;
	float v = FLT_MAX;
};

template<size_t STACK_MAX_SIZE = 196u>
SFZ_CUDA_CALLABLE RayCastResult castRay(BVHNode* nodes, TriangleVertices* triangles, const Ray& ray, float tMin = 0.0001f, float tMax = FLT_MAX) noexcept
{
	// Create local stack
	uint32_t stack[STACK_MAX_SIZE];
	for (uint32_t& s : stack) s = ~0u;
	
	// Place initial node on stack
	stack[0] = 0u;
	uint32_t stackSize = 1u;

	// Traverse through the tree
	RayCastResult closest;
	while (stackSize > 0u) {
		
		// Retrieve node on top of stack
		stackSize--;
		BVHNode node = nodes[stack[stackSize]];

		// Node is a leaf
		if (node.isLeaf()) {
			uint32_t triCount = node.numTriangles();
			TriangleVertices* triList = triangles + node.triangleListIndex();

			for (uint32_t i = 0; i < triCount; i++) {
				TriangleVertices& tri = triList[i];
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

		// Node is a not leaf
		else {
			AABBHit hit = intersects(ray, node.min, node.max);
			if (hit.hit && hit.t <= closest.t && hit.t <= tMax) {
				
				stack[stackSize] = node.leftChildIndex();
				stack[stackSize + 1u] = node.rightChildIndex();
				stackSize += 2u;
			}
		}
	}

	return closest;
}

} // namespace phe
