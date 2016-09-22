// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <phantasy_engine/ray_tracer_common/BVHTraversal.hpp>

#include "CudaSfzVectorCompatibility.cuh"

namespace phe {

// cudaCastRay
// ------------------------------------------------------------------------------------------------

__device__ BVHNode loadBvhNode(cudaTextureObject_t bvhNodesTex, uint32_t nodeIndex) noexcept
{
	nodeIndex *= 4; // 4 texture reads per index
	BVHNode node;
	node.fData[0] = toSFZ(tex1Dfetch<float4>(bvhNodesTex, nodeIndex));
	node.fData[1] = toSFZ(tex1Dfetch<float4>(bvhNodesTex, nodeIndex + 1));
	node.fData[2] = toSFZ(tex1Dfetch<float4>(bvhNodesTex, nodeIndex + 2));
	uint4 dataTmp = tex1Dfetch<uint4>(bvhNodesTex, nodeIndex + 3);
	node.iData.x = dataTmp.x;
	node.iData.y = dataTmp.y;
	node.iData.z = dataTmp.z;
	node.iData.w = dataTmp.w;
	return node;
}

template<size_t STACK_SIZE = 128>
__device__ RayCastResult cudaCastRay(cudaTextureObject_t bvhNodesTex, const TriangleVertices* triangles,
                                     const Ray& ray, float tMin = 0.0001f, float tMax = FLT_MAX) noexcept
{
	// Create local stack
	uint32_t stack[STACK_SIZE];

	// Place initial node on stack
	stack[0] = 0u;
	uint32_t stackSize = 1u;

	// Traverse through the tree
	RayCastResult closest;
	while (stackSize > 0u) {
		
		// Retrieve node on top of stack
		stackSize--;
		uint32_t nodeIndex = stack[stackSize];
		BVHNode node = loadBvhNode(bvhNodesTex, nodeIndex);

		// Perform AABB intersection tests and figure out which children we want to visit
		AABBHit lcHit = intersects(ray, node.leftChildAABBMin(), node.leftChildAABBMax());
		AABBHit rcHit = intersects(ray, node.rightChildAABBMin(), node.rightChildAABBMax());
		float tCurrMax = std::min(tMax, closest.t);
		bool visitLC = lcHit.hit && lcHit.tOut > tMin && lcHit.tIn < tCurrMax;
		bool visitRC = rcHit.hit && rcHit.tOut > tMin && rcHit.tIn < tCurrMax;

		// Visit children
		if (visitLC) {
			uint32_t lcIndex = node.leftChildIndex();
			uint32_t numTriangles = node.leftChildNumTriangles();

			// Node is inner
			if (numTriangles == 0) {
				stack[stackSize] = lcIndex;
				stackSize += 1;
			}
			// Node is leaf
			else {
				const TriangleVertices* triList = triangles + lcIndex;

				for (uint32_t i = 0; i < numTriangles; i++) {
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
		if (visitRC) {
			uint32_t rcIndex = node.rightChildIndex();
			uint32_t numTriangles = node.rightChildNumTriangles();

			// Node is inner
			if (numTriangles == 0) {
				stack[stackSize] = rcIndex;
				stackSize += 1;
			}
			// Node is leaf
			else {
				const TriangleVertices* triList = triangles + rcIndex;

				for (uint32_t i = 0; i < numTriangles; i++) {
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
	}

	return closest;
}

} // namespace phe
