// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <phantasy_engine/ray_tracer_common/BVHTraversal.hpp>

#include "CudaSfzVectorCompatibility.cuh"

namespace phe {

// cudaCastRay helpers
// ------------------------------------------------------------------------------------------------

// Kepler video instructions, copied from the following papers implemntation
// https://research.nvidia.com/sites/default/files/publications/nvr-2012-02.pdf
/*
 *  Copyright (c) 2009-2011, NVIDIA Corporation
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *      * Redistributions of source code must retain the above copyright
 *        notice, this list of conditions and the following disclaimer.
 *      * Redistributions in binary form must reproduce the above copyright
 *        notice, this list of conditions and the following disclaimer in the
 *        documentation and/or other materials provided with the distribution.
 *      * Neither the name of NVIDIA Corporation nor the
 *        names of its contributors may be used to endorse or promote products
 *        derived from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 *  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 *  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *  DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
 *  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 *  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 *  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 *  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
__device__ __inline__ int min_min(int a, int b, int c) { int v; asm("vmin.s32.s32.s32.min %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ int min_max(int a, int b, int c) { int v; asm("vmin.s32.s32.s32.max %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ int max_min(int a, int b, int c) { int v; asm("vmax.s32.s32.s32.min %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ int max_max(int a, int b, int c) { int v; asm("vmax.s32.s32.s32.max %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ float fmin_fmin(float a, float b, float c) { return __int_as_float(min_min(__float_as_int(a), __float_as_int(b), __float_as_int(c))); }
__device__ __inline__ float fmin_fmax(float a, float b, float c) { return __int_as_float(min_max(__float_as_int(a), __float_as_int(b), __float_as_int(c))); }
__device__ __inline__ float fmax_fmin(float a, float b, float c) { return __int_as_float(max_min(__float_as_int(a), __float_as_int(b), __float_as_int(c))); }
__device__ __inline__ float fmax_fmax(float a, float b, float c) { return __int_as_float(max_max(__float_as_int(a), __float_as_int(b), __float_as_int(c))); }

// Experimentally determined best mix of float/int/video minmax instructions for Kepler.
__device__ __inline__ float spanBeginKepler(float a0, float a1, float b0, float b1, float c0, float c1, float d) { return fmax_fmax(fminf(a0, a1), fminf(b0, b1), fmin_fmax(c0, c1, d)); }
__device__ __inline__ float spanEndKepler(float a0, float a1, float b0, float b1, float c0, float c1, float d) { return fmin_fmin(fmaxf(a0, a1), fmaxf(b0, b1), fmax_fmin(c0, c1, d)); }

struct AABBIsect final {
	float tIn;
	float tOut;
};

__inline__ __device__ AABBIsect cudaIntersects(const vec3& invDir, const vec3& originDivDir, const vec3& min, const vec3& max, float tCurrMin, float tCurrMax) noexcept
{
	//vec3 lo = (min - ray.origin) * ray.invDir;
	//vec3 hi = (max - ray.origin) * ray.invDir;

	// FMA operations
	float loX = min.x * invDir.x - originDivDir.x;
	float loY = min.y * invDir.y - originDivDir.y;
	float loZ = min.z * invDir.z - originDivDir.z;
	float hiX = max.x * invDir.x - originDivDir.x;
	float hiY = max.y * invDir.y - originDivDir.y;
	float hiZ = max.z * invDir.z - originDivDir.z;

	AABBIsect tmp;
	tmp.tIn = spanBeginKepler(loX, hiX, loY, hiY, loZ, hiZ, tCurrMin);
	tmp.tOut = spanEndKepler(loX, hiX, loY, hiY, loZ, hiZ, tCurrMax);
	return tmp;
}

__device__ BVHNode loadBvhNode(cudaTextureObject_t bvhNodesTex, uint32_t nodeIndex) noexcept
{
	nodeIndex *= 4; // 4 texture reads per index
	BVHNode node;
	node.fData[0] = toSFZ(tex1Dfetch<float4>(bvhNodesTex, nodeIndex));
	node.fData[1] = toSFZ(tex1Dfetch<float4>(bvhNodesTex, nodeIndex + 1));
	node.fData[2] = toSFZ(tex1Dfetch<float4>(bvhNodesTex, nodeIndex + 2));
	node.iData = toSFZ(tex1Dfetch<int4>(bvhNodesTex, nodeIndex + 3));
	return node;
}

__device__ TriangleVertices loadTriangle(cudaTextureObject_t trianglesTex, uint32_t triIndex) noexcept
{
	triIndex *= 3; // 3 texture reads per index
	TriangleVertices v;
	v.v0 = toSFZ(tex1Dfetch<float4>(trianglesTex, triIndex));
	v.v1 = toSFZ(tex1Dfetch<float4>(trianglesTex, triIndex + 1));
	v.v2 = toSFZ(tex1Dfetch<float4>(trianglesTex, triIndex + 2));
	return v;
}

// cudaCastRay
// ------------------------------------------------------------------------------------------------

/// tMin is the closest the ray is allowed to intersect with the scene
/// tMax is the farthest away the ray is allowed to intersect with the scene
/// if noResultOnlyHit is true the function will return as soon as an intersection is found
/// the result will therefore only contain information about whether an intersection happens or not
template<size_t STACK_SIZE = 128>
__device__ RayCastResult cudaCastRay(cudaTextureObject_t bvhNodesTex, cudaTextureObject_t trianglesTex,
                                     const vec3& origin, const vec3& dir, float tMin = 0.0001f,
                                     float tMax = FLT_MAX, bool noResultOnlyHit = false) noexcept
{
	// Calculate extra ray information for ray vs AABB intersection test
	vec3 invDir = vec3(1.0f) / dir;
	vec3 originDivDir = origin * invDir;

	const int32_t SENTINEL = int32_t(0x7FFFFFFF);

	// Create local stack
	int32_t stack[STACK_SIZE];
	stack[0] = SENTINEL;
	uint32_t stackIndex = 0u; // Currently pointing to the topmost element (the sentinel)

	// The current index to check, start of with first node
	int32_t currentIndex = 0u;

	// Temporary to store the final result in
	RayCastResult closest;
	closest.t = tMax;

	// Traverse through the tree
	while (currentIndex != SENTINEL) {
		
		// Traverse nodes until all threads in warp have found leaf
		int32_t leafIndex = 1; // Positive, so not a leaf
		while (currentIndex != SENTINEL && currentIndex >= 0) { // If currentIndex is negative we have a leaf node
			
			// Load inner node pointed at by currentIndex
			BVHNode node = loadBvhNode(bvhNodesTex, currentIndex);

			// Perform AABB intersection tests and figure out which children we want to visit
			AABBIsect lcHit = cudaIntersects(invDir, originDivDir, node.leftChildAABBMin(), node.leftChildAABBMax(), tMin, closest.t);
			AABBIsect rcHit = cudaIntersects(invDir, originDivDir, node.rightChildAABBMin(), node.rightChildAABBMax(), tMin, closest.t);

			bool visitLC = lcHit.tIn <= lcHit.tOut;
			bool visitRC = rcHit.tIn <= rcHit.tOut;

			// If we don't need to visit any children we simply pop a new index from the stack
			if (!visitLC && !visitRC) {
				currentIndex = stack[stackIndex];
				stackIndex -= 1;
			}

			// If we need to visit at least one child
			else {
				int32_t lcIndex = node.leftChildIndex();
				int32_t rcIndex = node.rightChildIndex();

				// Put left child in currentIndex if we need to visit it, otherwise right child
				currentIndex = visitLC ? lcIndex : rcIndex;

				// If we need to visit both children we push the furthest one away to the stack
				if (visitLC && visitRC) {
					stackIndex += 1;
					if (lcHit.tIn > rcHit.tIn) {
						stack[stackIndex] = lcIndex;
						currentIndex = rcIndex;
					} else {
						stack[stackIndex] = rcIndex;
					}
				}
			}

			// If currentIndex is a leaf and we have not yet found a leaf index
			if (currentIndex < 0 && leafIndex >= 0) {
				leafIndex = currentIndex; // Store leaf index for later processing

				// Pop index from stack
				// If this is also a leaf index we will process it together with leafIndex later
				currentIndex = stack[stackIndex];
				stackIndex -= 1;
			}

			// Exit loop if all threads in warp have found at least one triangle
			//	if (!__any(leafIndex < 0)) break;
			// Equivalent inline ptx:
			unsigned int mask;
			asm(R"({
			.reg .pred p;
			setp.ge.s32 p, %1, 0;
			vote.ballot.b32 %0, p;
			})"
			: "=r"(mask)
			: "r"(leafIndex));
			if (!mask) break;
		}

		// Process leafs found
		while (leafIndex < 0) {

			// Go through all triangles in leaf
			int32_t triIndex = ~leafIndex; // Get actual index
			while (true) {
				TriangleVertices tri = loadTriangle(trianglesTex, triIndex);
				TriangleHit hit = intersects(tri, origin, dir);

				if (hit.hit && hit.t < closest.t && tMin <= hit.t) {
					closest.triangleIndex = uint32_t(triIndex);
					closest.t = hit.t;
					closest.u = hit.u;
					closest.v = hit.v;

					if (noResultOnlyHit) {
						return closest;
					}
				}

				// Check if triangle was marked as last one
				if (tri.v0.w < 0) {
					break;
				}
				triIndex += 1;
			}

			// currentIndex could potentially contain another leaf we want to process,
			// otherwise this will end the loop
			leafIndex = currentIndex;

			// If currentIndex contained a leaf it will be processed, so we pop a new element from the stack into it
			if (currentIndex < 0) {
				currentIndex = stack[stackIndex];
				stackIndex -= 1;
			}
		}
	}

	return closest;
}

} // namespace phe
