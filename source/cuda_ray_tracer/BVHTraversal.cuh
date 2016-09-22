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

__inline__ __device__ AABBIsect cudaIntersects(const Ray& ray, const vec3& min, const vec3& max, float tCurrMin, float tCurrMax) noexcept
{
	vec3 lo = (min - ray.origin) * ray.invDir;
	vec3 hi = (max - ray.origin) * ray.invDir;

	AABBIsect tmp;
	tmp.tIn = spanBeginKepler(lo.x, hi.x, lo.y, hi.y, lo.z, hi.z, tCurrMin);
	tmp.tOut = spanEndKepler(lo.x, hi.x, lo.y, hi.y, lo.z, hi.z, tCurrMax);
	return tmp;
}

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

template<size_t STACK_SIZE = 128>
__device__ RayCastResult cudaCastRay(cudaTextureObject_t bvhNodesTex, cudaTextureObject_t trianglesTex,
                                     const Ray& ray, float tMin = 0.0001f, float tMax = FLT_MAX) noexcept
{
	// Create local stack
	uint32_t stack[STACK_SIZE];

	// Place initial node on stack
	stack[0] = 0u;
	uint32_t stackSize = 1u;

	// Traverse through the tree
	RayCastResult closest;
	closest.t = tMax;
	while (stackSize > 0u) {
		
		// Retrieve node on top of stack
		stackSize--;
		uint32_t nodeIndex = stack[stackSize];
		BVHNode node = loadBvhNode(bvhNodesTex, nodeIndex);

		// Perform AABB intersection tests and figure out which children we want to visit
		AABBIsect lcHit = cudaIntersects(ray, node.leftChildAABBMin(), node.leftChildAABBMax(), tMin, closest.t);
		AABBIsect rcHit = cudaIntersects(ray, node.rightChildAABBMin(), node.rightChildAABBMax(), tMin, closest.t);
		
		bool visitLC = lcHit.tIn <= lcHit.tOut;
		bool visitRC = rcHit.tIn <= rcHit.tOut;

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
				for (uint32_t i = 0; i < numTriangles; i++) {
					TriangleVertices tri = loadTriangle(trianglesTex, lcIndex + i);
					TriangleHit hit = intersects(tri, ray.origin, ray.dir);

					if (hit.hit && hit.t < closest.t && tMin <= hit.t) {
						closest.triangleIndex = lcIndex + i;
						closest.t = hit.t;
						closest.u = hit.u;
						closest.v = hit.v;
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
				for (uint32_t i = 0; i < numTriangles; i++) {
					TriangleVertices tri = loadTriangle(trianglesTex, rcIndex + i);
					TriangleHit hit = intersects(tri, ray.origin, ray.dir);

					if (hit.hit && hit.t < closest.t && tMin <= hit.t) {
						closest.triangleIndex = rcIndex + i;
						closest.t = hit.t;
						closest.u = hit.u;
						closest.v = hit.v;
					}
				}
			}
		}
	}

	return closest;
}

} // namespace phe
