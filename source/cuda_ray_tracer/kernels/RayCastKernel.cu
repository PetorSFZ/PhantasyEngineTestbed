// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#include "RayCastKernel.hpp"

#include <phantasy_engine/ray_tracer_common/BVHNode.hpp>
#include <phantasy_engine/ray_tracer_common/Triangle.hpp>

#include "CudaHelpers.hpp"
#include "CudaSfzVectorCompatibility.cuh"

#include "math_constants.h"

namespace phe {

using sfz::vec2;
using sfz::vec2i;
using sfz::vec3;
using sfz::vec3i;
using sfz::vec4;
using sfz::vec4i;

// Optimized ray vs AABB test, courtesy of Timo Aila (NVIDIA), Samuli Laine (NVIDIA), Tero Karras (NVIDIA)
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

static __device__ AABBIsect rayVsAaabb(const vec3& invDir, const vec3& originDivDir, const vec3& min, const vec3& max, float tCurrMin, float tCurrMax)
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

// Line vs triangle intersection test
// ------------------------------------------------------------------------------------------------

// See page 750 in Real-Time Rendering 3
static __device__ void rayVsTriangle(const TriangleVertices& tri, const vec3& origin, const vec3& dir, float& t, float& u, float& v) noexcept
{
	const float EPS = 0.00001f;
	vec3 p0 = tri.v0.xyz;
	vec3 p1 = tri.v1.xyz;
	vec3 p2 = tri.v2.xyz;

	vec3 e1 = p1 - p0;
	vec3 e2 = p2 - p0;
	vec3 q = cross(dir, e2);
	float a = dot(e1, q);
	if (-EPS < a && a < EPS) {
		t = CUDART_INF_F;
		return;
	}

	// Backface culling here?
	// dot(cross(e1, e2), dir) <= 0.0 ??

	float f = 1.0f / a;
	vec3 s = origin - p0;
	u = f * dot(s, q);
	if (u < 0.0f) {
		t = CUDART_INF_F;
		return;
	}

	vec3 r = cross(s, e1);
	v = f * dot(dir, r);
	if (v < 0.0f || (u + v) > 1.0f) {
		t = CUDART_INF_F;
		return;
	}

	t = f * dot(e2, r);
}

// Helper functions
// ------------------------------------------------------------------------------------------------

static __device__ BVHNode loadBvhNode(cudaTextureObject_t bvhNodesTex, uint32_t nodeIndex)
{
	nodeIndex *= 4; // 4 texture reads per index
	BVHNode node;
	node.fData[0] = toSFZ(tex1Dfetch<float4>(bvhNodesTex, nodeIndex));
	node.fData[1] = toSFZ(tex1Dfetch<float4>(bvhNodesTex, nodeIndex + 1));
	node.fData[2] = toSFZ(tex1Dfetch<float4>(bvhNodesTex, nodeIndex + 2));
	node.iData = toSFZ(tex1Dfetch<int4>(bvhNodesTex, nodeIndex + 3));
	return node;
}

static __device__ TriangleVertices loadTriangle(cudaTextureObject_t trianglesTex, uint32_t triIndex)
{
	triIndex *= 3; // 3 texture reads per index
	TriangleVertices v;
	v.v0 = toSFZ(tex1Dfetch<float4>(trianglesTex, triIndex));
	v.v1 = toSFZ(tex1Dfetch<float4>(trianglesTex, triIndex + 1));
	v.v2 = toSFZ(tex1Dfetch<float4>(trianglesTex, triIndex + 2));
	return v;
}

// Main ray cast kernel
// ------------------------------------------------------------------------------------------------

// Constants
// The block dimensions for the ray cast kernel. Only height (y) may be varied.
static const uint32_t RAY_CAST_KERNEL_BLOCK_WIDTH = 32;
static const uint32_t RAY_CAST_KERNEL_MAX_BLOCK_HEIGHT = 64;
static const uint32_t RAY_CAST_KERNEL_BLOCK_DEPTH = 1;
// The node stack size, BVHs may not have deeper leaves than this
static const const uint32_t RAY_CAST_KERNEL_STACK_SIZE = 128;
// The smallest number of active threads in a warp before we attempt to allocate new rays for the
// remaining threads
static const const uint32_t RAY_CAST_DYNAMIC_FETCH_THRESHOLD = 20;

// The global (atomic) index used to allocate arrays from global list
static __device__ uint32_t nextGlobalRayIndex = 0u;

static __global__ void zeroNextGlobalRayIndexKernel()
{
	nextGlobalRayIndex = 0u;
}

static __global__ void rayCastKernel(cudaTextureObject_t bvhNodes, cudaTextureObject_t triangleVerts,
                                     const RayIn* __restrict__ rays, RayHit* __restrict__ rayHits, uint32_t numRays)
{
	static_assert(sizeof(RayIn) == 32, "RayIn is padded");
	static_assert(sizeof(RayHit) == 16, "RayHitOut is padded");
	
	// Constants
	const int32_t SENTINEL = int32_t(0x7FFFFFFF);

	// Retreive thread id in warp
	// NOTE: Width of block (i.e. blockDim.x) and warp size MUST be exactly 32
	const uint32_t warpThreadIdx = threadIdx.x;
	// Below code could have been used to calculate warpThreadIdx without the width=32 assumption
	// const uint32_t tid = threadIdx.x + threadIdx.y * blockDim.x;
	// const uint32_t warpThreadIdx = tid / WARP_SIZE;
	
	// Each row (i.e. warp) gets its own shared index. This index is used as the base for the block
	// of allocated rays for the warp.
	__shared__ volatile uint32_t baseAllocatedRayIndices[RAY_CAST_KERNEL_MAX_BLOCK_HEIGHT];
	volatile uint32_t& baseAllocatedRayIndex = baseAllocatedRayIndices[threadIdx.y];
	
	// Stack holding node indices
	int32_t nodeIndexStack[RAY_CAST_KERNEL_STACK_SIZE];
	nodeIndexStack[0] = SENTINEL;

	// Index to the current topmost element in the stack
	int32_t stackTopIndex = 0;
	
	// Holds the index to the current node to check out
	int32_t currentNodeIndex = SENTINEL; // Start of as "terminated"
	
	// Holds the index of the current thread
	uint32_t globalRayIndex;

	// Temporary to store the final result in
	RayHit resHit;

	// Ray information
	vec3 origin, dir, invDir, originDivDir;
	float tMin;
	bool noResultOnlyHit;

	// Very naive loop over all input rays
	//for (uint32_t rayIdx = tid; rayIdx < numRays; rayIdx += numThreads) {
	while (true) {

		// Check whether this thread needs a new ray or not
		const bool needWork = currentNodeIndex == SENTINEL;

		// Mask that contains the needWork flag for every thread in this warp
		const uint32_t needWorkMask = __ballot(needWork);

		// Find number of threads that need work by counting the number of ones in needWorkMask
		const uint32_t numThreadsNeedWork = __popc(needWorkMask);

		// All bit indices >= to warpThreadIdx will be 0, all indices < will be 1.
		// I.e. for warpThreadIdx 3: [0, 0, ..., 0, 1, 1, 1]
		// warpThreadIdx 2: [0, 0, ..., 0, 1, 1]
		// warpTrheadIdx 0: [0, 0, ..., 0]
		const uint32_t lowerThreadsBallotMask = (1u << warpThreadIdx) - 1u;

		// Remove all 1s for warp thread with larger or equal idx, then count the remaining ones.
		// I.e. the thread with the lowest warpThreadIdx that needs work will get 0, the next lowest
		// one 1, etc.
		const uint32_t rayAllocationIdx = __popc(needWorkMask & lowerThreadsBallotMask);

		// Allocate rays from global list of rays
		if (needWork) {
		
			// Allocate number of needed rays for warp atomically with only one warp thread
			if (rayAllocationIdx == 0) {
				// atomicAdd(addr, diff) => returns *addr, then adds diff
				baseAllocatedRayIndex = atomicAdd(&nextGlobalRayIndex, numThreadsNeedWork);
			}

			// Set globalRayIndex for current thread
			globalRayIndex = baseAllocatedRayIndex + rayAllocationIdx;

			// Terminate thread if the allocated ray does not exist.
			// NOTE: This is the only place a thread can terminate in this kernel
			if (globalRayIndex >= numRays) break;
			
			// Retrieve ray and set information
			const RayIn ray = rays[globalRayIndex];
			origin = ray.origin();
			dir = ray.dir();
			invDir = vec3(1.0f) / dir;
			originDivDir = origin * invDir;
			tMin = 0.0001f;
			noResultOnlyHit = ray.noResultOnlyHit();

			// Reset stack
			stackTopIndex = 0;
			currentNodeIndex = 0;

			// Reset temporary
			resHit.triangleIndex = ~0u;
			resHit.t = ray.maxDist();
		}

		// Traverse through the tree
		while (currentNodeIndex != SENTINEL) {
		
			// Traverse nodes until all threads in warp have found leaf
			int32_t leafIndex = 1; // Positive, so not a leaf
			// If currentIndex is negative we have a leaf node
			while (currentNodeIndex != SENTINEL && currentNodeIndex >= 0) {
			
				// Load inner node pointed at by currentIndex
				BVHNode node = loadBvhNode(bvhNodes, currentNodeIndex);

				// Perform AABB intersection tests and figure out which children we want to visit
				AABBIsect lcHit = rayVsAaabb(invDir, originDivDir, node.leftChildAABBMin(),
				                             node.leftChildAABBMax(), tMin, resHit.t);
				AABBIsect rcHit = rayVsAaabb(invDir, originDivDir, node.rightChildAABBMin(),
				                             node.rightChildAABBMax(), tMin, resHit.t);

				bool visitLC = lcHit.tIn <= lcHit.tOut;
				bool visitRC = rcHit.tIn <= rcHit.tOut;

				// If we don't need to visit any children we simply pop a new index from the stack
				if (!visitLC && !visitRC) {
					currentNodeIndex = nodeIndexStack[stackTopIndex];
					stackTopIndex -= 1;
				}

				// If we need to visit at least one child
				else {
					int32_t lcIndex = node.leftChildIndexRaw();
					int32_t rcIndex = node.rightChildIndexRaw();

					// Put left child in currentIndex if we need to visit it, otherwise right child
					currentNodeIndex = visitLC ? lcIndex : rcIndex;

					// If we need to visit both children we push the furthest one away to the stack
					if (visitLC && visitRC) {
						stackTopIndex += 1;
						if (lcHit.tIn > rcHit.tIn) {
							nodeIndexStack[stackTopIndex] = lcIndex;
							currentNodeIndex = rcIndex;
						} else {
							nodeIndexStack[stackTopIndex] = rcIndex;
						}
					}
				}

				// If currentIndex is a leaf and we have not yet found a leaf index
				if (currentNodeIndex < 0 && leafIndex >= 0) {
					leafIndex = currentNodeIndex; // Store leaf index for later processing

					// Pop index from stack
					// If this is also a leaf index we will process it together with leafIndex later
					currentNodeIndex = nodeIndexStack[stackTopIndex];
					stackTopIndex -= 1;
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
					TriangleVertices tri = loadTriangle(triangleVerts, triIndex);

					float t, u, v;
					rayVsTriangle(tri, origin, dir, t, u, v);

					if (tMin <= t && t < resHit.t) {
						resHit.t = t;
						resHit.u = u;
						resHit.v = v;

						if (noResultOnlyHit) {
							currentNodeIndex = SENTINEL;
							break;
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
				leafIndex = currentNodeIndex;

				// If currentIndex contained a leaf it will be processed, so we pop a new element
				// from the stack into it
				if (currentNodeIndex < 0) {
					currentNodeIndex = nodeIndexStack[stackTopIndex];
					stackTopIndex -= 1;
				}
			}

			// Calculate how many threads are currently working, threads not working are idling
			// and will not contribute a "true" to the ballot.
			uint32_t numThreadsCurrentlyWorking = __popc(__ballot(true));
			
			// If not enough threads are working we attempt to allocate more rays for them
			if (numThreadsCurrentlyWorking < RAY_CAST_DYNAMIC_FETCH_THRESHOLD) {
				break;
			}
		}

		// Store rayhit in output array
		rayHits[globalRayIndex] = resHit;
	}
}

// RayCastKernel without thread persistence
// ------------------------------------------------------------------------------------------------

static __global__ void rayCastNoPersistenceKernel(cudaTextureObject_t bvhNodes,
                                                  cudaTextureObject_t triangleVerts,
                                                  const RayIn* __restrict__ rays, RayHit* __restrict__ rayHits, uint32_t numRays)
{
	static_assert(sizeof(RayIn) == 32, "RayIn is padded");
	static_assert(sizeof(RayHit) == 16, "RayHitOut is padded");
	

	// Constants
	const int32_t SENTINEL = int32_t(0x7FFFFFFF);

	// Calculate ray index in array
	uint32_t rayIdx = blockDim.x * blockIdx.x + threadIdx.x;
	if (rayIdx >= numRays) return;

	// Stack holding node indices
	int32_t nodeIndexStack[RAY_CAST_KERNEL_STACK_SIZE];
	nodeIndexStack[0] = SENTINEL;

	// Index to the current topmost element in the stack
	int32_t stackTopIndex = 0;
	
	// Holds the index to the current node to check out
	int32_t currentNodeIndex = 0;

	// Retrieve ray
	const RayIn ray = rays[rayIdx];
	vec3 origin = ray.origin();
	vec3 dir = ray.dir();
	vec3 invDir = vec3(1.0f) / dir;
	vec3 originDivDir = origin * invDir;
	float tMin = 0.0001f;
	bool noResultOnlyHit = ray.noResultOnlyHit();

	// Temporary to store the final result in
	RayHit resHit;
	resHit.triangleIndex = ~0u;
	resHit.t = ray.maxDist();

	// Traverse through the tree
	while (currentNodeIndex != SENTINEL) {
		
		// Traverse nodes until all threads in warp have found leaf
		int32_t leafIndex = 1; // Positive, so not a leaf
		// If currentIndex is negative we have a leaf node
		while (currentNodeIndex != SENTINEL && currentNodeIndex >= 0) {
			
			// Load inner node pointed at by currentIndex
			BVHNode node = loadBvhNode(bvhNodes, currentNodeIndex);

			// Perform AABB intersection tests and figure out which children we want to visit
			AABBIsect lcHit = rayVsAaabb(invDir, originDivDir, node.leftChildAABBMin(),
				                            node.leftChildAABBMax(), tMin, resHit.t);
			AABBIsect rcHit = rayVsAaabb(invDir, originDivDir, node.rightChildAABBMin(),
				                            node.rightChildAABBMax(), tMin, resHit.t);

			bool visitLC = lcHit.tIn <= lcHit.tOut;
			bool visitRC = rcHit.tIn <= rcHit.tOut;

			// If we don't need to visit any children we simply pop a new index from the stack
			if (!visitLC && !visitRC) {
				currentNodeIndex = nodeIndexStack[stackTopIndex];
				stackTopIndex -= 1;
			}

			// If we need to visit at least one child
			else {
				int32_t lcIndex = node.leftChildIndexRaw();
				int32_t rcIndex = node.rightChildIndexRaw();

				// Put left child in currentIndex if we need to visit it, otherwise right child
				currentNodeIndex = visitLC ? lcIndex : rcIndex;

				// If we need to visit both children we push the furthest one away to the stack
				if (visitLC && visitRC) {
					stackTopIndex += 1;
					if (lcHit.tIn > rcHit.tIn) {
						nodeIndexStack[stackTopIndex] = lcIndex;
						currentNodeIndex = rcIndex;
					} else {
						nodeIndexStack[stackTopIndex] = rcIndex;
					}
				}
			}

			// If currentIndex is a leaf and we have not yet found a leaf index
			if (currentNodeIndex < 0 && leafIndex >= 0) {
				leafIndex = currentNodeIndex; // Store leaf index for later processing

				// Pop index from stack
				// If this is also a leaf index we will process it together with leafIndex later
				currentNodeIndex = nodeIndexStack[stackTopIndex];
				stackTopIndex -= 1;
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
				TriangleVertices tri = loadTriangle(triangleVerts, triIndex);

				float t, u, v;
				rayVsTriangle(tri, origin, dir, t, u, v);

				if (tMin <= t && t < resHit.t) {
					resHit.t = t;
					resHit.u = u;
					resHit.v = v;

					if (noResultOnlyHit) {
						currentNodeIndex = SENTINEL;
						break;
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
			leafIndex = currentNodeIndex;

			// If currentIndex contained a leaf it will be processed, so we pop a new element
			// from the stack into it
			if (currentNodeIndex < 0) {
				currentNodeIndex = nodeIndexStack[stackTopIndex];
				stackTopIndex -= 1;
			}
		}
	}

	// Store rayhit in output array
	rayHits[rayIdx] = resHit;
}

// RayCastKernel launch function
// ------------------------------------------------------------------------------------------------

void launchRayCastKernel(const RayCastKernelInput& input, RayHit* rayResults,
                         const cudaDeviceProp& deviceProperties) noexcept
{
	uint32_t numSM = deviceProperties.multiProcessorCount;
	uint32_t threadsPerSM = deviceProperties.maxThreadsPerMultiProcessor;
	uint32_t maxThreadsPerBlock = deviceProperties.maxThreadsPerBlock;

	uint32_t factorExtraBlocks = 1; // Creates more blocks than necessary to fill device (more threads)
	uint32_t factorSmallerBlocks = 4; // Creates smaller blocks without changing number of threads

	uint32_t blocksPerSM = threadsPerSM / maxThreadsPerBlock;
	uint32_t threadsPerBlock = threadsPerSM / (factorSmallerBlocks * blocksPerSM);
	uint32_t numBlocks = blocksPerSM * numSM * factorSmallerBlocks * factorExtraBlocks;

	dim3 blockDims;
	blockDims.x = RAY_CAST_KERNEL_BLOCK_WIDTH;
	blockDims.y = threadsPerBlock / RAY_CAST_KERNEL_BLOCK_WIDTH;
	sfz_assert_debug(blockDims.y <= RAY_CAST_KERNEL_MAX_BLOCK_HEIGHT);
	blockDims.z = RAY_CAST_KERNEL_BLOCK_DEPTH;

	rayCastKernel<<<numBlocks, blockDims>>>(input.bvhNodes, input.triangleVerts, input.rays,
	                                        rayResults, input.numRays);
	CHECK_CUDA_ERROR(cudaGetLastError());
	CHECK_CUDA_ERROR(cudaDeviceSynchronize());

	// Set the ray counter to 0
	zeroNextGlobalRayIndexKernel<<<1, 1>>>();
	// Probably not necessary to synchronize here
	//CHECK_CUDA_ERROR(cudaGetLastError());
	//CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

void launchRayCastNoPersistenceKernel(const RayCastKernelInput& input, RayHit* rayResults,
                                      const cudaDeviceProp& deviceProperties) noexcept
{
	uint32_t raysPerBlock = 256;
	uint32_t numBlocks = (input.numRays / raysPerBlock) + 1;

	rayCastNoPersistenceKernel<<<numBlocks, raysPerBlock>>>(input.bvhNodes, input.triangleVerts,
	                                                        input.rays, rayResults, input.numRays);
	CHECK_CUDA_ERROR(cudaGetLastError());
	CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

// Secondary helper kernels (for debugging and profiling)
// ------------------------------------------------------------------------------------------------

static __device__ vec3 calculatePrimaryRayDir(const CameraDef& cam, vec2 loc, vec2 surfaceRes)
{
	vec2 locNormalized = loc / surfaceRes; // [0, 1]
	vec2 centerOffsCoord = locNormalized * 2.0f - vec2(1.0f); // [-1.0, 1.0]
	vec3 nonNormRayDir = cam.dir + cam.dX * centerOffsCoord.x + cam.dY * centerOffsCoord.y;
	return normalize(nonNormRayDir);
}

static __global__ void genPrimaryRaysKernel(RayIn* rays, CameraDef cam, vec2i res)
{
	static_assert(sizeof(RayIn) == 32, "RayIn is padded");

	// Calculate surface coordinates
	vec2i loc = vec2i(blockIdx.x * blockDim.x + threadIdx.x,
	                  blockIdx.y * blockDim.y + threadIdx.y);
	if (loc.x >= res.x || loc.y >= res.y) return;

	// Create ray
	RayIn ray;
	ray.setDir(calculatePrimaryRayDir(cam, vec2(loc), vec2(res)));
	ray.setOrigin(cam.origin);
	ray.setNoResultOnlyHit(false);
	ray.setMaxDist(FLT_MAX);

	// Write ray to array
	uint32_t id = loc.y * res.x + loc.x;
	rays[id] = ray;
}

// Assumes both parameters are normalized
static __device__ vec3 reflect(vec3 in, vec3 normal) noexcept
{
	return in - 2.0f * dot(normal, in) * normal;
}

static __global__ void genSecondaryRaysKernel(RayIn* rays, vec3 camPos, vec2i res,
                                              cudaSurfaceObject_t posTex,
                                              cudaSurfaceObject_t normalTex)
{
	// Calculate surface coordinates
	vec2i loc = vec2i(blockIdx.x * blockDim.x + threadIdx.x,
	                  blockIdx.y * blockDim.y + threadIdx.y);
	if (loc.x >= res.x || loc.y >= res.y) return;

	// Read GBuffer
	float4 posTmp = surf2Dread<float4>(posTex, loc.x * sizeof(float4), loc.y);
	float4 normalTmp = surf2Dread<float4>(normalTex, loc.x * sizeof(float4), loc.y);
	vec3 pos = vec3(posTmp.x, posTmp.y, posTmp.z);
	vec3 normal = vec3(normalTmp.x, normalTmp.y, normalTmp.z);

	// Calculate reflect direction
	vec3 camDir = normalize(pos - camPos);
	vec3 reflected = reflect(camDir, normal);

	// Create ray
	RayIn ray;
	ray.setDir(reflected);
	ray.setOrigin(pos);
	ray.setNoResultOnlyHit(false);
	ray.setMaxDist(FLT_MAX);

	// Write ray to array
	uint32_t id = loc.y * res.x + loc.x;
	rays[id] = ray;
}

static __global__ void writeRayHitsToScreenKernel(cudaSurfaceObject_t surface, vec2i res, const RayHit* rayHits)
{
	static_assert(sizeof(RayHit) == 16, "RayHitOut is padded");

	// Calculate surface coordinates
	vec2i loc = vec2i(blockIdx.x * blockDim.x + threadIdx.x,
	                  blockIdx.y * blockDim.y + threadIdx.y);
	if (loc.x >= res.x || loc.y >= res.y) return;

	// Read rayhit from array
	uint32_t id = loc.y * res.x + loc.x;
	RayHit hit = rayHits[id];

	vec4 color = vec4(hit.u, hit.v, hit.t, 1.0f);
	surf2Dwrite(toFloat4(color), surface, loc.x * sizeof(float4), loc.y);
}

void launchGenPrimaryRaysKernel(RayIn* rays, const CameraDef& cam, vec2i res) noexcept
{
	// Calculate number of threads and blocks to run
	dim3 threadsPerBlock(8, 8);
	dim3 numBlocks((res.x + threadsPerBlock.x - 1) / threadsPerBlock.x,
	               (res.y + threadsPerBlock.y - 1) / threadsPerBlock.y);

	// Run cuda ray tracer kernel
	genPrimaryRaysKernel<<<numBlocks, threadsPerBlock>>>(rays, cam, res);
	CHECK_CUDA_ERROR(cudaGetLastError());
	CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

void launchGenSecondaryRaysKernel(RayIn* rays, vec3 camPos, vec2i res, cudaSurfaceObject_t posTex,
                                  cudaSurfaceObject_t normalTex) noexcept
{
	// Calculate number of threads and blocks to run
	dim3 threadsPerBlock(8, 8);
	dim3 numBlocks((res.x + threadsPerBlock.x - 1) / threadsPerBlock.x,
	               (res.y + threadsPerBlock.y - 1) / threadsPerBlock.y);

	// Run cuda ray tracer kernel
	genSecondaryRaysKernel<<<numBlocks, threadsPerBlock>>>(rays, camPos, res, posTex, normalTex);

	CHECK_CUDA_ERROR(cudaGetLastError());
	CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

void launchWriteRayHitsToScreenKernel(cudaSurfaceObject_t surface, vec2i res, const RayHit* rayHits) noexcept
{
	// Calculate number of threads and blocks to run
	dim3 threadsPerBlock(8, 8);
	dim3 numBlocks((res.x + threadsPerBlock.x - 1) / threadsPerBlock.x,
	               (res.y + threadsPerBlock.y - 1) / threadsPerBlock.y);

	// Run cuda ray tracer kernel
	writeRayHitsToScreenKernel<<<numBlocks, threadsPerBlock>>>(surface, res, rayHits);
	CHECK_CUDA_ERROR(cudaGetLastError());
	CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

} // namespace phe
