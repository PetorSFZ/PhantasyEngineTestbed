// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#include "RayCastKernel.cuh"

#include <phantasy_engine/ray_tracer_common/BVHNode.hpp>
#include <phantasy_engine/ray_tracer_common/Triangle.hpp>

#include "CudaHelpers.hpp"
#include "CudaSfzVectorCompatibility.cuh"

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

struct TriangleHit final {
	bool hit;
	float t, u, v;
};

// See page 750 in Real-Time Rendering 3
static __device__ TriangleHit rayVsTriangle(const TriangleVertices& tri, const vec3& origin, const vec3& dir) noexcept
{
	TriangleHit result;

	const float EPS = 0.00001f;
	vec3 p0 = tri.v0.xyz;
	vec3 p1 = tri.v1.xyz;
	vec3 p2 = tri.v2.xyz;

	vec3 e1 = p1 - p0;
	vec3 e2 = p2 - p0;
	vec3 q = cross(dir, e2);
	float a = dot(e1, q);
	if (-EPS < a && a < EPS) {
		result.hit = false;
		return result;
	}

	// Backface culling here?
	// dot(cross(e1, e2), dir) <= 0.0 ??

	float f = 1.0f / a;
	vec3 s = origin - p0;
	float u = f * dot(s, q);
	if (u < 0.0f) {
		result.hit = false;
		return result;
	}

	vec3 r = cross(s, e1);
	float v = f * dot(dir, r);
	if (v < 0.0f || (u + v) > 1.0f) {
		result.hit = false;
		return result;
	}

	float t = f * dot(e2, r);

	result.hit = true;
	result.t = t;
	result.u = u;
	result.v = v;
	return result;
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

__global__ void rayCastKernel(cudaTextureObject_t bvhNodes, cudaTextureObject_t triangleVerts,
                              const RayIn* rays, RayHit* rayHits, uint32_t numRays, uint32_t numThreads)
{
	static_assert(sizeof(RayIn) == 32, "RayIn is padded");
	static_assert(sizeof(RayHit) == 16, "RayHitOut is padded");
	
	// Constants
	const int32_t SENTINEL = int32_t(0x7FFFFFFF);
	const uint32_t STACK_SIZE = 128;

	// Retreive thread id
	const int tid = threadIdx.x + blockIdx.x * blockDim.x;
	
	// Create local stack
	int32_t stack[STACK_SIZE];
	
	// Very naive loop over all input rays
	for (uint32_t rayIdx = tid; rayIdx < numRays; rayIdx += numThreads) {
		
		// Retrieve ray information
		const RayIn ray = rays[rayIdx];
		const vec3 origin = ray.origin();
		const vec3 dir = ray.dir();
		const vec3 invDir = vec3(1.0f) / dir;
		const vec3 originDivDir = origin * invDir;
		const float tMin = 0.0001f;
		const float tMax = ray.maxDist();
		const bool noResultOnlyHit = ray.noResultOnlyHit();

		// Reset stack
		stack[0] = SENTINEL;
		uint32_t stackIndex = 0u; // Currently pointing to the topmost element (the sentinel)

		int32_t currentIndex = 0u; // The current index to check, start of with first node

		// Temporary to store the final result in
		RayHit resHit;
		resHit.triangleIndex = ~0u;
		resHit.t = tMax;

		// Traverse through the tree
		while (currentIndex != SENTINEL) {
		
			// Traverse nodes until all threads in warp have found leaf
			int32_t leafIndex = 1; // Positive, so not a leaf
			while (currentIndex != SENTINEL && currentIndex >= 0) { // If currentIndex is negative we have a leaf node
			
				// Load inner node pointed at by currentIndex
				BVHNode node = loadBvhNode(bvhNodes, currentIndex);

				// Perform AABB intersection tests and figure out which children we want to visit
				AABBIsect lcHit = rayVsAaabb(invDir, originDivDir, node.leftChildAABBMin(), node.leftChildAABBMax(), tMin, resHit.t);
				AABBIsect rcHit = rayVsAaabb(invDir, originDivDir, node.rightChildAABBMin(), node.rightChildAABBMax(), tMin, resHit.t);

				bool visitLC = lcHit.tIn <= lcHit.tOut;
				bool visitRC = rcHit.tIn <= rcHit.tOut;

				// If we don't need to visit any children we simply pop a new index from the stack
				if (!visitLC && !visitRC) {
					currentIndex = stack[stackIndex];
					stackIndex -= 1;
				}

				// If we need to visit at least one child
				else {
					int32_t lcIndex = node.leftChildIndexRaw();
					int32_t rcIndex = node.rightChildIndexRaw();

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
					TriangleVertices tri = loadTriangle(triangleVerts, triIndex);
					TriangleHit hit = rayVsTriangle(tri, origin, dir);

					if (hit.hit && hit.t < resHit.t && tMin <= hit.t) {
						resHit.triangleIndex = uint32_t(triIndex);
						resHit.t = hit.t;
						resHit.u = hit.u;
						resHit.v = hit.v;

						if (noResultOnlyHit) {
							// TODO: HMMM
							leafIndex = SENTINEL;
							currentIndex = SENTINEL;
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
				leafIndex = currentIndex;

				// If currentIndex contained a leaf it will be processed, so we pop a new element from the stack into it
				if (currentIndex < 0) {
					currentIndex = stack[stackIndex];
					stackIndex -= 1;
				}
			}
		}

		rayHits[rayIdx] = resHit;
	}
}

// RayCastKernel launch function
// ------------------------------------------------------------------------------------------------

void launchRayCastKernel(cudaTextureObject_t bvhNodes, cudaTextureObject_t triangleVerts,
                         const RayIn* rays, RayHit* rayHits, uint32_t numRays) noexcept
{
	cudaDeviceProp properties;
	cudaGetDeviceProperties(&properties, 0); // TODO: Maybe not device 0?
	int maxNumMultiProcessors = properties.multiProcessorCount;
	int maxNumThreadsPerMultiProcessor = properties.maxThreadsPerMultiProcessor;
	int maxNumThreadsPerBlock = properties.maxThreadsPerBlock;
	vec3i maxGridSize = vec3i(properties.maxGridSize);

	//printf("maxNumMultiProcessors: %i\n", maxNumMultiProcessors);
	//printf("maxNumThreadsPerMultiProcessor: %i\n", maxNumThreadsPerMultiProcessor);
	//printf("maxNumThreadsPerBlock: %i\n", maxNumThreadsPerBlock);
	//printf("maxGridSize: %s\n", toString(maxGridSize).str);

	uint32_t blocksPerMultiprocessor = maxNumThreadsPerMultiProcessor / maxNumThreadsPerBlock;

	uint32_t threadsPerBlock = maxNumThreadsPerMultiProcessor / blocksPerMultiprocessor;
	uint32_t numBlocks = blocksPerMultiprocessor * maxNumMultiProcessors;
	uint32_t numThreads = threadsPerBlock * numBlocks;

	rayCastKernel<<<numBlocks, threadsPerBlock>>>(bvhNodes, triangleVerts, rays, rayHits, numRays, numThreads);
	CHECK_CUDA_ERROR(cudaGetLastError());
	CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

// Secondary helper kernels (for debugging and profiling)
// ------------------------------------------------------------------------------------------------

__device__ vec3 calculatePrimaryRayDir(const CameraDef& cam, vec2 loc, vec2 surfaceRes)
{
	vec2 locNormalized = loc / surfaceRes; // [0, 1]
	vec2 centerOffsCoord = locNormalized * 2.0f - vec2(1.0f); // [-1.0, 1.0]
	vec3 nonNormRayDir = cam.dir + cam.dX * centerOffsCoord.x + cam.dY * centerOffsCoord.y;
	return normalize(nonNormRayDir);
}

__global__ void genPrimaryRaysKernel(RayIn* rays, CameraDef cam, vec2i res)
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

__global__ void writeRayHitsToScreenKernel(cudaSurfaceObject_t surface, vec2i res, const RayHit* rayHits)
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
