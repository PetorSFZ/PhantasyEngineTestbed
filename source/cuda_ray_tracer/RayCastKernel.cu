// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#include "RayCastKernel.cuh"

#include "CudaHelpers.hpp"
#include "CudaSfzVectorCompatibility.cuh"

namespace phe {

using sfz::vec2;
using sfz::vec2i;
using sfz::vec3;
using sfz::vec3i;
using sfz::vec4;
using sfz::vec4i;

// Generate primary rays kernel
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

// Write ray hits to surface kernel
// ------------------------------------------------------------------------------------------------

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

// Main ray cast kernel
// ------------------------------------------------------------------------------------------------

__global__ void rayCastKernel(cudaTextureObject_t bvhNodes, cudaTextureObject_t triangleVerts,
                              const RayIn* rays, RayHit* rayHits, uint32_t numRays)
{
	static_assert(sizeof(RayIn) == 32, "RayIn is padded");
	static_assert(sizeof(RayHit) == 16, "RayHitOut is padded");

	const int tid = threadIdx.x;

	RayHit tmp;
	tmp.u = 1.0f;
	tmp.v = 0.0f;
	tmp.t = 0.0f;
	rayHits[tid] = tmp; 
}

// RayCastKernel launch function
// ------------------------------------------------------------------------------------------------

void genPrimaryRays(RayIn* rays, const CameraDef& cam, vec2i res) noexcept
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

void writeRayHitsToScreen(cudaSurfaceObject_t surface, vec2i res, const RayHit* rayHits) noexcept
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

	rayCastKernel<<<numBlocks, threadsPerBlock>>>(bvhNodes, triangleVerts, rays, rayHits, numRays);
	CHECK_CUDA_ERROR(cudaGetLastError());
	CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

} // namespace phe
