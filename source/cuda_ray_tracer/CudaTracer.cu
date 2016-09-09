// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#include "CudaTracer.cuh"

#include <sfz/math/Vector.hpp>

#include "CudaSfzVectorCompatibility.cuh"

#include <math.h>

namespace phe {

using namespace sfz;

inline __device__ vec3 calculateRayDir(const CameraDef& cam, vec2 loc, vec2 surfaceRes) noexcept
{
	vec2 locNormalized = loc / surfaceRes; // [0, 1]
	vec2 centerOffsCoord = locNormalized * 2.0f - vec2(1.0f); // [-1.0, 1.0]
	centerOffsCoord.y = -centerOffsCoord.y;
	vec3 nonNormRayDir = cam.dir + cam.dX * centerOffsCoord.x + cam.dY * centerOffsCoord.y;
	return normalize(nonNormRayDir);
}

__global__ void cudaRayTracerKernel(cudaSurfaceObject_t surface, vec2i surfaceRes, CameraDef cam, BVHNode* bvhNodes, TriangleVertices* triangles)
{
	// Calculate surface coordinates
	vec2i loc = vec2i(blockIdx.x * blockDim.x + threadIdx.x,
	                  blockIdx.y * blockDim.y + threadIdx.y);
	if (loc.x >= surfaceRes.x || loc.y >= surfaceRes.y) return;

	// Calculate ray direction
	vec3 rayDir = calculateRayDir(cam, vec2(loc), vec2(surfaceRes));

	// Write ray dir to texture for now
	float4 data = make_float4(rayDir.x, rayDir.y, rayDir.z, 1.0f);
	surf2Dwrite(data, surface, loc.x * sizeof(float4), loc.y);
}

void runCudaRayTracer(cudaSurfaceObject_t surface, vec2i surfaceRes, const CameraDef& cam, BVHNode* bvhNodes, TriangleVertices* triangles) noexcept
{	
	// Calculate number of threads and blocks to run
	dim3 threadsPerBlock(16, 16);
	dim3 numBlocks((surfaceRes.x + threadsPerBlock.x - 1) / threadsPerBlock.x,
	               (surfaceRes.y + threadsPerBlock.y  - 1) / threadsPerBlock.y);

	// Run cuda ray tracer kernel
	cudaRayTracerKernel<<<numBlocks, threadsPerBlock>>>(surface, surfaceRes, cam, bvhNodes, triangles);
	cudaDeviceSynchronize();
}

} // namespace phe
