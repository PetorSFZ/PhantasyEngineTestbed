// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#include "CudaTracer.cuh"

#include <sfz/math/Vector.hpp>

#include "CudaSfzVectorCompatibility.cuh"

#include <math.h>

namespace phe {

using namespace sfz;

inline __device__ void writeSurface(const cudaSurfaceObject_t& surface, vec2i loc, const vec4& data) noexcept
{
	float4 dataFloat4 = toFloat4(data);
	surf2Dwrite(dataFloat4, surface, loc.x * sizeof(float4), loc.y);
}

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
	Ray ray(cam.origin, rayDir);

	// Ray cast against BVH
	RayCastResult hit = castRay(bvhNodes, triangles, ray);
	if (hit.triangleIndex == ~0u) {
		writeSurface(surface, loc, vec4(0.0f));
		return;
	}

	// Draw depth
	writeSurface(surface, loc, vec4(vec3(hit.t / 10.0f), 1.0));
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