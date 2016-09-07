// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#include "CudaTracerEntry.cuh"

#include <sfz/math/Vector.hpp>

#include "CudaVectorHelpers.cuh"

#include <math.h>

struct CameraDefCuda final {
	float3 origin;
	float3 dir;
	float3 dX;
	float3 dY;
};

inline __device__ float3 calculateRayDir(const CameraDefCuda& cam, float2 loc, float2 surfaceRes) noexcept
{
	float2 locNormalized = loc / surfaceRes; // [0, 1]
	float2 centerOffsCoord = locNormalized * 2.0f - 1.0f; // [-1.0, 1.0]
	centerOffsCoord.y = -centerOffsCoord.y;
	float3 nonNormRayDir = cam.dir + cam.dX * centerOffsCoord.x + cam.dY * centerOffsCoord.y;
	return normalize(nonNormRayDir);
}

__global__ void cudaRayTracerKernel(cudaSurfaceObject_t surface, int2 surfaceRes, CameraDefCuda cam)
{
	// Calculate surface coordinates
	int2 loc = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
	                     blockIdx.y * blockDim.y + threadIdx.y);
	if (loc.x >= surfaceRes.x || loc.y >= surfaceRes.y) return;

	// Calculate ray direction
	float3 rayDir = calculateRayDir(cam, toFloat2(loc), toFloat2(surfaceRes));

	// Write ray dir to texture for now
	float4 data = make_float4(rayDir.x, rayDir.y, rayDir.z, 1.0f);
	surf2Dwrite(data, surface, loc.x * sizeof(float4), loc.y);
}

namespace phe {

using sfz::vec3;

void runCudaRayTracer(cudaSurfaceObject_t surface, vec2i surfaceRes, const CameraDef& cam) noexcept
{
	// Convert camera defintion to CUDA primitives
	CameraDefCuda camTmp;
	camTmp.origin = toFloat3(cam.origin);
	camTmp.dir = toFloat3(cam.dir);
	camTmp.dX = toFloat3(cam.dX);
	camTmp.dY = toFloat3(cam.dY);	
	
	// Calculate number of threads and blocks to run
	dim3 threadsPerBlock(16, 16);
	dim3 numBlocks((surfaceRes.x + threadsPerBlock.x - 1) / threadsPerBlock.x,
	               (surfaceRes.y + threadsPerBlock.y  - 1) / threadsPerBlock.y);

	// Run cuda ray tracer kernel
	cudaRayTracerKernel<<<numBlocks, threadsPerBlock>>>(surface, toInt2(surfaceRes), camTmp);
	cudaDeviceSynchronize();
}

} // namespace phe
