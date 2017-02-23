// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#include "kernels/ProcessGBufferKernel.hpp"

#include <cfloat>

#include <sfz/Cuda.hpp>

#include "CudaSfzVectorCompatibility.cuh"
#include "GBufferRead.cuh"

namespace phe {

using sfz::vec3;
using sfz::vec4;

// Helper functions
// ------------------------------------------------------------------------------------------------

// Assumes both parameters are normalized
inline __device__ vec3 reflect(vec3 in, vec3 normal) noexcept
{
	return in - 2.0f * dot(normal, in) * normal;
}

// Kernels
// ------------------------------------------------------------------------------------------------

static __global__ void createReflectRaysKernel(vec3 camPos, vec2i res,
                                               cudaSurfaceObject_t posTex,
                                               cudaSurfaceObject_t normalTex,
                                               cudaSurfaceObject_t albedoTex,
                                               cudaSurfaceObject_t materialTex,
                                               RayIn* raysOut)
{
	// Calculate surface coordinates
	vec2u loc = vec2u(blockIdx.x * blockDim.x + threadIdx.x,
	                  blockIdx.y * blockDim.y + threadIdx.y);
	if (loc.x >= res.x || loc.y >= res.y) return;

	// Read GBuffer
	GBufferValue pixelVal = readGBuffer(posTex, normalTex, albedoTex, materialTex, loc);

	// Calculate reflect direction
	vec3 camDir = normalize(pixelVal.pos - camPos);
	vec3 reflected = reflect(camDir, pixelVal.normal);

	// Create ray
	RayIn ray;
	ray.setDir(reflected);
	ray.setOrigin(pixelVal.pos);
	ray.setMinDist(0.0001f);
	ray.setMaxDist(FLT_MAX);

	// Write ray to array
	uint32_t id = loc.y * res.x + loc.x;
	raysOut[id] = ray;
}

// Kernel cpu interfaces
// ------------------------------------------------------------------------------------------------

void launchCreateReflectRaysKernel(const CreateReflectRaysInput& input, RayIn* raysOut) noexcept
{
	// Calculate number of threads and blocks to run
	dim3 threadsPerBlock(8, 8);
	dim3 numBlocks((input.res.x + threadsPerBlock.x - 1) / threadsPerBlock.x,
	               (input.res.y + threadsPerBlock.y - 1) / threadsPerBlock.y);

	// Run cuda ray tracer kernel
	createReflectRaysKernel<<<numBlocks, threadsPerBlock>>>(input.camPos, input.res,
	                                                        input.posTex, input.normalTex,
	                                                        input.albedoTex, input.materialTex,
	                                                        raysOut);
	CHECK_CUDA_ERROR(cudaGetLastError());
	CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

} // namespace phe
