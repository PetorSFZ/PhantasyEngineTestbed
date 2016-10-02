// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#include "kernels/ProcessGBufferKernel.hpp"

#include "CudaHelpers.hpp"
#include "CudaSfzVectorCompatibility.cuh"

namespace phe {

using sfz::vec3;
using sfz::vec4;

// Helper functions
// ------------------------------------------------------------------------------------------------

struct GBufferValue final {
	vec3 pos;
	vec3 normal;
	vec2 uv;
	uint16_t materialId;
};

static __device__ GBufferValue readGBuffer(cudaSurfaceObject_t posTex,
                                           cudaSurfaceObject_t normalTex,
                                           cudaSurfaceObject_t materialIdTex,
                                           vec2i loc) noexcept
{
	float4 posTmp = surf2Dread<float4>(posTex, loc.x * sizeof(float4), loc.y);
	float4 normalTmp = surf2Dread<float4>(normalTex, loc.x * sizeof(float4), loc.y);
	ushort1 materialId = surf2Dread<ushort1>(materialIdTex, loc.x * sizeof(ushort1), loc.y);

	GBufferValue tmp;
	tmp.pos = vec3(posTmp.x, posTmp.y, posTmp.z);
	tmp.normal = vec3(normalTmp.x, normalTmp.y, normalTmp.z);
	tmp.uv = vec2(posTmp.w, normalTmp.w);
	tmp.materialId = materialId.x;
	return tmp;
}

// Assumes both parameters are normalized
static __device__ vec3 reflect(vec3 in, vec3 normal) noexcept
{
	return in - 2.0f * dot(normal, in) * normal;
}

// Kernels
// ------------------------------------------------------------------------------------------------

static __global__ void tempWriteColorKernel(cudaSurfaceObject_t surface, vec2i res,
                                            cudaSurfaceObject_t normalTex)
{
	// Calculate surface coordinates
	vec2i loc = vec2i(blockIdx.x * blockDim.x + threadIdx.x,
	                  blockIdx.y * blockDim.y + threadIdx.y);
	if (loc.x >= res.x || loc.y >= res.y) return;
	
	float4 tmp = surf2Dread<float4>(normalTex, loc.x * sizeof(float4), loc.y);
	vec4 color = vec4(1.0f, 0.0f, 0.0f, 1.0f);
	//surf2Dwrite(toFloat4(color), surface, loc.x * sizeof(float4), loc.y)
	surf2Dwrite(tmp, surface, loc.x * sizeof(float4), loc.y);
}

static __global__ void createReflectRaysKernel(vec3 camPos, vec2i res,
                                               cudaSurfaceObject_t posTex,
                                               cudaSurfaceObject_t normalTex,
                                               cudaSurfaceObject_t materialIdTex,
                                               RayIn* raysOut)
{
	// Calculate surface coordinates
	vec2i loc = vec2i(blockIdx.x * blockDim.x + threadIdx.x,
	                  blockIdx.y * blockDim.y + threadIdx.y);
	if (loc.x >= res.x || loc.y >= res.y) return;

	// Read GBuffer
	GBufferValue pixelVal = readGBuffer(posTex, normalTex, materialIdTex, loc);

	// Calculate reflect direction
	vec3 camDir = normalize(pixelVal.pos - camPos);
	vec3 reflected = reflect(camDir, pixelVal.normal);

	// Create ray
	RayIn ray;
	ray.setDir(reflected);
	ray.setOrigin(pixelVal.pos);
	ray.setNoResultOnlyHit(false);
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
	                                                        input.materialIdTex, raysOut);
	CHECK_CUDA_ERROR(cudaGetLastError());
	CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

} // namespace phe
