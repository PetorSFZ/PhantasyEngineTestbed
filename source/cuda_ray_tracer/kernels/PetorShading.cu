// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#include "kernels/PetorShading.hpp"

#include "math_constants.h"

#include "CudaHelpers.hpp"
#include "CudaDeviceHelpers.cuh"
#include "CudaPbr.cuh"
#include "CudaSfzVectorCompatibility.cuh"
#include "GBufferRead.cuh"

namespace phe {

// Static helpers
// ------------------------------------------------------------------------------------------------

static __device__ void writeResult(cudaSurfaceObject_t result, vec2u loc, vec4 value) noexcept
{
	surf2Dwrite(toFloat4(value), result, loc.x * sizeof(float4), loc.y);
}

// GatherRaysShadeKernel
// ------------------------------------------------------------------------------------------------

static __global__ void gatherRaysShadeKernel(GatherRaysShadeKernelInput input,
                                             cudaSurfaceObject_t resultOut)
{
	// Calculate surface coordinates
	vec2u loc = vec2u(blockIdx.x * blockDim.x + threadIdx.x,
	                  blockIdx.y * blockDim.y + threadIdx.y);
	if (loc.x >= input.res.x || loc.y >= input.res.y) return;

	// Read GBuffer
	GBufferValue val = readGBuffer(input.posTex, input.normalTex, input.albedoTex,
	                               input.materialTex, loc);
	vec3 p = val.pos;
	vec3 n = val.normal;
	vec3 albedo = val.albedo.xyz;
	float roughness = val.roughness;
	float metallic = val.metallic;

	vec3 v = normalize(input.camPos - p); // to view

	// Interpolation of normals sometimes makes them face away from the camera. Clamp
	// these to almost zero, to not break shading calculations.
	float nDotV = fmaxf(0.001f, dot(n, v));

	vec3 color = vec3(0.0f);

	// Shade using light sources
	uint32_t baseShadowIdx = (loc.y * input.res.x + loc.x) * input.numStaticSphereLights;
	for (uint32_t i = 0; i < input.numStaticSphereLights; i++) {

		// Check if in shadow or not
		bool inLight = input.inLights[baseShadowIdx + i];
		if (!inLight) continue;

		// Retrieve light source
		SphereLight light = input.staticSphereLights[i];
		vec3 toLight = light.pos - p;
		float toLightDist = length(toLight);
		vec3 l = toLight / toLightDist; 

		// Shade
		vec3 shading = shade(p, n, v, albedo, roughness, metallic, l, toLightDist, light.strength, light.range);
		color += shading * fallofFactor(toLightDist, light.range);
	}

	// Shade using secondary rays
	int baseSecondaryIdx = (loc.y * input.res.x + loc.x) * input.numIncomingLights;
	IncomingLight info = input.incomingLights[baseSecondaryIdx];

	if (sfz::sum(info.amount()) > 0.001f) {
		vec3 toHit = info.origin() - p;
		float toHitDist = length(toHit);
		vec3 toHitDir = toHit / toHitDist;

		vec3 secondaryShading = shade(p, n, v, albedo, roughness, metallic, toHitDir, toHitDist, info.amount(), FLT_MAX);
		color += secondaryShading * fallofFactor(toHitDist, 10.0f);
	}

	

	writeResult(resultOut, loc, vec4(color, 1.0));
}

void launchGatherRaysShadeKernel(const GatherRaysShadeKernelInput& input,
                                 cudaSurfaceObject_t resultOut) noexcept
{
	// Calculate number of threads and blocks to run
	dim3 threadsPerBlock(8, 8);
	dim3 numBlocks((input.res.x + threadsPerBlock.x - 1) / threadsPerBlock.x,
	               (input.res.y + threadsPerBlock.y - 1) / threadsPerBlock.y);


	gatherRaysShadeKernel<<<numBlocks,threadsPerBlock>>>(input, resultOut);
	CHECK_CUDA_ERROR(cudaGetLastError());
	CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

} // namespace phe
