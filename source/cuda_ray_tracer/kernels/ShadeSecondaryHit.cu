// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#include "kernels/ShadeSecondaryHit.hpp"

#include "math_constants.h"

#include "CudaHelpers.hpp"
#include "CudaPbr.cuh"

namespace phe {

// ShadeSecondaryHitKernel
// ------------------------------------------------------------------------------------------------

static __global__ void shadeSecondaryHitKernel(ShadeSecondaryHitKernelInput input,
                                               IncomingLight* __restrict__ incomingLightsOut)
{
	// Calculate index in array
	uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= input.numRayHitInfos) return;

	RayIn ray = input.secondaryRays[idx];
	RayHitInfo info = input.rayHitInfos[idx];

	vec3 p = info.position();
	vec3 n = info.normal();
	vec3 albedo = info.albedo();
	float roughness = info.roughness();
	float metallic = info.metallic();

	vec3 v = normalize(ray.origin() - p);

	// Interpolation of normals sometimes makes them face away from the camera. Clamp
	// these to almost zero, to not break shading calculations.
	float nDotV = fmaxf(0.001f, dot(n, v));

	vec3 color = vec3(0.0f);

	uint32_t baseShadowIdx = idx * input.numStaticSphereLights;
	for (uint32_t i = 0; i < input.numStaticSphereLights; i++) {
		
		// Check if light source is occluded or not
		bool inLight = input.shadowRayResults[baseShadowIdx + i];
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

	IncomingLight tmp;
	tmp.setOrigin(p);
	tmp.setAmount(color);
	tmp.setFallofFactor(1.0f); // TODO: Assume that we don't need to scale this by distance
	incomingLightsOut[idx] = tmp;
}

// ShadeSecondaryHitKernel launch function
// ------------------------------------------------------------------------------------------------

void launchShadeSecondaryHitKernel(const ShadeSecondaryHitKernelInput& input,
                                   IncomingLight* __restrict__ incomingLightsOut) noexcept
{
	const uint32_t numThreadsPerBlock = 256;
	uint32_t numBlocks = (input.numRayHitInfos / numThreadsPerBlock) + 1;

	shadeSecondaryHitKernel<<<numBlocks, numThreadsPerBlock>>>(input, incomingLightsOut);
	CHECK_CUDA_ERROR(cudaGetLastError());
	CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

} // namespace phe
