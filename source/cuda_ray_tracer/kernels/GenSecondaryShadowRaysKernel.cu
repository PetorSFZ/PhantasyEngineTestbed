// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#include "kernels/GenSecondaryShadowRaysKernel.hpp"

#include "CudaHelpers.hpp"

#include "phantasy_engine/level/SphereLight.hpp"

namespace phe {

// GenSecondaryShadowRaysKernel
// ------------------------------------------------------------------------------------------------

__global__ void genSecondaryShadowRaysKernel(GenSecondaryShadowRaysKernelInput input,
                                             RayIn* __restrict__ raysOut)
{
	// Calculate index in array
	uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= input.numRayHitInfos) return;

	// TODO: Don't need to read the whole RayHitInfo
	RayHitInfo info = input.rayHitInfos[idx];
	vec3 p = info.position();
	vec3 n = info.normal();
	
	uint32_t baseIdx = idx * input.numStaticSphereLights;
	for (uint32_t i = 0; i < input.numStaticSphereLights; i++) {

		SphereLight light = input.staticSphereLights[i];
		
		vec3 toLight = light.pos - p;
		float toLightDist = length(toLight);
		vec3 l = toLight / toLightDist;

		RayIn tmpRay;

		// Check if light source even can contribute light
		if (dot(n, l) >= 0.0f && toLightDist <= light.range) {
			tmpRay.setOrigin(p);
			tmpRay.setDir(l);
			tmpRay.setMinDist(0.001f);
			tmpRay.setMaxDist(toLightDist - 0.001f);
		}

		// Light can't contribute light, create dummy ray that is exceptionally easy to compute
		else {
			tmpRay.setOrigin(vec3(1000000.0f, 1000000.0f, 1000000.0f));
			tmpRay.setDir(vec3(0.0f, 1.0f, 0.0f));
			tmpRay.setMinDist(1.0f);
			tmpRay.setMaxDist(0.1f);
		}

		// Write ray to array
		raysOut[baseIdx + i] = tmpRay;
	}

}

// GenSecondaryShadowRaysKernel launch function
// ------------------------------------------------------------------------------------------------

void launchGenSecondaryShadowRaysKernel(const GenSecondaryShadowRaysKernelInput& input,
                                        RayIn* __restrict__ raysOut) noexcept
{
	const uint32_t numThreadsPerBlock = 256;
	uint32_t numBlocks = (input.numRayHitInfos / numThreadsPerBlock) + 1;

	genSecondaryShadowRaysKernel<<<numBlocks, numThreadsPerBlock>>>(input, raysOut);
	CHECK_CUDA_ERROR(cudaGetLastError());
	CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

} // namespace phe
