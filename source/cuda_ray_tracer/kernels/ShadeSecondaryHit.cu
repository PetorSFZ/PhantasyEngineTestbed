// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#include "kernels/ShadeSecondaryHit.hpp"

#include "CudaHelpers.hpp"

namespace phe {

// ShadeSecondaryHitKernel
// ------------------------------------------------------------------------------------------------

static __global__ void shadeSecondaryHitKernel(ShadeSecondaryHitKernelInput input,
                                               IncomingLight* __restrict__ incomingLightsOut)
{
	// Calculate index in array
	uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= input.numRayHitInfos) return;

	RayHitInfo info = input.rayHitInfos[idx];

	vec3 p = info.position();
	vec3 n = info.normal();
	vec3 albedo = info.albedo();
	float roughness = info.roughness();
	float metallic = info.metallic();


	IncomingLight tmp;
	tmp.setAmount(albedo);
	tmp.setOrigin(p);
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
