// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#include "kernels/GenSecondaryRaysKernel.hpp"

#include <cfloat>

#include "CudaHelpers.hpp"
#include "CudaDeviceHelpers.cuh"
#include "GBufferRead.cuh"

namespace phe {

// GenSecondaryRaysKernel
// ------------------------------------------------------------------------------------------------

static __global__ void genSecondaryRaysKernel(GenSecondaryRaysKernelInput input,
                                              RayIn* __restrict__ secondaryRays,
                                              RayIn* __restrict__ shadowRays)
{
	// Calculate surface coordinates
	const vec2u halfRes = input.res / 2u;
	vec2u halfResLoc = vec2u(blockIdx.x * blockDim.x + threadIdx.x,
	                         blockIdx.y * blockDim.y + threadIdx.y);
	if (halfResLoc.x >= halfRes.x || halfResLoc.y >= halfRes.y) return;

	// Calculate 4 corresponding fullscreen locations
	vec2u loc1 = halfResLoc * 2u;
	vec2u loc2 = loc1 + vec2u(1u, 0u);
	vec2u loc3 = loc1 + vec2u(0u, 1u);
	vec2u loc4 = loc1 + vec2u(1u, 1u);

	// Read GBuffer values for the 4 pixel block
	GBufferValue gval1 = readGBuffer(input.posTex, input.normalTex, input.albedoTex,
	                                input.materialTex, loc1);
	GBufferValue gval2 = readGBuffer(input.posTex, input.normalTex, input.albedoTex,
	                                input.materialTex, loc2);
	GBufferValue gval3 = readGBuffer(input.posTex, input.normalTex, input.albedoTex,
	                                input.materialTex, loc3);
	GBufferValue gval4 = readGBuffer(input.posTex, input.normalTex, input.albedoTex,
	                                input.materialTex, loc4);

	// Take average or now, but should probably do something stochastic.
	// Maybe ignore values if they are very far away from each other
	vec3 pos = (gval1.pos + gval2.pos + gval3.pos + gval4.pos) / 4.0f;
	vec3 normal = normalize((gval1.normal + gval2.normal + gval3.normal + gval4.normal) / 4.0f);

	// Calculate reflect direction
	vec3 camDir = normalize(pos - input.camPos);
	vec3 reflected = reflect(camDir, normal);

	// Create secondary ray
	RayIn secondaryRay;
	secondaryRay.setDir(reflected);
	secondaryRay.setOrigin(pos);
	secondaryRay.setMinDist(0.0001f);
	secondaryRay.setMaxDist(FLT_MAX);

	// Write secondary ray to array
	uint32_t secondaryRayIdx = halfResLoc.y * halfRes.x + halfResLoc.x;
	secondaryRays[secondaryRayIdx] = secondaryRay;

	// Calculate base indices for shadow rays 
	const uint32_t totalNumLightSources = input.numStaticSphereLights;
	uint32_t baseIdx1 = (loc1.y * input.res.x + loc1.x) * totalNumLightSources;
	uint32_t baseIdx2 = (loc2.y * input.res.x + loc2.x) * totalNumLightSources;
	uint32_t baseIdx3 = (loc3.y * input.res.x + loc3.x) * totalNumLightSources;
	uint32_t baseIdx4 = (loc4.y * input.res.x + loc4.x) * totalNumLightSources;

	// Create shadow rays and write
	for (uint32_t i = 0; i < input.numStaticSphereLights; i++) {

		const float EPS = 0.001f;

		// Retrieve static light source
		SphereLight light = input.staticSphereLights[i];
		
		// Shadow ray for 1st pixel
		vec3 toLight1 = light.pos - gval1.pos;
		float toLightDist1 = length(toLight1);
		vec3 toLightDir1 = toLight1 / toLightDist1;
		RayIn shadowRay1;
		if (dot(gval1.normal, toLightDir1) >= 0.0f && toLightDist1 <= light.range) {
			shadowRay1.setDir(toLightDir1);
			shadowRay1.setOrigin(gval1.pos);
			shadowRay1.setMinDist(EPS);
			shadowRay1.setMaxDist(toLightDist1 - EPS);
		}
		else {
			// Light can't contribute light, create dummy ray that is exceptionally easy to cast
			shadowRay1.setOrigin(vec3(1000000.0f, 1000000.0f, 1000000.0f));
			shadowRay1.setDir(vec3(0.0f, 1.0f, 0.0f));
			shadowRay1.setMinDist(1.0f);
			shadowRay1.setMaxDist(0.1f);
		}
		shadowRays[baseIdx1 + i] = shadowRay1;

		// Shadow ray for 2nd pixel
		vec3 toLight2 = light.pos - gval2.pos;
		float toLightDist2 = length(toLight2);
		vec3 toLightDir2 = toLight2 / toLightDist2;
		RayIn shadowRay2;
		if (dot(gval2.normal, toLightDir2) >= 0.0f && toLightDist2 <= light.range) {
			shadowRay2.setDir(toLightDir2);
			shadowRay2.setOrigin(gval2.pos);
			shadowRay2.setMinDist(EPS);
			shadowRay2.setMaxDist(toLightDist2 - EPS);
		}
		else {
			// Light can't contribute light, create dummy ray that is exceptionally easy to cast
			shadowRay2.setOrigin(vec3(1000000.0f, 1000000.0f, 1000000.0f));
			shadowRay2.setDir(vec3(0.0f, 1.0f, 0.0f));
			shadowRay2.setMinDist(1.0f);
			shadowRay2.setMaxDist(0.1f);
		}
		shadowRays[baseIdx2 + i] = shadowRay2;

		// Shadow ray for 3rd pixel
		vec3 toLight3 = light.pos - gval3.pos;
		float toLightDist3 = length(toLight3);
		vec3 toLightDir3 = toLight3 / toLightDist3;
		RayIn shadowRay3;
		if (dot(gval3.normal, toLightDir3) >= 0.0f && toLightDist3 <= light.range) {
			shadowRay3.setDir(toLightDir3);
			shadowRay3.setOrigin(gval3.pos);
			shadowRay3.setMinDist(EPS);
			shadowRay3.setMaxDist(toLightDist3 - EPS);
		}
		else {
			// Light can't contribute light, create dummy ray that is exceptionally easy to cast
			shadowRay3.setOrigin(vec3(1000000.0f, 1000000.0f, 1000000.0f));
			shadowRay3.setDir(vec3(0.0f, 1.0f, 0.0f));
			shadowRay3.setMinDist(1.0f);
			shadowRay3.setMaxDist(0.1f);
		}
		shadowRays[baseIdx3 + i] = shadowRay3;

		// Shadow ray for 4th pixel
		vec3 toLight4 = light.pos - gval4.pos;
		float toLightDist4 = length(toLight4);
		vec3 toLightDir4 = toLight4 / toLightDist4;
		RayIn shadowRay4;
		if (dot(gval4.normal, toLightDir4) >= 0.0f && toLightDist4 <= light.range) {
			shadowRay4.setDir(toLightDir4);
			shadowRay4.setOrigin(gval4.pos);
			shadowRay4.setMinDist(EPS);
			shadowRay4.setMaxDist(toLightDist4 - EPS);
		}
		else {
			// Light can't contribute light, create dummy ray that is exceptionally easy to cast
			shadowRay4.setOrigin(vec3(1000000.0f, 1000000.0f, 1000000.0f));
			shadowRay4.setDir(vec3(0.0f, 1.0f, 0.0f));
			shadowRay4.setMinDist(1.0f);
			shadowRay4.setMaxDist(0.1f);
		}
		shadowRays[baseIdx4 + i] = shadowRay4;
	}

}

// GenSecondaryRaysKernel launch function
// ------------------------------------------------------------------------------------------------

void launchGenSecondaryRaysKernel(const GenSecondaryRaysKernelInput& input,
                                  RayIn* __restrict__ secondaryRays,
                                  RayIn* __restrict__ shadowRays) noexcept
{
	vec2u halfRes = input.res / 2u;
	dim3 threadsPerBlock(8, 8);
	dim3 numBlocks((halfRes.x + threadsPerBlock.x - 1) / threadsPerBlock.x,
	               (halfRes.y + threadsPerBlock.y - 1) / threadsPerBlock.y);

	genSecondaryRaysKernel<<<numBlocks,threadsPerBlock>>>(input, secondaryRays, shadowRays);
	CHECK_CUDA_ERROR(cudaGetLastError());
	CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

} // namespace phe
