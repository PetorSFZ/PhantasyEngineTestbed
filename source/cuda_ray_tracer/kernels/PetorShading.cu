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
	vec3 albedo = val.albedo;
	float roughness = val.roughness;
	float metallic = val.metallic;

	vec3 v = normalize(input.camPos - p); // to view

	// Interpolation of normals sometimes makes them face away from the camera. Clamp
	// these to almost zero, to not break shading calculations.
	float nDotV = fmaxf(0.001f, dot(n, v));

	vec3 color = vec3(0.0f);

	uint32_t baseShadowIdx = (loc.y * input.res.x + loc.x) * input.numStaticSphereLights;

	for (uint32_t i = 0; i < input.numStaticSphereLights; i++) {

		// Check if in shadow or not
		bool inLight = input.inLights[baseShadowIdx + i];
		if (!inLight) continue;

		// Retrieve light source
		SphereLight light = input.staticSphereLights[i];
		vec3 toLight = light.pos - p;
		float toLightDist = length(toLight);
		
		// Check if surface is in range of light source
		if (toLightDist > light.range) continue;

		// Shading parameters
		vec3 l = toLight / toLightDist; // to light (normalized)
		vec3 h = normalize(l + v); // half vector (normal of microfacet)
		
		// If nDotL is <= 0 then the light source is not in the hemisphere of the surface, i.e.
		// no shading needs to be performed
		float nDotL = dot(n, l);
		if (nDotL <= 0.0f) continue;

		// Lambert diffuse
		vec3 diffuse = albedo / float(CUDART_PI);

		// Cook-Torrance specular
		// Normal distribution function
		float nDotH = fmaxf(0.0f, dot(n, h)); // max() should be superfluous here
		float ctD = ggx(nDotH, roughness * roughness);

		// Geometric self-shadowing term
		float k = powf(roughness + 1.0f, 2.0f) / 8.0f;
		float ctG = geometricSchlick(nDotL, nDotV, k);

		// Fresnel function
		// Assume all dielectrics have a f0 of 0.04, for metals we assume f0 == albedo
		vec3 f0 = lerp(vec3(0.04f), albedo, metallic);
		vec3 ctF = fresnelSchlick(nDotL, f0);

		// Calculate final Cook-Torrance specular value
		vec3 specular = ctD * ctF * ctG / (4.0f * nDotL * nDotV);

		// Calculates light strength
		float fallofNumerator = powf(clamp(1.0f - powf(toLightDist / light.range, 4.0f), 0.0f, 1.0f), 2.0f);
		float fallofDenominator = (toLightDist * toLightDist + 1.0);
		float falloff = fallofNumerator / fallofDenominator;
		vec3 lightContrib = falloff * light.strength;

		// "Solves" reflectance equation under the assumption that the light source is a point light
		// and that there is no global illumination.
		vec3 res = (diffuse + specular) * lightContrib * nDotL;
		color += res;
	}

	// Rayhit
	int baseIdx = (loc.y * input.res.x + loc.x) * input.numIncomingLights;

	IncomingLight info = input.incomingLights[baseIdx];


	//if (info.wasHit()) {
		//color += 0.2f * (1.0f - roughness) * info.albedo();
		color += 0.2f * (1.0f - roughness) * info.amount();
	//}

	//color = info.amount();

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
