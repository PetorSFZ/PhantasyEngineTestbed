// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#include "kernels/PetorShading.hpp"

#include "math_constants.h"

#include "CudaHelpers.hpp"
#include "CudaSfzVectorCompatibility.cuh"

namespace phe {

// Static helpers
// ------------------------------------------------------------------------------------------------

struct GBufferValue final {
	vec3 pos;
	vec3 normal;
	vec3 albedo;
	float roughness;
	float metallic;
};

static __device__ vec3 linearize(vec3 rgbGamma) noexcept
{
	rgbGamma.x = powf(rgbGamma.x, 2.2f);
	rgbGamma.y = powf(rgbGamma.y, 2.2f);
	rgbGamma.z = powf(rgbGamma.z, 2.2f);
	return rgbGamma;
}

// https://devblogs.nvidia.com/parallelforall/lerp-faster-cuda/
static __device__ float lerp(float v0, float v1, float t) noexcept
{
	return fma(t, v1, fma(-t, v0, v0));
}

static __device__ vec3 lerp(const vec3& v0, const vec3& v1, float t) noexcept
{
	vec3 tmp;
	tmp.x = lerp(v0.x, v1.x, t);
	tmp.y = lerp(v0.y, v1.y, t);
	tmp.z = lerp(v0.z, v1.z, t);
	return tmp;
}

static __device__ float clamp(float val, float min, float max) noexcept
{
	return fminf(fmaxf(val, min), max);
}

static __device__ GBufferValue readGBuffer(cudaSurfaceObject_t posTex,
                                           cudaSurfaceObject_t normalTex,
                                           cudaSurfaceObject_t albedoTex,
                                           cudaSurfaceObject_t materialTex,
                                           vec2i loc) noexcept
{
	float4 posTmp = surf2Dread<float4>(posTex, loc.x * sizeof(float4), loc.y);
	float4 normalTmp = surf2Dread<float4>(normalTex, loc.x * sizeof(float4), loc.y);
	uchar4 albedoTmp = surf2Dread<uchar4>(albedoTex, loc.x * sizeof(uchar4), loc.y);
	float4 materialTmp = surf2Dread<float4>(materialTex, loc.x * sizeof(float4), loc.y);

	GBufferValue tmp;
	tmp.pos = vec3(posTmp.x, posTmp.y, posTmp.z);
	tmp.normal = vec3(normalTmp.x, normalTmp.y, normalTmp.z);
	tmp.albedo = linearize(vec3(albedoTmp.x, albedoTmp.y, albedoTmp.z) / vec3(255.0f));
	tmp.roughness = materialTmp.x;
	tmp.metallic = materialTmp.y;
	return tmp;
}

static __device__ void writeResult(cudaSurfaceObject_t result, vec2i loc, vec4 value) noexcept
{
	surf2Dwrite(toFloat4(value), result, loc.x * sizeof(float4), loc.y);
}

// PBR shading functions
// ------------------------------------------------------------------------------------------------

// References used:
// https://de45xmedrsdbp.cloudfront.net/Resources/files/2013SiggraphPresentationsNotes-26915738.pdf
// http://blog.selfshadow.com/publications/s2016-shading-course/
// http://www.codinglabs.net/article_physically_based_rendering_cook_torrance.aspx
// http://graphicrants.blogspot.se/2013/08/specular-brdf-reference.html

// Normal distribution function, GGX/Trowbridge-Reitz
// a = roughness^2, UE4 parameterization
// dot(n,h) term should be clamped to 0 if negative
static __device__ float ggx(float nDotH, float a) noexcept
{
	float a2 = a * a;
	float div = CUDART_PI * powf(nDotH * nDotH * (a2 - 1.0f) + 1.0f, 2.0f);
	return a2 / div;
}

// Schlick's model adjusted to fit Smith's method
// k = a/2, where a = roughness^2, however, for analytical light sources (non image based)
// roughness is first remapped to roughness = (roughnessOrg + 1) / 2.
// Essentially, for analytical light sources:
// k = (roughness + 1)^2 / 8
// For image based lighting:
// k = roughness^2 / 2
static __device__ float geometricSchlick(float nDotL, float nDotV, float k) noexcept
{
	float g1 = nDotL / (nDotL * (1.0f - k) + k);
	float g2 = nDotV / (nDotV * (1.0f - k) + k);
	return g1 * g2;
}

// Schlick's approximation. F0 should typically be 0.04 for dielectrics
static __device__ vec3 fresnelSchlick(float nDotL, vec3 f0) noexcept
{
	return f0 + (vec3(1.0f) - f0) * clamp(powf(1.0f - nDotL, 5.0f), 0.0f, 1.0f);
}

// ProccessGBufferGenRaysKernel
// ------------------------------------------------------------------------------------------------

void launchProcessGBufferGenRaysKernel(const ProcessGBufferGenRaysInput& input,
                                       RayIn* raysOut) noexcept
{

}

// GatherRaysShadeKernel
// ------------------------------------------------------------------------------------------------

static __global__ void gatherRaysShadeKernel(GatherRaysShadeKernelInput input,
                                             cudaSurfaceObject_t resultOut)
{
	// Calculate surface coordinates
	vec2i loc = vec2i(blockIdx.x * blockDim.x + threadIdx.x,
	                  blockIdx.y * blockDim.y + threadIdx.y);
	if (loc.x >= input.res.x || loc.y >= input.res.y) return;

	// Read GBuffer
	GBufferValue val = readGBuffer(input.posTex, input.normalTex, input.albedoTex,
	                               input.materialTex, loc);
	vec3 p = val.pos;
	vec3 n = val.normal;
	vec3 albedo = val.albedo;;
	float roughness = val.roughness;
	float metallic = val.metallic;

	vec3 v = normalize(input.camPos - p); // to view

	// Interpolation of normals sometimes makes them face away from the camera. Clamp
	// these to almost zero, to not break shading calculations.
	float nDotV = fmaxf(0.001f, dot(n, v));

	vec3 color = vec3(0.0f);

	for (uint32_t i = 0; i < input.numStaticSphereLights; i++) {
		
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
