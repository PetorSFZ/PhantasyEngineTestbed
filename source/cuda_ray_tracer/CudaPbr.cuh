// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include <sfz/math/Vector.hpp>

#include <math_constants.h>

#include "CudaDeviceHelpers.cuh"
#include "CudaSfzVectorCompatibility.cuh"

namespace phe {

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
inline __device__ float ggx(float nDotH, float a) noexcept
{
	float a2 = a * a;
	float div = CUDART_PI * powf(nDotH * nDotH * (a2 - 1.0f) + 1.0f, 2.0f);
	return a2 / std::max(div, 0.0001f);
}

// Schlick's model adjusted to fit Smith's method
// k = a/2, where a = roughness^2, however, for analytical light sources (non image based)
// roughness is first remapped to roughness = (roughnessOrg + 1) / 2.
// Essentially, for analytical light sources:
// k = (roughness + 1)^2 / 8
// For image based lighting:
// k = roughness^2 / 2
inline __device__ float geometricSchlick(float nDotL, float nDotV, float k) noexcept
{
	float g1 = nDotL / (nDotL * (1.0f - k) + k);
	float g2 = nDotV / (nDotV * (1.0f - k) + k);
	return g1 * g2;
}

// Schlick's approximation. F0 should typically be 0.04 for dielectrics
inline __device__ vec3 fresnelSchlick(float nDotL, vec3 f0) noexcept
{
	return f0 + (vec3(1.0f) - f0) * clamp(powf(1.0f - nDotL, 5.0f), 0.0f, 1.0f);
}

// Complete shading function for a single light source
// ------------------------------------------------------------------------------------------------

inline __device__ vec3 shade(const vec3& p, const vec3& n, const vec3& v,
                             const vec3& albedo, float roughness, float metallic,
                             const vec3& l, float toLightDist, const vec3& lightStrength, float lightRange) noexcept
{
	// Check if surface is in range of light source
	if (toLightDist > lightRange) return vec3(0.0f);

	// Shading parameters
	vec3 h = normalize(l + v); // half vector (normal of microfacet)
		
	// If nDotL is <= 0 then the light source is not in the hemisphere of the surface, i.e.
	// no shading needs to be performed
	float nDotL = dot(n, l);
	if (nDotL <= 0.0f) return vec3(0.0f);

	// Interpolation of normals sometimes makes them face away from the camera. Clamp
	// these to almost zero, to not break shading calculations.
	float nDotV = fmaxf(0.001f, dot(n, v));

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

	// Due to some unsolved NaN issues, replace inf,-inf and NaN with black
	vec3 shaded = (diffuse + specular) * lightStrength * nDotL;
	if (!isfinite(shaded.x) || !isfinite(shaded.y) | !isfinite(shaded.z)) {
		shaded = vec3(0.0f);
	}
	return shaded;
}

inline __device__ float falloffFactor(float toLightDist, float lightRange) noexcept
{
	float fallofNumerator = powf(clamp(1.0f - powf(toLightDist / lightRange, 4.0f), 0.0f, 1.0f), 2.0f);
	float fallofDenominator = (toLightDist * toLightDist + 1.0);
	return fallofNumerator / fallofDenominator;
}

} // namespace phe
