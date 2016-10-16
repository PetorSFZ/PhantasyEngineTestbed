// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

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
	return a2 / div;
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

} // namespace phe
