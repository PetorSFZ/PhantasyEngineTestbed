// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include <sfz/CudaCompatibility.hpp>
#include <sfz/math/Vector.hpp>
#include <sfz/math/MathSupport.hpp>

namespace phe {

using sfz::vec3;

// Helper functions
// ------------------------------------------------------------------------------------------------

SFZ_CUDA_CALL vec3 reflect(const vec3& u, const vec3& v) noexcept
{
	return u - 2.0f * dot(u, v) * v;
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
SFZ_CUDA_CALL float ggx(float nDotH, float a)
{
	float a2 = a * a;
	float div = sfz::PI * pow(nDotH * nDotH * (a2 - 1.0f) + 1.0f, 2);
	return a2 / div;
}

// Schlick's model adjusted to fit Smith's method
// k = a/2, where a = roughness^2, however, for analytical light sources (non image based)
// roughness is first remapped to roughness = (roughnessOrg + 1) / 2.
// Essentially, for analytical light sources:
// k = (roughness + 1)^2 / 8
// For image based lighting:
// k = roughness^2 / 2
SFZ_CUDA_CALL float geometricSchlick(float nDotL, float nDotV, float k)
{
	float g1 = nDotL / (nDotL * (1.0f - k) + k);
	float g2 = nDotV / (nDotV * (1.0f - k) + k);
	return g1 * g2;
}

// Schlick's approximation. F0 should typically be 0.04 for dielectrics
SFZ_CUDA_CALL vec3 fresnelSchlick(float nDotL, const vec3& f0)
{
	return f0 + (vec3(1.0f) - f0) * sfz::clamp(std::pow(1.0f - nDotL, 5.0f), 0.0f, 1.0f);
}

} // namespace phe
