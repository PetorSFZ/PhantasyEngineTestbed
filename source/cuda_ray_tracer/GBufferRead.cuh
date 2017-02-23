// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include <cstdint>

#include <sfz/math/Vector.hpp>

namespace phe {

using std::uint32_t;
using sfz::vec2u;
using sfz::vec3;

inline __device__ vec3 linearize(vec3 rgbGamma) noexcept
{
	rgbGamma.x = powf(rgbGamma.x, 2.2f);
	rgbGamma.y = powf(rgbGamma.y, 2.2f);
	rgbGamma.z = powf(rgbGamma.z, 2.2f);
	return rgbGamma;
}

inline __device__ vec4 linearize(vec4 rgbGamma) noexcept
{
	rgbGamma.x = powf(rgbGamma.x, 2.2f);
	rgbGamma.y = powf(rgbGamma.y, 2.2f);
	rgbGamma.z = powf(rgbGamma.z, 2.2f);
	rgbGamma.w = powf(rgbGamma.w, 2.2f);
	return rgbGamma;
}

struct GBufferValue final {
	vec4 albedo;
	vec3 pos;
	vec3 normal;
	float roughness;
	float metallic;
};

__device__ inline GBufferValue readGBuffer(cudaSurfaceObject_t posTex,
                                           cudaSurfaceObject_t normalTex,
                                           cudaSurfaceObject_t albedoTex,
                                           cudaSurfaceObject_t materialTex,
                                           vec2u loc) noexcept
{
	float4 posTmp = surf2Dread<float4>(posTex, loc.x * sizeof(float4), loc.y);
	float4 normalTmp = surf2Dread<float4>(normalTex, loc.x * sizeof(float4), loc.y);
	uchar4 albedoTmp = surf2Dread<uchar4>(albedoTex, loc.x * sizeof(uchar4), loc.y);
	float4 materialTmp = surf2Dread<float4>(materialTex, loc.x * sizeof(float4), loc.y);

	GBufferValue tmp;
	tmp.pos = vec3(posTmp.x, posTmp.y, posTmp.z);
	tmp.normal = vec3(normalTmp.x, normalTmp.y, normalTmp.z);
	tmp.albedo = linearize(vec4(albedoTmp.x, albedoTmp.y, albedoTmp.z, albedoTmp.w) / vec4(255.0f));
	tmp.roughness = materialTmp.x;
	tmp.metallic = materialTmp.y;
	return tmp;
}

} // namespace phe
