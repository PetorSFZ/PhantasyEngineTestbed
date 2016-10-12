// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include <cstdint>

#include <sfz/math/Vector.hpp>

namespace phe {

using std::uint32_t;
using sfz::vec2i;
using sfz::vec2u;
using sfz::vec3i;
using sfz::vec3u;

struct GBufferValue final {
	vec3 pos;
	vec3 normal;
	vec3 albedo;
	float roughness;
	float metallic;
};

__device__ inline GBufferValue readGBuffer(cudaSurfaceObject_t posTex,
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
	tmp.albedo = vec3(albedoTmp.x, albedoTmp.y, albedoTmp.z) / 255.0f;
	tmp.roughness = materialTmp.x;
	tmp.metallic = materialTmp.y;
	return tmp;
}

} // namespace phe
