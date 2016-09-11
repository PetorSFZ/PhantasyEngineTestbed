// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include <sfz/CUDACallable.hpp>
#include <sfz/math/Vector.hpp>

namespace phe {

using sfz::vec3;
using sfz::vec2;

// Triangle
// ------------------------------------------------------------------------------------------------

struct TriangleVertices final {
	vec3 v0;
	vec3 v1;
	vec3 v2;
};

static_assert(sizeof(TriangleVertices) == 36, "TrianglePosition is padded");

struct TriangleData final {
	vec3 n0;
	vec3 n1;
	vec3 n2;

	vec2 uv0;
	vec2 uv1;
	vec2 uv2;

	uint32_t materialIndex;
};

static_assert(sizeof(TriangleData) == 64, "TriangleData is padded");

struct TriangleUnused final {
	vec3 v0;
	vec3 v1;
	vec3 v2;

	vec3 n0;
	vec3 n1;
	vec3 n2;

	vec2 uv0;
	vec2 uv1;
	vec2 uv2;

	vec3 albedoValue;
	uint32_t albedoTexIndex;
	float roughness;
	uint32_t roughnessTexIndex;
	float metallic;
	uint32_t metallicTexIndex;
};

static_assert(sizeof(TriangleUnused) == 128, "TriangleUnused is padded");

} // namespace phe
