// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include <sfz/CUDACallable.hpp>
#include <sfz/math/Vector.hpp>

#include "phantasy_engine/ray_tracer_common/Ray.hpp"

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

// Ray triangle intersection test
// ------------------------------------------------------------------------------------------------

struct TriangleHit final {
	bool hit;
	float t, u, v;
};

// See page 750 in Real-Time Rendering 3
SFZ_CUDA_CALLABLE TriangleHit intersects(const TriangleVertices& tri, const vec3& origin, const vec3& dir) noexcept
{
	TriangleHit result;

	const float EPS = 0.00001f;
	vec3 p0 = tri.v0;
	vec3 p1 = tri.v1;
	vec3 p2 = tri.v2;

	vec3 e1 = p1 - p0;
	vec3 e2 = p2 - p0;
	vec3 q = cross(dir, e2);
	float a = dot(e1, q);
	if (-EPS < a && a < EPS) {
		result.hit = false;
		return result;
	}

	// Backface culling here?
	// dot(cross(e1, e2), dir) <= 0.0 ??

	float f = 1.0f / a;
	vec3 s = origin - p0;
	float u = f * dot(s, q);
	if (u < 0.0f) {
		result.hit = false;
		return result;
	}

	vec3 r = cross(s, e1);
	float v = f * dot(dir, r);
	if (v < 0.0f || (u + v) > 1.0f) {
		result.hit = false;
		return result;
	}

	float t = f * dot(e2, r);

	result.hit = true;
	result.t = t;
	result.u = u;
	result.v = v;
	return result;
}

SFZ_CUDA_CALLABLE TriangleHit intersects(const TriangleVertices& tri, const Ray& ray) noexcept
{
	return intersects(tri, ray.origin, ray.dir);
}

} // namespace phe
