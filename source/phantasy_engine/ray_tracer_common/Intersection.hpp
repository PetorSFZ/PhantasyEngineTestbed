// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include <sfz/CUDACallable.hpp>
#include <sfz/math/Vector.hpp>

#include "phantasy_engine/ray_tracer_common/Ray.hpp"
#include "phantasy_engine/ray_tracer_common/Triangle.hpp"

namespace phe {

using sfz::vec3;

// Line vs triangle intersection test
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

// Line vs AABB intersection test
// ------------------------------------------------------------------------------------------------

struct AABBHit final {
	bool hit;
	float tIn, tOut;
};

SFZ_CUDA_CALLABLE AABBHit intersects(const Ray& ray, const vec3& min, const vec3& max) noexcept
{
	vec3 t1 = (min - ray.origin) * ray.invDir;
	vec3 t2 = (max - ray.origin) * ray.invDir;

	float tmin = sfz::maxElement(sfz::min(t1, t2));
	float tmax = sfz::minElement(sfz::max(t1, t2));

	AABBHit tmp;
	tmp.hit = tmax >= tmin;
	tmp.tIn = tmin;
	tmp.tOut = tmax;
	return tmp;
}

} // namespace phe
