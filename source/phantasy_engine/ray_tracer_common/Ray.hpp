// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include <sfz/math/Vector.hpp>

namespace phe {

using sfz::vec3;

struct Ray
{
	Ray() noexcept = default;
	Ray(const Ray&) noexcept = default;
	Ray& operator= (const Ray&) noexcept = default;
	~Ray() noexcept = default;

	inline Ray(const vec3& origin, const vec3& direction) noexcept;

	inline void setOrigin(const vec3& origin) noexcept;
	inline void setDir(const vec3& dir) noexcept;

	vec3 origin;
	vec3 dir;
	vec3 invDir;
};

} // namespace phe

#include "phantasy_engine/ray_tracer_common/Ray.inl"
