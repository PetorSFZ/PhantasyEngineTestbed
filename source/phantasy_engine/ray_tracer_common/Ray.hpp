// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include "phantasy_engine/ray_tracer_common/RayTracerCommonSharedHeader.hpp"

namespace phe {

struct Ray {
	vec3_t origin;
	vec3_t dir;
	vec3_t invDir;

	Ray() noexcept = default;
	Ray(const Ray&) noexcept = default;
	Ray& operator= (const Ray&) noexcept = default;
	~Ray() noexcept = default;

	PHE_CUDA_AVAILABLE Ray(const vec3_t& originIn, const vec3_t& directionIn) noexcept
	{
		this->origin = originIn;
		this->dir = directionIn;
	}

	PHE_CUDA_AVAILABLE void setOrigin(const vec3_t& originIn) noexcept
	{
		this->origin = originIn;
	}

	PHE_CUDA_AVAILABLE void setDir(const vec3_t& dirIn) noexcept
	{
		this->dir = dirIn;
		this->invDir.x = 1.0f / dirIn.x;
		this->invDir.y = 1.0f / dirIn.y;
		this->invDir.z = 1.0f / dirIn.z;
	}
};

} // namespace phe
