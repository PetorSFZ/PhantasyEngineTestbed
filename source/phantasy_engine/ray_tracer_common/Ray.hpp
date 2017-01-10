// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include <sfz/CudaCompatibility.hpp>
#include <sfz/math/Vector.hpp>

namespace phe {

using sfz::vec3;

struct Ray {
	vec3 origin;
	vec3 dir;
	vec3 invDir;

	Ray() noexcept = default;
	Ray(const Ray&) noexcept = default;
	Ray& operator= (const Ray&) noexcept = default;
	~Ray() noexcept = default;

	SFZ_CUDA_CALL Ray(const vec3& originIn, const vec3& directionIn) noexcept
	{
		setOrigin(originIn);
		setDir(directionIn);
	}

	SFZ_CUDA_CALL void setOrigin(const vec3& originIn) noexcept
	{
		this->origin = originIn;
	}

	SFZ_CUDA_CALL void setDir(const vec3& dirIn) noexcept
	{
		this->dir = dirIn;
		this->invDir = vec3(1.0f) / dirIn;
	}
};

} // namespace phe
