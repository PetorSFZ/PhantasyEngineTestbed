// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include <cstdint>

#include <host_defines.h>

#include <sfz/math/Vector.hpp>

namespace phe {

using std::uint32_t;
using sfz::vec2i;
using sfz::vec2u;
using sfz::vec3;
using sfz::vec3i;
using sfz::vec3u;

// CUDA device code helper functions
// ------------------------------------------------------------------------------------------------

// https://devblogs.nvidia.com/parallelforall/lerp-faster-cuda/
inline __device__ float lerp(float v0, float v1, float t) noexcept
{
	return fma(t, v1, fma(-t, v0, v0));
}

inline __device__ vec3 lerp(const vec3& v0, const vec3& v1, float t) noexcept
{
	vec3 tmp;
	tmp.x = lerp(v0.x, v1.x, t);
	tmp.y = lerp(v0.y, v1.y, t);
	tmp.z = lerp(v0.z, v1.z, t);
	return tmp;
}

inline __device__ float clamp(float val, float min, float max) noexcept
{
	return fminf(fmaxf(val, min), max);
}

// Assumes both parameters are normalized
inline __device__ vec3 reflect(vec3 in, sfz::vec3 normal) noexcept
{
	return in - 2.0f * sfz::dot(normal, in) * normal;
}

inline __device__ uint32_t getRayIdx(vec2u res, dim3 inBlockDim, vec2u loc) noexcept
{
	const vec2u vecBlockDim(inBlockDim.x, inBlockDim.y);
	const uint32_t pixBlockSize = vecBlockDim.x * vecBlockDim.y;
	vec2u locDim = loc / vecBlockDim;
	vec2u locFrac;
	locFrac.x = loc.x % vecBlockDim.x;
	locFrac.y = loc.y % vecBlockDim.y;

	uint32_t pixBlockIdx = locDim.y * (res.x / vecBlockDim.x) + locDim.x;
	uint32_t blockRelativeIdx = locFrac.y * vecBlockDim.x + locFrac.x;

	return pixBlockIdx * pixBlockSize + blockRelativeIdx;
}

} // namespace phe
