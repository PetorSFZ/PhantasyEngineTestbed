// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include <sfz/CudaCompatibility.hpp>
#include <sfz/math/Vector.hpp>

#include "cuda_runtime.h"

// CUDA Vector -> sfz::Vector
// ------------------------------------------------------------------------------------------------

SFZ_CUDA_CALL float2 toFloat2(const sfz::vec2& v) noexcept
{
	return make_float2(v.x, v.y);
}

SFZ_CUDA_CALL float3 toFloat3(const sfz::vec3& v) noexcept
{
	return make_float3(v.x, v.y, v.z);
}

SFZ_CUDA_CALL float4 toFloat4(const sfz::vec4& v) noexcept
{
	return make_float4(v.x, v.y, v.z, v.w);
}

SFZ_CUDA_CALL int2 toInt2(const sfz::vec2i& v) noexcept
{
	return make_int2(v.x, v.y);
}

SFZ_CUDA_CALL int3 toInt3(const sfz::vec3i& v) noexcept
{
	return make_int3(v.x, v.y, v.z);
}

SFZ_CUDA_CALL int4 toInt4(const sfz::vec4i& v) noexcept
{
	return make_int4(v.x, v.y, v.z, v.w);
}

// sfz::Vector -> CUDA Vector
// ------------------------------------------------------------------------------------------------

SFZ_CUDA_CALL sfz::vec2 toSFZ(const float2& v) noexcept
{
	return sfz::vec2(v.x, v.y);
}

SFZ_CUDA_CALL sfz::vec3 toSFZ(const float3& v) noexcept
{
	return sfz::vec3(v.x, v.y, v.z);
}

SFZ_CUDA_CALL sfz::vec4 toSFZ(const float4& v) noexcept
{
	return sfz::vec4(v.x, v.y, v.z, v.w);
}

SFZ_CUDA_CALL sfz::vec2i toSFZ(const int2& v) noexcept
{
	return sfz::vec2i(v.x, v.y);
}

SFZ_CUDA_CALL sfz::vec3i toSFZ(const int3& v) noexcept
{
	return sfz::vec3i(v.x, v.y, v.z);
}

SFZ_CUDA_CALL sfz::vec4i toSFZ(const int4& v) noexcept
{
	return sfz::vec4i(v.x, v.y, v.z, v.w);
}
