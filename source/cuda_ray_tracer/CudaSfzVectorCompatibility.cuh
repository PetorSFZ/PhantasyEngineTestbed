// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include <sfz/CUDACallable.hpp>
#include <sfz/math/Vector.hpp>

#include "cuda_runtime.h"

// CUDA Vector -> sfz::Vector
// ------------------------------------------------------------------------------------------------

SFZ_CUDA_CALLABLE float2 toFloat2(const sfz::vec2& v) noexcept
{
	return make_float2(v.x, v.y);
}

SFZ_CUDA_CALLABLE float3 toFloat3(const sfz::vec3& v) noexcept
{
	return make_float3(v.x, v.y, v.z);
}

SFZ_CUDA_CALLABLE float4 toFloat4(const sfz::vec4& v) noexcept
{
	return make_float4(v.x, v.y, v.z, v.w);
}

SFZ_CUDA_CALLABLE int2 toInt2(const sfz::vec2i& v) noexcept
{
	return make_int2(v.x, v.y);
}

SFZ_CUDA_CALLABLE int3 toInt3(const sfz::vec3i& v) noexcept
{
	return make_int3(v.x, v.y, v.z);
}

SFZ_CUDA_CALLABLE int4 toInt4(const sfz::vec4i& v) noexcept
{
	return make_int4(v.x, v.y, v.z, v.w);
}

// sfz::Vector -> CUDA Vector
// ------------------------------------------------------------------------------------------------

SFZ_CUDA_CALLABLE sfz::vec2 toFloat2(const float2& v) noexcept
{
	return sfz::vec2(v.x, v.y);
}

SFZ_CUDA_CALLABLE sfz::vec3 toFloat3(const float3& v) noexcept
{
	return sfz::vec3(v.x, v.y, v.z);
}

SFZ_CUDA_CALLABLE sfz::vec4 toFloat4(const float4& v) noexcept
{
	return sfz::vec4(v.x, v.y, v.z, v.w);
}

SFZ_CUDA_CALLABLE sfz::vec2i toInt2(const int2& v) noexcept
{
	return sfz::vec2i(v.x, v.y);
}

SFZ_CUDA_CALLABLE sfz::vec3i toInt3(const int3& v) noexcept
{
	return sfz::vec3i(v.x, v.y, v.z);
}

SFZ_CUDA_CALLABLE sfz::vec4i toInt4(const int4& v) noexcept
{
	return sfz::vec4i(v.x, v.y, v.z, v.w);
}
