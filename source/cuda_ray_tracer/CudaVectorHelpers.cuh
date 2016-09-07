#pragma once

#include <sfz/math/Vector.hpp>

#include "cuda_runtime.h"

// sfz::Vector compatibility
// ------------------------------------------------------------------------------------------------

inline __device__ __host__ float2 toFloat2(const sfz::vec2& v) noexcept
{
	return make_float2(v.x, v.y);
}

inline __device__ __host__ float3 toFloat3(const sfz::vec3& v) noexcept
{
	return make_float3(v.x, v.y, v.z);
}

inline __device__ __host__ float4 toFloat4(const sfz::vec4& v) noexcept
{
	return make_float4(v.x, v.y, v.z, v.w);
}

inline __device__ __host__ int2 toInt2(const sfz::vec2i& v) noexcept
{
	return make_int2(v.x, v.y);
}

inline __device__ __host__ int3 toInt3(const sfz::vec3i& v) noexcept
{
	return make_int3(v.x, v.y, v.z);
}

inline __device__ __host__ int4 toInt4(const sfz::vec4i& v) noexcept
{
	return make_int4(v.x, v.y, v.z, v.w);
}

// Vector type convert
// ------------------------------------------------------------------------------------------------

inline __device__ __host__ float2 toFloat2(const int2& v) noexcept
{
	return make_float2(float(v.x), float(v.y));
}

inline __device__ __host__ float3 toFloat3(const int3& v) noexcept
{
	return make_float3(float(v.x), float(v.y), float(v.z));
}

inline __device__ __host__ float4 toFloat4(const int4& v) noexcept
{
	return make_float4(float(v.x), float(v.y), float(v.z), float(v.w));
}

inline __device__ __host__ int2 toInt2(const float2& v) noexcept
{
	return make_int2(float(v.x), float(v.y));
}

inline __device__ __host__ int3 toInt3(const float3& v) noexcept
{
	return make_int3(float(v.x), float(v.y), float(v.z));
}

inline __device__ __host__ int4 toInt4(const float4& v) noexcept
{
	return make_int4(float(v.x), float(v.y), float(v.z), float(v.w));
}

// Vector scalar constructors
// ------------------------------------------------------------------------------------------------

inline __device__ __host__ float2 toFloat2(float scalar) noexcept
{
	return make_float2(scalar, scalar);
}

inline __device__ __host__ float3 toFloat3(float scalar) noexcept
{
	return make_float3(scalar, scalar, scalar);
}

inline __device__ __host__ float4 toFloat4(float scalar) noexcept
{
	return make_float4(scalar, scalar, scalar, scalar);
}

inline __device__ __host__ int2 toInt2(int scalar) noexcept
{
	return make_int2(scalar, scalar);
}

inline __device__ __host__ int3 toint3(int scalar) noexcept
{
	return make_int3(scalar, scalar, scalar);
}

inline __device__ __host__ int4 toInt4(int scalar) noexcept
{
	return make_int4(scalar, scalar, scalar, scalar);
}

// Vector addition (+)
// ------------------------------------------------------------------------------------------------

inline __device__ __host__ float2 operator+ (const float2& lhs, const float2& rhs) noexcept
{
	return make_float2(lhs.x + rhs.x,
	                   lhs.y + rhs.y);
}

inline __device__ __host__ float3 operator+ (const float3& lhs, const float3& rhs) noexcept
{
	return make_float3(lhs.x + rhs.x,
	                   lhs.y + rhs.y,
	                   lhs.z + rhs.z);
}

inline __device__ __host__ float4 operator+ (const float4& lhs, const float4& rhs) noexcept
{
	return make_float4(lhs.x + rhs.x,
	                   lhs.y + rhs.y,
	                   lhs.z + rhs.z,
	                   lhs.w + rhs.w);
}

inline __device__ __host__ int2 operator+ (const int2& lhs, const int2& rhs) noexcept
{
	return make_int2(lhs.x + rhs.x,
	                 lhs.y + rhs.y);
}

inline __device__ __host__ int3 operator+ (const int3& lhs, const int3& rhs) noexcept
{
	return make_int3(lhs.x + rhs.x,
	                 lhs.y + rhs.y,
	                 lhs.z + rhs.z);
}

inline __device__ __host__ int4 operator+ (const int4& lhs, const int4& rhs) noexcept
{
	return make_int4(lhs.x + rhs.x,
	                 lhs.y + rhs.y,
	                 lhs.z + rhs.z,
	                 lhs.w + rhs.w);
}

// Vector scalar addition (+)
// ------------------------------------------------------------------------------------------------

inline __device__ __host__ float2 operator+ (const float2& lhs, float rhs) noexcept
{
	return make_float2(lhs.x + rhs,
	                   lhs.y + rhs);
}

inline __device__ __host__ float2 operator+ (float lhs, const float2& rhs) noexcept
{
	return rhs + lhs;
}

inline __device__ __host__ float3 operator+ (const float3& lhs, float rhs) noexcept
{
	return make_float3(lhs.x + rhs,
	                   lhs.y + rhs,
	                   lhs.z + rhs);
}

inline __device__ __host__ float3 operator+ (float lhs, const float3& rhs) noexcept
{
	return rhs + lhs;
}

inline __device__ __host__ float4 operator+ (const float4& lhs, float rhs) noexcept
{
	return make_float4(lhs.x + rhs,
	                   lhs.y + rhs,
	                   lhs.z + rhs,
	                   lhs.w + rhs);
}

inline __device__ __host__ float4 operator+ (float lhs, const float4& rhs) noexcept
{
	return rhs + lhs;
}

inline __device__ __host__ int2 operator+ (const int2& lhs, int rhs) noexcept
{
	return make_int2(lhs.x + rhs,
	                 lhs.y + rhs);
}

inline __device__ __host__ int2 operator+ (int lhs, const int2& rhs) noexcept
{
	return rhs + lhs;
}

inline __device__ __host__ int3 operator+ (const int3& lhs, int rhs) noexcept
{
	return make_int3(lhs.x + rhs,
	                 lhs.y + rhs,
	                 lhs.z + rhs);
}

inline __device__ __host__ int3 operator+ (int lhs, const int3& rhs) noexcept
{
	return rhs + lhs;
}

inline __device__ __host__ int4 operator+ (const int4& lhs, int rhs) noexcept
{
	return make_int4(lhs.x + rhs,
	                 lhs.y + rhs,
	                 lhs.z + rhs,
	                 lhs.w + rhs);
}

inline __device__ __host__ int4 operator+ (int lhs, const int4& rhs) noexcept
{
	return rhs + lhs;
}

// Vector subtraction (-)
// ------------------------------------------------------------------------------------------------

inline __device__ __host__ float2 operator- (const float2& lhs, const float2& rhs) noexcept
{
	return make_float2(lhs.x - rhs.x,
	                   lhs.y - rhs.y);
}

inline __device__ __host__ float3 operator- (const float3& lhs, const float3& rhs) noexcept
{
	return make_float3(lhs.x - rhs.x,
	                   lhs.y - rhs.y,
	                   lhs.z - rhs.z);
}

inline __device__ __host__ float4 operator- (const float4& lhs, const float4& rhs) noexcept
{
	return make_float4(lhs.x - rhs.x,
	                   lhs.y - rhs.y,
	                   lhs.z - rhs.z,
	                   lhs.w - rhs.w);
}

inline __device__ __host__ int2 operator- (const int2& lhs, const int2& rhs) noexcept
{
	return make_int2(lhs.x - rhs.x,
	                 lhs.y - rhs.y);
}

inline __device__ __host__ int3 operator- (const int3& lhs, const int3& rhs) noexcept
{
	return make_int3(lhs.x - rhs.x,
	                 lhs.y - rhs.y,
	                 lhs.z - rhs.z);
}

inline __device__ __host__ int4 operator- (const int4& lhs, const int4& rhs) noexcept
{
	return make_int4(lhs.x - rhs.x,
	                 lhs.y - rhs.y,
	                 lhs.z - rhs.z,
	                 lhs.w - rhs.w);
}

// Vector scalar subtraction (-)
// ------------------------------------------------------------------------------------------------

inline __device__ __host__ float2 operator- (const float2& lhs, float rhs) noexcept
{
	return make_float2(lhs.x - rhs,
	                   lhs.y - rhs);
}

inline __device__ __host__ float2 operator- (float lhs, const float2& rhs) noexcept
{
	return make_float2(lhs - rhs.x,
	                   lhs - rhs.y);
}

inline __device__ __host__ float3 operator- (const float3& lhs, float rhs) noexcept
{
	return make_float3(lhs.x - rhs,
	                   lhs.y - rhs,
	                   lhs.z - rhs);
}

inline __device__ __host__ float3 operator- (float lhs, const float3& rhs) noexcept
{
	return make_float3(lhs - rhs.x,
	                   lhs - rhs.y,
	                   lhs - rhs.z);
}

inline __device__ __host__ float4 operator- (const float4& lhs, float rhs) noexcept
{
	return make_float4(lhs.x - rhs,
	                   lhs.y - rhs,
	                   lhs.z - rhs,
	                   lhs.w - rhs);
}

inline __device__ __host__ float4 operator- (float lhs, const float4& rhs) noexcept
{
	return make_float4(lhs - rhs.x,
	                   lhs - rhs.y,
	                   lhs - rhs.z,
	                   lhs - rhs.w);
}

inline __device__ __host__ int2 operator- (const int2& lhs, int rhs) noexcept
{
	return make_int2(lhs.x - rhs,
	                 lhs.y - rhs);
}

inline __device__ __host__ int2 operator- (int lhs, const int2& rhs) noexcept
{
	return make_int2(lhs - rhs.x,
	                 lhs - rhs.y);
}

inline __device__ __host__ int3 operator- (const int3& lhs, int rhs) noexcept
{
	return make_int3(lhs.x - rhs,
	                 lhs.y - rhs,
	                 lhs.z - rhs);
}

inline __device__ __host__ int3 operator- (int lhs, const int3& rhs) noexcept
{
	return make_int3(lhs - rhs.x,
	                 lhs - rhs.y,
	                 lhs - rhs.z);
}

inline __device__ __host__ int4 operator- (const int4& lhs, int rhs) noexcept
{
	return make_int4(lhs.x - rhs,
	                 lhs.y - rhs,
	                 lhs.z - rhs,
	                 lhs.w - rhs);
}

inline __device__ __host__ int4 operator- (int lhs, const int4& rhs) noexcept
{
	return make_int4(lhs - rhs.x,
	                 lhs - rhs.y,
	                 lhs - rhs.z,
	                 lhs - rhs.w);
}

// Vector multiplication (*)
// ------------------------------------------------------------------------------------------------

inline __device__ __host__ float2 operator* (const float2& lhs, const float2& rhs) noexcept
{
	return make_float2(lhs.x * rhs.x,
	                   lhs.y * rhs.y);
}

inline __device__ __host__ float3 operator* (const float3& lhs, const float3& rhs) noexcept
{
	return make_float3(lhs.x * rhs.x,
	                   lhs.y * rhs.y,
	                   lhs.z * rhs.z);
}

inline __device__ __host__ float4 operator* (const float4& lhs, const float4& rhs) noexcept
{
	return make_float4(lhs.x * rhs.x,
	                   lhs.y * rhs.y,
	                   lhs.z * rhs.z,
	                   lhs.w * rhs.w);
}

inline __device__ __host__ int2 operator* (const int2& lhs, const int2& rhs) noexcept
{
	return make_int2(lhs.x * rhs.x,
	                 lhs.y * rhs.y);
}

inline __device__ __host__ int3 operator* (const int3& lhs, const int3& rhs) noexcept
{
	return make_int3(lhs.x * rhs.x,
	                 lhs.y * rhs.y,
	                 lhs.z * rhs.z);
}

inline __device__ __host__ int4 operator* (const int4& lhs, const int4& rhs) noexcept
{
	return make_int4(lhs.x * rhs.x,
	                 lhs.y * rhs.y,
	                 lhs.z * rhs.z,
	                 lhs.w * rhs.w);
}

// Vector scalar multiplication (*)
// ------------------------------------------------------------------------------------------------

inline __device__ __host__ float2 operator* (const float2& lhs, float rhs) noexcept
{
	return make_float2(lhs.x * rhs,
	                   lhs.y * rhs);
}

inline __device__ __host__ float2 operator* (float lhs, const float2& rhs) noexcept
{
	return rhs * lhs;
}

inline __device__ __host__ float3 operator* (const float3& lhs, float rhs) noexcept
{
	return make_float3(lhs.x * rhs,
	                   lhs.y * rhs,
	                   lhs.z * rhs);
}

inline __device__ __host__ float3 operator* (float lhs, const float3& rhs) noexcept
{
	return rhs * lhs;
}

inline __device__ __host__ float4 operator* (const float4& lhs, float rhs) noexcept
{
	return make_float4(lhs.x * rhs,
	                   lhs.y * rhs,
	                   lhs.z * rhs,
	                   lhs.w * rhs);
}

inline __device__ __host__ float4 operator* (float lhs, const float4& rhs) noexcept
{
	return rhs * lhs;
}

inline __device__ __host__ int2 operator* (const int2& lhs, int rhs) noexcept
{
	return make_int2(lhs.x * rhs,
	                 lhs.y * rhs);
}

inline __device__ __host__ int2 operator* (int lhs, const int2& rhs) noexcept
{
	return rhs * lhs;
}

inline __device__ __host__ int3 operator* (const int3& lhs, int rhs) noexcept
{
	return make_int3(lhs.x * rhs,
	                 lhs.y * rhs,
	                 lhs.z * rhs);
}

inline __device__ __host__ int3 operator* (int lhs, const int3& rhs) noexcept
{
	return rhs * lhs;
}

inline __device__ __host__ int4 operator* (const int4& lhs, int rhs) noexcept
{
	return make_int4(lhs.x * rhs,
	                 lhs.y * rhs,
	                 lhs.z * rhs,
	                 lhs.w * rhs);
}

inline __device__ __host__ int4 operator* (int lhs, const int4& rhs) noexcept
{
	return rhs * lhs;
}

// Vector division (/)
// ------------------------------------------------------------------------------------------------

inline __device__ __host__ float2 operator/ (const float2& lhs, const float2& rhs) noexcept
{
	return make_float2(lhs.x / rhs.x,
	                   lhs.y / rhs.y);
}

inline __device__ __host__ float3 operator/ (const float3& lhs, const float3& rhs) noexcept
{
	return make_float3(lhs.x / rhs.x,
	                   lhs.y / rhs.y,
	                   lhs.z / rhs.z);
}

inline __device__ __host__ float4 operator/ (const float4& lhs, const float4& rhs) noexcept
{
	return make_float4(lhs.x / rhs.x,
	                   lhs.y / rhs.y,
	                   lhs.z / rhs.z,
	                   lhs.w / rhs.w);
}

inline __device__ __host__ int2 operator/ (const int2& lhs, const int2& rhs) noexcept
{
	return make_int2(lhs.x / rhs.x,
	                 lhs.y / rhs.y);
}

inline __device__ __host__ int3 operator/ (const int3& lhs, const int3& rhs) noexcept
{
	return make_int3(lhs.x / rhs.x,
	                 lhs.y / rhs.y,
	                 lhs.z / rhs.z);
}

inline __device__ __host__ int4 operator/ (const int4& lhs, const int4& rhs) noexcept
{
	return make_int4(lhs.x / rhs.x,
	                 lhs.y / rhs.y,
	                 lhs.z / rhs.z,
	                 lhs.w / rhs.w);
}

// Vector scalar division (/)
// ------------------------------------------------------------------------------------------------

inline __device__ __host__ float2 operator/ (const float2& lhs, float rhs) noexcept
{
	return make_float2(lhs.x / rhs,
	                   lhs.y / rhs);
}

inline __device__ __host__ float2 operator/ (float lhs, const float2& rhs) noexcept
{
	return make_float2(lhs / rhs.x,
	                   lhs / rhs.y);
}

inline __device__ __host__ float3 operator/ (const float3& lhs, float rhs) noexcept
{
	return make_float3(lhs.x / rhs,
	                   lhs.y / rhs,
	                   lhs.z / rhs);
}

inline __device__ __host__ float3 operator/ (float lhs, const float3& rhs) noexcept
{
	return make_float3(lhs / rhs.x,
	                   lhs / rhs.y,
	                   lhs / rhs.z);
}

inline __device__ __host__ float4 operator/ (const float4& lhs, float rhs) noexcept
{
	return make_float4(lhs.x / rhs,
	                   lhs.y / rhs,
	                   lhs.z / rhs,
	                   lhs.w / rhs);
}

inline __device__ __host__ float4 operator/ (float lhs, const float4& rhs) noexcept
{
	return make_float4(lhs / rhs.x,
	                   lhs / rhs.y,
	                   lhs / rhs.z,
	                   lhs / rhs.w);
}

inline __device__ __host__ int2 operator/ (const int2& lhs, int rhs) noexcept
{
	return make_int2(lhs.x / rhs,
	                 lhs.y / rhs);
}

inline __device__ __host__ int2 operator/ (int lhs, const int2& rhs) noexcept
{
	return make_int2(lhs / rhs.x,
	                 lhs / rhs.y);
}

inline __device__ __host__ int3 operator/ (const int3& lhs, int rhs) noexcept
{
	return make_int3(lhs.x / rhs,
	                 lhs.y / rhs,
	                 lhs.z / rhs);
}

inline __device__ __host__ int3 operator/ (int lhs, const int3& rhs) noexcept
{
	return make_int3(lhs / rhs.x,
	                 lhs / rhs.y,
	                 lhs / rhs.z);
}

inline __device__ __host__ int4 operator/ (const int4& lhs, int rhs) noexcept
{
	return make_int4(lhs.x / rhs,
	                 lhs.y / rhs,
	                 lhs.z / rhs,
	                 lhs.w / rhs);
}

inline __device__ __host__ int4 operator/ (int lhs, const int4& rhs) noexcept
{
	return make_int4(lhs / rhs.x,
	                 lhs / rhs.y,
	                 lhs / rhs.z,
	                 lhs / rhs.w);
}

// Functions
// ------------------------------------------------------------------------------------------------

inline __device__ __host__ float dot(const float3& left, const float3& right) noexcept
{
	return left.x * right.x
	     + left.y * right.y
	     + left.z * right.z;
}

inline __device__ __host__ float dot(const float4& left, const float4& right) noexcept
{
	return left.x * right.x
	     + left.y * right.y
	     + left.z * right.z
	     + left.w * right.w;
}

inline __device__ __host__ float length(const float3& vector) noexcept
{
	return sqrt(dot(vector, vector));
}

inline __device__ __host__ float length(const float4& vector) noexcept
{
	return sqrt(dot(vector, vector));
}

inline __device__ __host__ float3 normalize(const float3& vector) noexcept
{
	float lengthTmp = length(vector);
	return vector / lengthTmp;
}

inline __device__ __host__ float4 normalize(const float4& vector) noexcept
{
	float lengthTmp = length(vector);
	return vector / lengthTmp;
}
