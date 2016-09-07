// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include <sfz/math/Vector.hpp>

#include "cuda_runtime.h"
#include "math.h"

// sfz::Vector compatibility
// ------------------------------------------------------------------------------------------------

inline __host__ float2 toFloat2(const sfz::vec2& v) noexcept
{
	return make_float2(v.x, v.y);
}

inline __host__ float3 toFloat3(const sfz::vec3& v) noexcept
{
	return make_float3(v.x, v.y, v.z);
}

inline __host__ float4 toFloat4(const sfz::vec4& v) noexcept
{
	return make_float4(v.x, v.y, v.z, v.w);
}

inline __host__ int2 toInt2(const sfz::vec2i& v) noexcept
{
	return make_int2(v.x, v.y);
}

inline __host__ int3 toInt3(const sfz::vec3i& v) noexcept
{
	return make_int3(v.x, v.y, v.z);
}

inline __host__ int4 toInt4(const sfz::vec4i& v) noexcept
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

// Vector (2D) constructors
// ------------------------------------------------------------------------------------------------

inline __device__ __host__ float2 toFloat2(float scalar) noexcept
{
	return make_float2(scalar, scalar);
}

inline __device__ __host__ float2 toFloat2(float x, float y) noexcept
{
	return make_float2(x, y);
}

inline __device__ __host__ int2 toInt2(int scalar) noexcept
{
	return make_int2(scalar, scalar);
}

inline __device__ __host__ int2 toInt2(int x, int y) noexcept
{
	return make_int2(x, y);
}

// Vector (3D) constructors
// ------------------------------------------------------------------------------------------------

inline __device__ __host__ float3 toFloat3(float scalar) noexcept
{
	return make_float3(scalar, scalar, scalar);
}

inline __device__ __host__ float3 toFloat3(float x, float y, float z) noexcept
{
	return make_float3(x, y, z);
}

inline __device__ __host__ float3 toFloat3(const float2& xy, float z) noexcept
{
	return make_float3(xy.x, xy.y, z);
}

inline __device__ __host__ float3 toFloat3(float x, const float2& yz) noexcept
{
	return make_float3(x, yz.x, yz.y);
}

inline __device__ __host__ float3 toFloat3(const float4& v) noexcept
{
	return make_float3(v.x, v.y, v.z);
}

inline __device__ __host__ int3 toInt3(int scalar) noexcept
{
	return make_int3(scalar, scalar, scalar);
}

inline __device__ __host__ int3 toInt3(int x, int y, int z) noexcept
{
	return make_int3(x, y, z);
}

inline __device__ __host__ int3 toInt3(const int2& xy, int z) noexcept
{
	return make_int3(xy.x, xy.y, z);
}

inline __device__ __host__ int3 toInt3(int x, const int2& yz) noexcept
{
	return make_int3(x, yz.x, yz.y);
}

inline __device__ __host__ int3 toInt3(const int4& v) noexcept
{
	return make_int3(v.x, v.y, v.z);
}

// Vector (4D) constructors
// ------------------------------------------------------------------------------------------------

inline __device__ __host__ float4 toFloat4(float scalar) noexcept
{
	return make_float4(scalar, scalar, scalar, scalar);
}

inline __device__ __host__ float4 toFloat4(float x, float y, float z, float w) noexcept
{
	return make_float4(x, y, z, w);
}

inline __device__ __host__ float4 toFloat4(const float3& xyz, float w) noexcept
{
	return make_float4(xyz.x, xyz.y, xyz.z, w);
}

inline __device__ __host__ float4 toFloat4(const float2& xy, float z, float w) noexcept
{
	return make_float4(xy.x, xy.y, z, w);
}

inline __device__ __host__ int4 toInt4(int scalar) noexcept
{
	return make_int4(scalar, scalar, scalar, scalar);
}

inline __device__ __host__ int4 toInt4(int x, int y, int z, int w) noexcept
{
	return make_int4(x, y, z, w);
}

inline __device__ __host__ int4 toInt4(const int3& xyz, int w) noexcept
{
	return make_int4(xyz.x, xyz.y, xyz.z, w);
}

inline __device__ __host__ int4 toInt4(const int2& xy, int z, int w) noexcept
{
	return make_int4(xy.x, xy.y, z, w);
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

// Vector dot()
// ------------------------------------------------------------------------------------------------

inline __device__ __host__ float dot(const float2& left, const float2& right) noexcept
{
	return left.x * right.x
	     + left.y * right.y;
}


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

inline __device__ __host__ int dot(const int2& left, const int2& right) noexcept
{
	return left.x * right.x
	     + left.y * right.y;
}


inline __device__ __host__ int dot(const int3& left, const int3& right) noexcept
{
	return left.x * right.x
	     + left.y * right.y
	     + left.z * right.z;
}

inline __device__ __host__ int dot(const int4& left, const int4& right) noexcept
{
	return left.x * right.x
	     + left.y * right.y
	     + left.z * right.z
	     + left.w * right.w;
}

// Vector length()
// ------------------------------------------------------------------------------------------------

inline __device__ __host__ float length(const float2& vector) noexcept
{
	return sqrt(dot(vector, vector));
}

inline __device__ __host__ float length(const float3& vector) noexcept
{
	return sqrt(dot(vector, vector));
}

inline __device__ __host__ float length(const float4& vector) noexcept
{
	return sqrt(dot(vector, vector));
}

// Vector normalize()
// ------------------------------------------------------------------------------------------------

inline __device__ __host__ float2 normalize(const float2& vector) noexcept
{
	float lengthTmp = length(vector);
	return vector / lengthTmp;
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

// Vector cross()
// ------------------------------------------------------------------------------------------------

inline __device__ __host__ float3 cross(const float3& lhs, const float3& rhs) noexcept
{
	return make_float3(lhs.y * rhs.z - lhs.z * rhs.y,
	                   lhs.z * rhs.x - lhs.x * rhs.z,
	                   lhs.x * rhs.y - lhs.y * rhs.x);
}

inline __device__ __host__ int3 cross(const int3& lhs, const int3& rhs) noexcept
{
	return make_int3(lhs.y * rhs.z - lhs.z * rhs.y,
	                 lhs.z * rhs.x - lhs.x * rhs.z,
	                 lhs.x * rhs.y - lhs.y * rhs.x);
}

// Vector abs()
// ------------------------------------------------------------------------------------------------

inline __device__ __host__ float2 abs(const float2& v) noexcept
{
	return make_float2(fabs(v.x), fabs(v.y));
}

inline __device__ __host__ float3 abs(const float3& v) noexcept
{
	return make_float3(fabs(v.x), fabs(v.y), fabs(v.z));
}

inline __device__ __host__ float4 abs(const float4& v) noexcept
{
	return make_float4(fabs(v.x), fabs(v.y), fabs(v.z), fabs(v.w));
}

// Vector max()
// ------------------------------------------------------------------------------------------------

inline __device__ __host__ float2 max(const float2& lhs, const float2& rhs) noexcept
{
	return make_float2(fmax(lhs.x, rhs.x),
	                   fmax(lhs.y, rhs.y));
}

inline __device__ __host__ float2 max(const float2& lhs, float scalar) noexcept
{
	return make_float2(fmax(lhs.x, scalar),
	                   fmax(lhs.y, scalar));
}

inline __device__ __host__ float2 max(float scalar, const float2& rhs) noexcept
{
	return max(rhs, scalar);
}

inline __device__ __host__ float3 max(const float3& lhs, const float3& rhs) noexcept
{
	return make_float3(fmax(lhs.x, rhs.x),
	                   fmax(lhs.y, rhs.y),
	                   fmax(lhs.z, rhs.z));
}

inline __device__ __host__ float3 max(const float3& lhs, float scalar) noexcept
{
	return make_float3(fmax(lhs.x, scalar),
	                   fmax(lhs.y, scalar),
	                   fmax(lhs.z, scalar));
}

inline __device__ __host__ float3 max(float scalar, const float3& rhs) noexcept
{
	return max(rhs, scalar);
}

inline __device__ __host__ float4 max(const float4& lhs, const float4& rhs) noexcept
{
	return make_float4(fmax(lhs.x, rhs.x),
	                   fmax(lhs.y, rhs.y),
	                   fmax(lhs.z, rhs.z),
	                   fmax(lhs.w, rhs.w));
}

inline __device__ __host__ float4 max(const float4& lhs, float scalar) noexcept
{
	return make_float4(fmax(lhs.x, scalar),
	                   fmax(lhs.y, scalar),
	                   fmax(lhs.z, scalar),
	                   fmax(lhs.w, scalar));
}

inline __device__ __host__ float4 max(float scalar, const float4& rhs) noexcept
{
	return max(rhs, scalar);
}

// Vector min()
// ------------------------------------------------------------------------------------------------

inline __device__ __host__ float2 min(const float2& lhs, const float2& rhs) noexcept
{
	return make_float2(fmin(lhs.x, rhs.x),
	                   fmin(lhs.y, rhs.y));
}

inline __device__ __host__ float2 min(const float2& lhs, float scalar) noexcept
{
	return make_float2(fmin(lhs.x, scalar),
	                   fmin(lhs.y, scalar));
}

inline __device__ __host__ float2 min(float scalar, const float2& rhs) noexcept
{
	return min(rhs, scalar);
}

inline __device__ __host__ float3 min(const float3& lhs, const float3& rhs) noexcept
{
	return make_float3(fmin(lhs.x, rhs.x),
	                   fmin(lhs.y, rhs.y),
	                   fmin(lhs.z, rhs.z));
}

inline __device__ __host__ float3 min(const float3& lhs, float scalar) noexcept
{
	return make_float3(fmin(lhs.x, scalar),
	                   fmin(lhs.y, scalar),
	                   fmin(lhs.z, scalar));
}

inline __device__ __host__ float3 min(float scalar, const float3& rhs) noexcept
{
	return min(rhs, scalar);
}

inline __device__ __host__ float4 min(const float4& lhs, const float4& rhs) noexcept
{
	return make_float4(fmin(lhs.x, rhs.x),
	                   fmin(lhs.y, rhs.y),
	                   fmin(lhs.z, rhs.z),
	                   fmin(lhs.w, rhs.w));
}

inline __device__ __host__ float4 min(const float4& lhs, float scalar) noexcept
{
	return make_float4(fmin(lhs.x, scalar),
	                   fmin(lhs.y, scalar),
	                   fmin(lhs.z, scalar),
	                   fmin(lhs.w, scalar));
}

inline __device__ __host__ float4 min(float scalar, const float4& rhs) noexcept
{
	return min(rhs, scalar);
}

