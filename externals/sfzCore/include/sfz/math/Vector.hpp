// Copyright (c) Peter Hillerström (skipifzero.com, peter@hstroem.se)
//
// This software is provided 'as-is', without any express or implied
// warranty. In no event will the authors be held liable for any damages
// arising from the use of this software.
//
// Permission is granted to anyone to use this software for any purpose,
// including commercial applications, and to alter it and redistribute it
// freely, subject to the following restrictions:
//
// 1. The origin of this software must not be misrepresented; you must not
//    claim that you wrote the original software. If you use this software
//    in a product, an acknowledgment in the product documentation would be
//    appreciated but is not required.
// 2. Altered source versions must be plainly marked as such, and must not be
//    misrepresented as being the original software.
// 3. This notice may not be removed or altered from any source distribution.

#pragma once

#include <algorithm> // std::min & std::max
#include <cstddef> // std::size_t
#include <cstdint> // std::int32_t
#include <cmath> // std::sqrt
#include <functional>

#include "sfz/Assert.hpp"
#include "sfz/containers/StackString.hpp"
#include "sfz/CUDACallable.hpp"
#include "sfz/math/MathConstants.hpp"

/// A mathematical vector POD class that imitates a built-in primitive.
///
/// Typedefs are provided for float vectors (vec2, vec3 and vec4) and (32-bit signed) integer
/// vectors (ivec2, ivec3, ivec4). Note that for integers some operations, such as as calculating
/// the length, may give unexpected results due to truncation or overflow.
///
/// 2, 3 and 4 dimensional vectors are specialized to have more constructors and ways of accessing
/// data. For example, you can construct a vec3 with 3 floats (vec3(x, y, z)), or with a vec2 and a 
/// float (vec3(vec2(x,y), z) or vec3(x, vec2(y, z))). To access the x value of a vec3 v you can
/// write v[0], v.elements[0] or v.x, you can also access two adjacent elements as a vector by
/// writing v.xy or v.yz.
///
/// Satisfies the conditions of std::is_pod, std::is_trivial and std::is_standard_layout if used
/// with standard primitives.

namespace sfz {

using std::size_t;
using std::int32_t;

// Vector struct declaration
// ------------------------------------------------------------------------------------------------

template<typename T, size_t N>
struct Vector final {

	T elements[N];

	Vector() noexcept = default;
	Vector(const Vector<T,N>&) noexcept = default;
	Vector<T,N>& operator= (const Vector<T,N>&) noexcept = default;
	~Vector() noexcept = default;

	SFZ_CUDA_CALLABLE explicit Vector(const T* arrayPtr) noexcept;

	template<typename T2>
	SFZ_CUDA_CALLABLE explicit Vector(const Vector<T2,N>& other) noexcept;

	SFZ_CUDA_CALLABLE T& operator[] (const size_t index) noexcept;
	SFZ_CUDA_CALLABLE T operator[] (const size_t index) const noexcept;
};

template<typename T>
struct Vector<T,2> final {
	union {
		T elements[2];
		struct { T x, y; };
	};	

	Vector() noexcept = default;
	Vector(const Vector<T,2>&) noexcept = default;
	Vector<T,2>& operator= (const Vector<T,2>&) noexcept = default;
	~Vector() noexcept = default;

	SFZ_CUDA_CALLABLE explicit Vector(const T* arrayPtr) noexcept;
	SFZ_CUDA_CALLABLE explicit Vector(T value) noexcept;
	SFZ_CUDA_CALLABLE Vector(T x, T y) noexcept;

	template<typename T2>
	SFZ_CUDA_CALLABLE explicit Vector(const Vector<T2,2>& other) noexcept;

	SFZ_CUDA_CALLABLE T& operator[] (const size_t index) noexcept;
	SFZ_CUDA_CALLABLE T operator[] (const size_t index) const noexcept;
};

template<typename T>
struct Vector<T,3> final {
	union {
		T elements[3];
		struct { T x, y, z; };
		struct { Vector<T,2> xy; };
		struct { T xAlias; Vector<T,2> yz; };
	};

	Vector() noexcept = default;
	Vector(const Vector<T,3>&) noexcept = default;
	Vector<T,3>& operator= (const Vector<T,3>&) noexcept = default;
	~Vector() noexcept = default;

	SFZ_CUDA_CALLABLE explicit Vector(const T* arrayPtr) noexcept;
	SFZ_CUDA_CALLABLE explicit Vector(T value) noexcept;
	SFZ_CUDA_CALLABLE Vector(T x, T y, T z) noexcept;
	SFZ_CUDA_CALLABLE Vector(Vector<T,2> xy, T z) noexcept;
	SFZ_CUDA_CALLABLE Vector(T x, Vector<T,2> yz) noexcept;

	template<typename T2>
	SFZ_CUDA_CALLABLE explicit Vector(const Vector<T2,3>& other) noexcept;

	SFZ_CUDA_CALLABLE T& operator[] (const size_t index) noexcept;
	SFZ_CUDA_CALLABLE T operator[] (const size_t index) const noexcept;
};

template<typename T>
struct Vector<T,4> final {
	union {
		T elements[4];
		struct { T x, y, z, w; };
		struct { Vector<T,3> xyz; };
		struct { T xAlias1; Vector<T,3> yzw; };
		struct { Vector<T,2> xy, zw; };
		struct { T xAlias2; Vector<T,2> yz; };
	};

	Vector() noexcept = default;
	Vector(const Vector<T,4>&) noexcept = default;
	Vector<T,4>& operator= (const Vector<T,4>&) noexcept = default;
	~Vector() noexcept = default;

	SFZ_CUDA_CALLABLE explicit Vector(const T* arrayPtr) noexcept;
	SFZ_CUDA_CALLABLE explicit Vector(T value) noexcept;
	SFZ_CUDA_CALLABLE Vector(T x, T y, T z, T w) noexcept;
	SFZ_CUDA_CALLABLE Vector(Vector<T,3> xyz, T w) noexcept;
	SFZ_CUDA_CALLABLE Vector(T x, Vector<T,3> yzw) noexcept;
	SFZ_CUDA_CALLABLE Vector(Vector<T,2> xy, Vector<T,2> zw) noexcept;
	SFZ_CUDA_CALLABLE Vector(Vector<T,2> xy, T z, T w) noexcept;
	SFZ_CUDA_CALLABLE Vector(T x, Vector<T,2> yz, T w) noexcept;
	SFZ_CUDA_CALLABLE Vector(T x, T y, Vector<T,2> zw) noexcept;

	template<typename T2>
	SFZ_CUDA_CALLABLE  Vector(const Vector<T2,4>& other) noexcept;

	SFZ_CUDA_CALLABLE T& operator[] (const size_t index) noexcept;
	SFZ_CUDA_CALLABLE T operator[] (const size_t index) const noexcept;
};

using vec2 = Vector<float,2>;
using vec3 = Vector<float,3>;
using vec4 = Vector<float,4>;

using vec2i = Vector<int32_t,2>;
using vec3i = Vector<int32_t,3>;
using vec4i = Vector<int32_t,4>;

// Vector constants
// ------------------------------------------------------------------------------------------------

template<typename T = float>
SFZ_CUDA_CALLABLE Vector<T,3> UNIT_X() noexcept;

template<typename T = float>
SFZ_CUDA_CALLABLE Vector<T,3> UNIT_Y() noexcept;

template<typename T = float>
SFZ_CUDA_CALLABLE Vector<T,3> UNIT_Z() noexcept;

// Vector functions
// ------------------------------------------------------------------------------------------------

/// Calculates length of the vector
template<typename T, size_t N>
SFZ_CUDA_CALLABLE T length(const Vector<T,N>& vector) noexcept;

/// Calculates squared length of vector
template<typename T, size_t N>
SFZ_CUDA_CALLABLE T squaredLength(const Vector<T,N>& vector) noexcept;

/// Normalizes vector
/// sfz_assert_debug: length of vector is not zero
template<typename T, size_t N>
SFZ_CUDA_CALLABLE Vector<T,N> normalize(const Vector<T,N>& vector) noexcept;

/// Normalizes vector, returns zero if vector is zero
template<typename T, size_t N>
SFZ_CUDA_CALLABLE Vector<T,N> safeNormalize(const Vector<T,N>& vector) noexcept;

/// Calculates the dot product of two vectors
template<typename T, size_t N>
SFZ_CUDA_CALLABLE T dot(const Vector<T,N>& left, const Vector<T,N>& right) noexcept;

/// Calculates the cross product of two vectors
template<typename T>
SFZ_CUDA_CALLABLE Vector<T,3> cross(const Vector<T,3>& left, const Vector<T,3>& right) noexcept;

/// Calculates the sum of all the elements in the vector
template<typename T, size_t N>
SFZ_CUDA_CALLABLE T sum(const Vector<T,N>& vector) noexcept;

/// Calculates the positive angle (in radians) between two vectors
/// Range: [0, Pi)
/// sfz_assert_debug: norm of both vectors != 0
template<typename T, size_t N>
SFZ_CUDA_CALLABLE T angle(const Vector<T,N>& left, const Vector<T,N>& right) noexcept;

/// Calculates the positive angle (in radians) between the vector and the x-axis
/// Range: [0, 2*Pi).
/// sfz_assert_debug: norm of vector != 0
template<typename T>
SFZ_CUDA_CALLABLE T angle(Vector<T,2> vector) noexcept;

/// Rotates a 2-dimensional vector with the specified angle (in radians) around origo
template<typename T>
SFZ_CUDA_CALLABLE Vector<T,2> rotate(Vector<T,2> vector, T angleRadians) noexcept;

/// Returns the element-wise minimum of two vectors.
template<typename T, size_t N>
SFZ_CUDA_CALLABLE Vector<T,N> min(const Vector<T,N>& left, const Vector<T,N>& right) noexcept;

/// Returns the element-wise maximum of two vectors.
template<typename T, size_t N>
SFZ_CUDA_CALLABLE Vector<T,N> max(const Vector<T,N>& left, const Vector<T,N>& right) noexcept;

/// Returns the element-wise minimum of a vector and a scalar.
template<typename T, size_t N>
SFZ_CUDA_CALLABLE Vector<T,N> min(const Vector<T,N>& vector, T scalar) noexcept;
template<typename T, size_t N>
SFZ_CUDA_CALLABLE Vector<T,N> min(T scalar, const Vector<T,N>& vector) noexcept;

/// Returns the element-wise maximum of a vector and a scalar.
template<typename T, size_t N>
SFZ_CUDA_CALLABLE Vector<T,N> max(const Vector<T,N>& vector, T scalar) noexcept;
template<typename T, size_t N>
SFZ_CUDA_CALLABLE Vector<T,N> max(T scalar, const Vector<T,N>& vector) noexcept;

/// Returns the smallest element in a vector (as defined by the min function)
template<typename T, size_t N>
SFZ_CUDA_CALLABLE T minElement(const Vector<T,N>& vector) noexcept;

/// Returns the largest element in a vector (as defined by the max function)
template<typename T, size_t N>
SFZ_CUDA_CALLABLE T maxElement(const Vector<T,N>& vector) noexcept;

/// Returns the element-wise abs() of the vector.
template<typename T, size_t N>
SFZ_CUDA_CALLABLE Vector<T,N> abs(const Vector<T,N>& vector) noexcept;

/// Hashes the vector
template<typename T, size_t N>
size_t hash(const Vector<T,N>& vector) noexcept;

/// Creates string representation of a float vector
template<size_t N>
StackString toString(const Vector<float,N>& vector, uint32_t numDecimals = 2) noexcept;

/// Creates string representation of a float vector
template<size_t N>
void toString(const Vector<float,N>& vector, StackString& string, uint32_t numDecimals = 2) noexcept;

/// Creates string representation of an int vector
template<size_t N>
StackString toString(const Vector<int32_t,N>& vector) noexcept;

/// Creates string representation of an int vector
template<size_t N>
void toString(const Vector<int32_t,N>& vector, StackString& string) noexcept;

// Operators (arithmetic & assignment)
// ------------------------------------------------------------------------------------------------

template<typename T, size_t N>
SFZ_CUDA_CALLABLE Vector<T,N>& operator+= (Vector<T,N>& left, const Vector<T,N>& right) noexcept;

template<typename T, size_t N>
SFZ_CUDA_CALLABLE Vector<T,N>& operator-= (Vector<T,N>& left, const Vector<T,N>& right) noexcept;

template<typename T, size_t N>
SFZ_CUDA_CALLABLE Vector<T,N>& operator*= (Vector<T,N>& left, T right) noexcept;

/// Element-wise multiplication assignment
template<typename T, size_t N>
SFZ_CUDA_CALLABLE Vector<T,N>& operator*= (Vector<T,N>& left, const Vector<T,N>& right) noexcept;

template<typename T, size_t N>
SFZ_CUDA_CALLABLE Vector<T,N>& operator/= (Vector<T,N>& left, T right) noexcept;

/// Element-wise division assignment
template<typename T, size_t N>
SFZ_CUDA_CALLABLE Vector<T,N>& operator/= (Vector<T,N>& left, const Vector<T,N>& right) noexcept;

// Operators (arithmetic)
// ------------------------------------------------------------------------------------------------

template<typename T, size_t N>
SFZ_CUDA_CALLABLE Vector<T,N> operator+ (const Vector<T,N>& left, const Vector<T,N>& right) noexcept;

template<typename T, size_t N>
SFZ_CUDA_CALLABLE Vector<T,N> operator- (const Vector<T,N>& left, const Vector<T,N>& right) noexcept;

template<typename T, size_t N>
SFZ_CUDA_CALLABLE Vector<T,N> operator- (const Vector<T,N>& vector) noexcept;

template<typename T, size_t N>
SFZ_CUDA_CALLABLE Vector<T,N> operator* (const Vector<T,N>& left, T right) noexcept;

/// Element-wise multiplication of two vectors
template<typename T, size_t N>
SFZ_CUDA_CALLABLE Vector<T,N> operator* (const Vector<T,N>& left, const Vector<T,N>& right) noexcept;

template<typename T, size_t N>
SFZ_CUDA_CALLABLE Vector<T,N> operator* (T left, const Vector<T,N>& right) noexcept;

template<typename T, size_t N>
SFZ_CUDA_CALLABLE Vector<T,N> operator/ (const Vector<T,N>& left, T right) noexcept;

/// Element-wise division of two vectors
template<typename T, size_t N>
SFZ_CUDA_CALLABLE Vector<T,N> operator/ (const Vector<T,N>& left, const Vector<T,N>& right) noexcept;

// Operators (comparison)
// ------------------------------------------------------------------------------------------------

template<typename T, size_t N>
SFZ_CUDA_CALLABLE bool operator== (const Vector<T,N>& left, const Vector<T,N>& right) noexcept;

template<typename T, size_t N>
SFZ_CUDA_CALLABLE bool operator!= (const Vector<T,N>& left, const Vector<T,N>& right) noexcept;

// Standard iterator functions
// ------------------------------------------------------------------------------------------------

template<typename T, size_t N>
T* begin(Vector<T,N>& vector) noexcept;

template<typename T, size_t N>
const T* begin(const Vector<T,N>& vector) noexcept;

template<typename T, size_t N>
const T* cbegin(const Vector<T,N>& vector) noexcept;

template<typename T, size_t N>
T* end(Vector<T,N>& vector) noexcept;

template<typename T, size_t N>
const T* end(const Vector<T,N>& vector) noexcept;

template<typename T, size_t N>
const T* cend(const Vector<T,N>& vector) noexcept;

} // namespace sfz

// Specializations of standard library for sfz::Vector
// ------------------------------------------------------------------------------------------------

namespace std {

template<typename T, size_t N>
struct hash<sfz::Vector<T,N>> {
	size_t operator() (const sfz::Vector<T,N>& vector) const noexcept;
};

} // namespace std

#include "sfz/math/Vector.inl"
