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

#include <functional> // std::hash

#include "sfz/math/Vector.hpp"

namespace sfz {

struct Sphere final {

	// Members
	// --------------------------------------------------------------------------------------------

	vec3 position;
	float radius;

	// Constructors & destructors
	// --------------------------------------------------------------------------------------------

	Sphere() noexcept = default;
	Sphere(const Sphere&) noexcept = default;
	Sphere& operator= (const Sphere&) noexcept = default;
	
	inline Sphere(const vec3& position, float radius) noexcept;

	// Member functions
	// --------------------------------------------------------------------------------------------

	inline size_t hash() const noexcept;
	inline vec3 closestPoint(const vec3& point) const noexcept;
};

} // namespace sfz

// Specializations of standard library for sfz::Sphere
// ------------------------------------------------------------------------------------------------

namespace std {

template<>
struct hash<sfz::Sphere> {
	inline size_t operator() (const sfz::Sphere& sphere) const noexcept;
};

} // namespace std

#include "sfz/geometry/Sphere.inl"
