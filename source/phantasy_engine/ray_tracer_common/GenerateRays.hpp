// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include <sfz/math/Vector.hpp>

namespace phe {

using sfz::vec2;
using sfz::vec3;

// CameraDef
// ------------------------------------------------------------------------------------------------

/// The information needed to create the initial rays to cast
struct CameraDef final {
	vec3 origin;
	vec3 dir;

	// Used for calculating the ray
	// coord in [-1, 1], (0,0) in lower left corner
	// ray = normalize(dir + coord.x * dX + coord.y * dY)
	vec3 dX; // Camera's "right" vector scaled by some value
	vec3 dY; // Camera's up vector scaled by some value
};

CameraDef generateCameraDef(vec3 camPos, vec3 camDir, vec3 camUp, float vertFovRad, vec2 res) noexcept;

} // namespace phe
