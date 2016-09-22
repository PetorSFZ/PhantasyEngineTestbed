// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include <sfz/math/Vector.hpp>

namespace phe {

using sfz::vec3;

struct SphereLight {
	vec3 pos{ 0.0f, 0.0f, 0.0f };
	vec3 strength{ 0.0f, 0.0f, 0.0f };
	float range{ 0.0f };
	float radius{ 0.0f }; // A radius of 0 makes it a point light
	bool shadows{ false };
};

} // namespace phe
