// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include <sfz/math/Vector.hpp>

namespace sfz {

struct PointLight {
	vec3 pos{ 0.0f, 0.0f, 0.0f };
	vec3 strength{ 0.0f, 0.0f, 0.0f };
	float range{ 0.0f };
	bool shadows{ false };
};

}
