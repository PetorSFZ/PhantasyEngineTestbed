#pragma once

#include <sfz/math/Vector.hpp>

namespace sfz {

struct PointLight {
	vec3 pos{ 0, 0, 0 };
	vec3 strength{ 0, 0, 0 };
	float radius{ 0 };
	bool shadows{ false };
};

}
