// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include <sfz/math/Matrix.hpp>
#include <sfz/math/Vector.hpp>

namespace phe {

using sfz::mat4;
using sfz::vec3;

// DynObject
// ------------------------------------------------------------------------------------------------

struct DynObject final {
	uint32_t meshIndex = ~0u;
	mat4 transform = sfz::identityMatrix4<float>();
	vec3 velocity = vec3(0.0f);
};

} // namespace phe
