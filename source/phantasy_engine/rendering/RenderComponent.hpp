// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include <sfz/math/Matrix.hpp>
#include <sfz/math/Vector.hpp>

namespace phe {

using sfz::mat4;
using sfz::vec3;

// RenderComponent struct
// ------------------------------------------------------------------------------------------------

/// A struct containing the information necessary to render an entity
struct RenderComponent final {
	mat4 transform;
	uint32_t meshIndex;
	vec3 velocity; // TODO: Should perhaps not be here?
};

static_assert(sizeof(RenderComponent) == ((4 * 4 + 4) * 4), "RenderComponent is padded");

} // namespace phe
