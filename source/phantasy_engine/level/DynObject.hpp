// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include <sfz/math/Matrix.hpp>

namespace phe {

using sfz::mat4;

// DynObject
// ------------------------------------------------------------------------------------------------

struct DynObject final {
	uint32_t meshIndex = ~0u;
	mat4 transform = sfz::identityMatrix4<float>();
};

} // namespace phe
