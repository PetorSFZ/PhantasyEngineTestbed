// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include <sfz/math/Matrix.hpp>
#include <sfz/math/Vector.hpp>

#include "phantasy_engine/level/Level.hpp"

namespace phe {

using sfz::mat4;

// Sponza loading functions
// ------------------------------------------------------------------------------------------------

void loadStaticSceneSponza(const char* basePath, const char* fileName, Level& level,
                           const mat4& modelMatrix = sfz::identityMatrix4<float>()) noexcept;

} // namespace phe
