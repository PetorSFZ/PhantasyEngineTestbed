// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include <cuda_runtime.h>

#include <sfz/math/Vector.hpp>

namespace phe {

using sfz::vec2i;

void writeBlau(cudaSurfaceObject_t surf, vec2i surfRes, vec2i currRes) noexcept;

}
