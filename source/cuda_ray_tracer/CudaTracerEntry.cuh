// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include <cuda_runtime.h>

#include <sfz/math/Vector.hpp>

namespace phe {

using sfz::vec2i;

void runCudaRayTracer(cudaSurfaceObject_t surface, vec2i surfaceRes) noexcept;

} // namespace phe
