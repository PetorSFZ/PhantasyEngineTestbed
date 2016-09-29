// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <sfz/math/Vector.hpp>

namespace phe {

using sfz::vec2i;

void launchTempWriteColorKernel(cudaSurfaceObject_t surface, vec2i res, 
                                cudaSurfaceObject_t normalTex) noexcept;

} // namespace phe
