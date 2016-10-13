// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <sfz/math/Vector.hpp>

namespace phe {

using sfz::vec2i;

// InitCurand launch function
// ------------------------------------------------------------------------------------------------

void launchInitCurandKernel(vec2i res, curandState* curandStates) noexcept;

} // namespace phe
