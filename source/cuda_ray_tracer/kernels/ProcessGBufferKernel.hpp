// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <sfz/math/Vector.hpp>

#include "kernels/RayCastKernel.hpp"

namespace phe {

using sfz::vec2i;
using sfz::vec3;

struct CreateReflectRaysInput final {
	// Camera info
	vec3 camPos;
	
	// GBuffer
	vec2i res;
	cudaSurfaceObject_t posTex;
	cudaSurfaceObject_t normalTex;
	cudaSurfaceObject_t materialIdTex;
};

void launchCreateReflectRaysKernel(const CreateReflectRaysInput& input, RayIn* raysOut) noexcept;

} // namespace phe
