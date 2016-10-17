// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <sfz/math/Vector.hpp>

#include "phantasy_engine/level/SphereLight.hpp"

#include "kernels/RayCastKernel.hpp"

namespace phe {

using sfz::vec2u;
using sfz::vec3;

// GenSecondaryRaysKernel
// ------------------------------------------------------------------------------------------------

struct GenSecondaryRaysKernelInput final {
	// Camera info
	vec3 camPos;

	// GBuffer
	vec2u res;
	cudaSurfaceObject_t posTex;
	cudaSurfaceObject_t normalTex;
	cudaSurfaceObject_t albedoTex;
	cudaSurfaceObject_t materialTex;

	// Light sources
	const SphereLight* __restrict__ staticSphereLights;
	uint32_t numStaticSphereLights;
};

void launchGenSecondaryRaysKernel(const GenSecondaryRaysKernelInput& input,
                                  RayIn* __restrict__ secondaryRays,
                                  RayIn* __restrict__ shadowRays) noexcept;

} // namespace phe
