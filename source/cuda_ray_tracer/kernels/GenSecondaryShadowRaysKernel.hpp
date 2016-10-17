// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include "kernels/InterpretRayHitKernel.hpp"
#include "kernels/RayCastKernel.hpp"

#include <phantasy_engine/level/SphereLight.hpp>

namespace phe {

// GenSecondaryShadowRaysKernel
// ------------------------------------------------------------------------------------------------

struct GenSecondaryShadowRaysKernelInput final {
	const RayHitInfo* __restrict__ rayHitInfos;
	uint32_t numRayHitInfos;

	// Light sources
	const SphereLight* staticSphereLights;
	uint32_t numStaticSphereLights;
};

void launchGenSecondaryShadowRaysKernel(const GenSecondaryShadowRaysKernelInput& input,
                                        RayIn* __restrict__ raysOut) noexcept;

} // namespace phe
