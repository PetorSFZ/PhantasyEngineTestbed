// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include "kernels/InterpretRayHitKernel.hpp"
#include "kernels/PetorShading.hpp"

#include <phantasy_engine/level/SphereLight.hpp>

namespace phe {

// ShadeSecondaryHitKernel
// ------------------------------------------------------------------------------------------------

struct ShadeSecondaryHitKernelInput final {
	const RayIn* secondaryRays; // The rays used to generate the secondary hits, needed for view dir
	const RayHitInfo* rayHitInfos;
	uint32_t numRayHitInfos;

	// Light sources
	const SphereLight* staticSphereLights;
	uint32_t numStaticSphereLights;

	const bool* __restrict__ shadowRayResults;
};

void launchShadeSecondaryHitKernel(const ShadeSecondaryHitKernelInput& input,
                                   IncomingLight* __restrict__ incomingLightsOut) noexcept;


} // namespace phe
