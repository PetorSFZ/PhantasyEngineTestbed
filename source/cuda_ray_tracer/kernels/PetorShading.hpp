// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <sfz/math/Vector.hpp>

#include "phantasy_engine/level/SphereLight.hpp"
#include "phantasy_engine/ray_tracer_common/Triangle.hpp"
#include "phantasy_engine/rendering/Material.hpp"

#include "kernels/InterpretRayHitKernel.hpp"

namespace phe {

// ProccessGBufferGenRaysKernel
// ------------------------------------------------------------------------------------------------

struct ProcessGBufferGenRaysInput final {
	// Camera info
	vec3 camPos;

	// GBuffer
	vec2i res;
	cudaSurfaceObject_t posTex;
	cudaSurfaceObject_t normalTex;
	cudaSurfaceObject_t albedoTex;
	cudaSurfaceObject_t materialTex;
};

void launchProcessGBufferGenRaysKernel(const ProcessGBufferGenRaysInput& input,
                                       RayIn* raysOut) noexcept;

// GatherRaysShadeKernel
// ------------------------------------------------------------------------------------------------

struct GatherRaysShadeKernelInput final {
	// Camera info
	vec3 camPos;

	// GBuffer
	vec2i res;
	cudaSurfaceObject_t posTex;
	cudaSurfaceObject_t normalTex;
	cudaSurfaceObject_t albedoTex;
	cudaSurfaceObject_t materialTex;

	// Ray cast results
	const RayHitInfo* __restrict__ rayHitInfos;

	// Light sources
	const SphereLight* staticSphereLights;
	uint32_t numStaticSphereLights;
};

void launchGatherRaysShadeKernel(const GatherRaysShadeKernelInput& input,
                                 cudaSurfaceObject_t resultOut) noexcept;

} // namespace phe