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

using sfz::vec3;
using sfz::vec4;

// Light input
// ------------------------------------------------------------------------------------------------

struct IncomingLight final {
	vec4 fData1;
	vec4 fData2;

	SFZ_CUDA_CALL vec3 origin() const noexcept { return fData1.xyz; }
	SFZ_CUDA_CALL vec3 amount() const noexcept { return fData2.xyz; }
	SFZ_CUDA_CALL float fallofFactor() const noexcept { return fData1.w; }

	SFZ_CUDA_CALL void setOrigin(const vec3& val) noexcept { fData1.xyz = val; }
	SFZ_CUDA_CALL void setAmount(const vec3& val) noexcept { fData2.xyz = val; }
	SFZ_CUDA_CALL void setFallofFactor(float val) noexcept { fData1.w = val; }
};

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
	const IncomingLight* __restrict__ incomingLights;
	uint32_t numIncomingLights;

	// Light sources
	const SphereLight* staticSphereLights;
	const bool* __restrict__ inLights;
	uint32_t numStaticSphereLights;
};

void launchGatherRaysShadeKernel(const GatherRaysShadeKernelInput& input,
                                 cudaSurfaceObject_t resultOut) noexcept;

} // namespace phe
