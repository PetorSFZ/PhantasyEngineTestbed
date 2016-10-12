// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <sfz/math/Vector.hpp>

#include "phantasy_engine/level/SphereLight.hpp"
#include "phantasy_engine/ray_tracer_common/Triangle.hpp"
#include "phantasy_engine/rendering/Material.hpp"

#include "kernels/RayCastKernel.hpp"

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

	// Cuda materials & textures
	const Material* __restrict__ materials;
	const cudaTextureObject_t* __restrict__ textures;

	// Static triangle data
	const TriangleData* __restrict__ staticTriangleDatas;

	// Ray hits
	const RayIn* __restrict__ castRays;
	const RayHit* __restrict__ rayResults;

	// Light sources
	const SphereLight* staticSphereLights;
	uint32_t numStaticSphereLights;
};

void launchGatherRaysShadeKernel(const GatherRaysShadeKernelInput& input,
                                 cudaSurfaceObject_t resultOut) noexcept;

} // namespace phe
