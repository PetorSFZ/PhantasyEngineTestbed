// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>

#include <sfz/math/Vector.hpp>

#include <phantasy_engine/level/SphereLight.hpp>

#include "kernels/InterpretRayHitKernel.hpp"
#include "kernels/RayCastKernel.hpp"

namespace phe {

using sfz::vec2u;
using sfz::vec3;
using sfz::vec4;

// Material kernel I/O structs
// ------------------------------------------------------------------------------------------------

struct PathState final {
	vec3 throughput;
	uint32_t pathLength;
};

struct GBufferMaterialKernelInput final {
	vec2u res;
	vec3 camPos;
	RayIn* shadowRays;
	vec3* lightContributions;
	curandState* randStates;
	cudaSurfaceObject_t posTex;
	cudaSurfaceObject_t normalTex;
	cudaSurfaceObject_t albedoTex;
	cudaSurfaceObject_t materialTex;
	const SphereLight* staticSphereLights;
	uint32_t numStaticSphereLights;
};

struct MaterialKernelInput final {
	vec2u res;
	RayIn* shadowRays;
	vec3* lightContributions;
	PathState* pathStates;
	curandState* randStates;
	const RayIn* rays;
	const RayHitInfo* rayHitInfos;
	const SphereLight* staticSphereLights;
	uint32_t numStaticSphereLights;
};

struct CreateSecondaryRaysKernelInput final {
	vec2u res;
	vec3 camPos;
	RayIn* extensionRays;
	PathState* pathStates;
	curandState* randStates;
	cudaSurfaceObject_t posTex;
	cudaSurfaceObject_t normalTex;
	cudaSurfaceObject_t albedoTex;
	cudaSurfaceObject_t materialTex;
};

struct ShadowLogicKernelInput final {
	cudaSurfaceObject_t surface;
	vec2u res;
	uint32_t resolutionScale;
	bool addToSurface;
	const bool* shadowRayHits;
	PathState* pathStates;
	const vec3* lightContributions;
	uint32_t numStaticSphereLights;
};

struct WriteResultKernelInput final {
	cudaSurfaceObject_t surface;
	vec2u res;
	PathState* pathStates;
	const RayHit* rayHits;
};

// Material kernel launch function
// ------------------------------------------------------------------------------------------------

void launchGBufferMaterialKernel(const GBufferMaterialKernelInput& input) noexcept;

void launchMaterialKernel(const MaterialKernelInput& input) noexcept;

void launchCreateSecondaryRaysKernel(const CreateSecondaryRaysKernelInput& input) noexcept;

void launchInitPathStatesKernel(vec2u res, PathState* pathStates) noexcept;

void launchShadowLogicKernel(const ShadowLogicKernelInput& input) noexcept;

} // namespace phe
