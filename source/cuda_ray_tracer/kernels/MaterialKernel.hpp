// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>

#include <sfz/math/Vector.hpp>

#include <phantasy_engine/level/SphereLight.hpp>
#include <phantasy_engine/ray_tracer_common/Triangle.hpp>
#include <phantasy_engine/rendering/Material.hpp>

#include "kernels/RayCastKernel.hpp"

namespace phe {

using sfz::vec2i;
using sfz::vec3;
using sfz::vec4;

// Material kernel I/O structs
// ------------------------------------------------------------------------------------------------

struct PathState final {
	vec3 finalColor;
	vec3 throughput;
	uint32_t pathLength;
};

struct GBufferMaterialKernelInput final {
	vec2i res;
	vec3 camPos;
	RayIn* extensionRays;
	RayIn* shadowRays;
	vec3* lightContributions;
	PathState* pathStates;
	curandState* randState;
	cudaSurfaceObject_t posTex;
	cudaSurfaceObject_t normalTex;
	cudaSurfaceObject_t albedoTex;
	cudaSurfaceObject_t materialTex;
	const SphereLight* staticSphereLights;
	uint32_t numStaticSphereLights;
};

struct MaterialKernelInput final {
	vec2i res;
	RayIn* shadowRays;
	vec3* lightContributions;
	PathState* pathStates;
	curandState* randStates;
	const RayIn* rays;
	const RayHit* rayHits;
	const TriangleData* staticTriangleDatas;
	const Material* materials;
	const cudaTextureObject_t* textures;
	const SphereLight* staticSphereLights;
	uint32_t numStaticSphereLights;
};

struct ShadowLogicKernelInput final {
	vec2i res;
	const bool* shadowRayHits;
	PathState* pathStates;
	const vec3* lightContributions;
	uint32_t numStaticSphereLights;
};

struct WriteResultKernelInput final {
	cudaSurfaceObject_t surface;
	vec2i res;
	PathState* pathStates;
	const RayHit* rayHits;
};

// Material kernel launch function
// ------------------------------------------------------------------------------------------------

void launchGBufferMaterialKernel(const GBufferMaterialKernelInput& input) noexcept;

void launchMaterialKernel(const MaterialKernelInput& input) noexcept;

void launchInitPathStatesKernel(vec2i res, PathState* pathStates) noexcept;

void launchShadowLogicKernel(const ShadowLogicKernelInput& input) noexcept;

void launchWriteResultKernel(const WriteResultKernelInput& input) noexcept;

} // namespace phe
