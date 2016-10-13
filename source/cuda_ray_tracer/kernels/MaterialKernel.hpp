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
	vec3 pendingLightContribution;
	vec3 finalColor;
	vec3 throughput;
	uint32_t pathLength;
	uint32_t shadowRayStartIndex;
	uint32_t numShadowRays;
};

struct MaterialRequest final {
	uint32_t materialID;
};

struct MaterialKernelInput final {
	vec2i res;
	RayIn* shadowRays;
	PathState* pathStates;
	curandState* randStates;
	const RayIn* rays;
	const RayHit* rayHits;
	const TriangleData* staticTriangleDatas;
	const Material* materials;
	const cudaTextureObject_t* textures;
	const SphereLight* sphereLights;
	uint32_t numSphereLights;
};

struct GBufferMaterialKernelInput final {
	vec2i res;
	vec3 camPos;
	RayIn* extensionRays;
	RayIn* shadowRays;
	PathState* pathStates;
	curandState* randState;
	cudaSurfaceObject_t posTex;
	cudaSurfaceObject_t normalTex;
	cudaSurfaceObject_t albedoTex;
	cudaSurfaceObject_t materialTex;
	const SphereLight* sphereLights;
	uint32_t numSphereLights;
};

struct WriteResultKernelInput final {
	cudaSurfaceObject_t surface;
	vec2i res;
	PathState* pathStates;
	const RayHit* rayHits;
};

// Material kernel launch function
// ------------------------------------------------------------------------------------------------

void launchMaterialKernel(const MaterialKernelInput& input) noexcept;

void launchGBufferMaterialKernel(const GBufferMaterialKernelInput& input) noexcept;

void launchInitPathStatesKernel(vec2i res, PathState* pathStates) noexcept;

void launchShadowLogicKernel(vec2i res, const RayHit* shadowRayHits, PathState* pathStates) noexcept;

void launchWriteResultKernel(const WriteResultKernelInput& input) noexcept;

} // namespace phe
