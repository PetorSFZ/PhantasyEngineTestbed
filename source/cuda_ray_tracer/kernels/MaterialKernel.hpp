// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include "cuda.h"
#include "cuda_runtime.h"

#include <sfz/math/Vector.hpp>

namespace phe {

using sfz::vec2i;
using sfz::vec3;
using sfz::vec4;

// Material kernel I/O structs
// ------------------------------------------------------------------------------------------------

struct PathState final {
	vec3 pendingLightContribution;
	vec3 finalColor;
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
	RayIn* shadowRays;
	PathState* pathStates;
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
	RayIn* shadowRays;
	PathState* pathStates;
	const RayHit* rayHits;
};

// Material kernel launch function
// ------------------------------------------------------------------------------------------------

void launchMaterialKernel(const MaterialKernelInput& input) noexcept;

void launchGBufferMaterialKernel(const GBufferMaterialKernelInput& input) noexcept;

void launchInitPathStatesKernel(vec2i res, PathState* pathStates) noexcept;

void launchWriteResultKernel(const WriteResultKernelInput& input) noexcept;

} // namespace phe
