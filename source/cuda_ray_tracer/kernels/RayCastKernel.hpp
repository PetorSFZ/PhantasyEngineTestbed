// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <sfz/math/Vector.hpp>

#include <phantasy_engine/ray_tracer_common/GenerateRays.hpp>

namespace phe {

using sfz::vec2i;
using sfz::vec3;
using sfz::vec4;

// RayCastKernel i/o structs
// ------------------------------------------------------------------------------------------------

struct RayIn final {
	vec4 data1;
	vec4 data2;

	SFZ_CUDA_CALLABLE vec3 origin() const noexcept { return data1.xyz; }
	SFZ_CUDA_CALLABLE vec3 dir() const noexcept { return data2.xyz; }
	SFZ_CUDA_CALLABLE float minDist() const noexcept { return data1.w; }
	SFZ_CUDA_CALLABLE float maxDist() const noexcept { return data2.w; }

	SFZ_CUDA_CALLABLE void setOrigin(const vec3& origin) noexcept { data1.xyz = origin; }
	SFZ_CUDA_CALLABLE void setDir(const vec3& dir) noexcept { data2.xyz = dir; }
	SFZ_CUDA_CALLABLE void setMinDist(float dist) noexcept { data1.w = dist; }
	SFZ_CUDA_CALLABLE void setMaxDist(float dist) noexcept { data2.w = dist; }
};

static_assert(sizeof(RayIn) == 32, "RayIn is padded");

struct RayHit final {
	uint32_t triangleIndex; // Index of triangle hit, ~0 if no triangle collision occured
	float t; // Amount to go in ray direction from origin
	float u, v; // Position hit on triangle
};

static_assert(sizeof(RayHit) == 16, "RayHitOut is padded");

// RayCastKernel launch function
// ------------------------------------------------------------------------------------------------

struct RayCastKernelInput final {
	cudaTextureObject_t bvhNodes;
	cudaTextureObject_t triangleVerts;
	uint32_t numRays;
	const RayIn* rays;
};

void launchRayCastKernel(const RayCastKernelInput& input, RayHit* __restrict__ rayResults,
                         const cudaDeviceProp& deviceProperties) noexcept;

void launchRayCastNoPersistenceKernel(const RayCastKernelInput& input, RayHit* __restrict__ rayResults,
                                      const cudaDeviceProp& deviceProperties) noexcept;

void launchShadowRayCastKernel(const RayCastKernelInput& input, bool* __restrict__ inLight,
                               const cudaDeviceProp& deviceProperties) noexcept;

// Secondary helper kernels (for debugging and profiling)
// ------------------------------------------------------------------------------------------------

void launchGenPrimaryRaysKernel(RayIn* rays, const CameraDef& cam, vec2i res) noexcept;

void launchGenSecondaryRaysKernel(RayIn* rays, vec3 camPos, vec2i res, cudaSurfaceObject_t posTex,
                                  cudaSurfaceObject_t normalTex) noexcept;

void launchWriteRayHitsToScreenKernel(cudaSurfaceObject_t surface, vec2i res, const RayHit* rayHits) noexcept;

} // namespace phe
