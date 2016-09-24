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
	vec4 data[2];

	SFZ_CUDA_CALLABLE vec3 origin() const noexcept { return data[0].xyz; }
	SFZ_CUDA_CALLABLE vec3 dir() const noexcept { return data[1].xyz; }
	SFZ_CUDA_CALLABLE float maxDist() const noexcept { return data[0].w; }
	SFZ_CUDA_CALLABLE bool noResultOnlyHit() const noexcept { return (data[1].w < 0.0f); }

	SFZ_CUDA_CALLABLE vec3 setOrigin(const vec3& origin) noexcept { data[0].xyz = origin; }
	SFZ_CUDA_CALLABLE vec3 setDir(const vec3& dir) noexcept { data[1].xyz = dir; }
	SFZ_CUDA_CALLABLE float setMaxDist(float dist) noexcept { data[0].w = dist; }
	SFZ_CUDA_CALLABLE bool setNoResultOnlyHit(bool val) noexcept { data[1].w = val ? -1.0f : 1.0f; }
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

void genPrimaryRays(RayIn* rays, const CameraDef& cam, vec2i res) noexcept;

void writeRayHitsToScreen(cudaSurfaceObject_t surface, vec2i res, const RayHit* rayHits) noexcept;

void launchRayCastKernel(cudaTextureObject_t bvhNodes, cudaTextureObject_t triangleVerts,
                         const RayIn* rays, RayHit* rayHits, uint32_t numRays) noexcept;

} // namespace phe
