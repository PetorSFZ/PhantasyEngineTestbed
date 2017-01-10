// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include <sfz/math/Vector.hpp>
#include <sfz/math/MathSupport.hpp>

#include "phantasy_engine/ray_tracer_common/Triangle.hpp"
#include "phantasy_engine/rendering/Material.hpp"

#include "kernels/RayCastKernel.hpp"

namespace phe {

using sfz::vec3;
using sfz::vec4;

// RayHitInfo struct
// ------------------------------------------------------------------------------------------------

struct RayHitInfo final {

	// Raw storage, use getters/setters
	// --------------------------------------------------------------------------------------------

	// [pos.x, pos.y, pos.z, roughness]
	// [normal.x, normal.y, normal.z, metallic]
	// [albedo.r, albedo.g, albedo.b, alpha] // alpha is negative if no hit
	vec4 data1;
	vec4 data2;
	vec4 data3;

	// Getters
	// --------------------------------------------------------------------------------------------

	SFZ_CUDA_CALL bool wasHit() const noexcept { return data3.w >= 0.0f; }

	SFZ_CUDA_CALL vec3 position() const noexcept { return data1.xyz; }
	SFZ_CUDA_CALL vec3 normal() const noexcept { return data2.xyz; }
	SFZ_CUDA_CALL vec3 albedo() const noexcept { return data3.xyz; }
	SFZ_CUDA_CALL float alpha() const noexcept { return sfz::abs(data3.w); }
	SFZ_CUDA_CALL float roughness() const noexcept { return data1.w; }
	SFZ_CUDA_CALL float metallic() const noexcept { return data2.w; }

	// Setters
	// --------------------------------------------------------------------------------------------

	SFZ_CUDA_CALL void setHitStatus(bool status) { data3.w = status ? sfz::abs(data3.w) : -sfz::abs(data3.w); }

	SFZ_CUDA_CALL void setPosition(const vec3& val) noexcept { data1.xyz = val; }
	SFZ_CUDA_CALL void setNormal(const vec3& val) noexcept { data2.xyz = val; }
	SFZ_CUDA_CALL void setAlbedo(const vec3& val) noexcept { data3.xyz = val; }
	SFZ_CUDA_CALL void setAlpha(float val) noexcept { data3.w = (data3.w >= 0.0f) ? sfz::abs(val) : -sfz::abs(val); }
	SFZ_CUDA_CALL void setRoughness(float val) noexcept { data1.w = val; }
	SFZ_CUDA_CALL void setMetallic(float val) noexcept { data2.w = val; }
};

static_assert(sizeof(RayHitInfo) == 48, "RayHitInfo is padded");

// InterpretRayHitKernel
// ------------------------------------------------------------------------------------------------

struct InterpretRayHitKernelInput final {
	// Ray information
	const RayIn* __restrict__ rays;
	const RayHit* __restrict__ rayHits;
	uint32_t numRays;

	// Cuda materials & textures
	cudaTextureObject_t materialsTex;
	const cudaTextureObject_t* __restrict__ textures;

	// Static triangle data
	const TriangleData* __restrict__ staticTriangleDatas;
};

void launchInterpretRayHitKernel(const InterpretRayHitKernelInput& input,
                                 RayHitInfo* __restrict__ outInfos,
                                 const cudaDeviceProp& deviceProperties) noexcept;

} // namespace phe
