// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <sfz/math/Vector.hpp>

#include <phantasy_engine/level/SphereLight.hpp>
#include <phantasy_engine/RayTracerCommon.hpp>
#include <phantasy_engine/rendering/Material.hpp>

#include "cuda_ray_tracer/CudaTracerRenderer.hpp"

namespace phe {

using sfz::vec2i;
using sfz::vec3;

struct CudaTracerParams final {
	// Target surface
	cudaSurfaceObject_t targetSurface = 0;
	vec2i targetRes;

	// Camera definition (for generating rays)
	CameraDef cam;

	// RNG states
	curandState* curandStates = nullptr;
	uint32_t numCurandStates = ~0u;

	// Materials & textures
	Material* materials = nullptr;
	uint32_t numMaterials = ~0u;
	cudaTextureObject_t* textures = nullptr;
	uint32_t numTextures = ~0u;

	// Static Geometry
	BVHNode* staticBvhNodes = nullptr;
	cudaTextureObject_t staticBvhNodesTex = 0;
	TriangleVertices* staticTriangleVertices = nullptr;
	cudaTextureObject_t staticTriangleVerticesTex = 0;
	TriangleData* staticTriangleDatas = nullptr;

	// Static light sources
	SphereLight* staticSphereLights = nullptr;
	uint32_t numStaticSphereLights = ~0u;
/*
	// Dynamic Geometry
	OuterBVHNode* dynamicOuterBvhNodes = nullptr;
	BVHNode** dynamicBvhNodes = nullptr;
	cudaTextureObject_t* dynamicBvhNodesTex = nullptr;
	TriangleVertices** dynamicTriangleVertices = nullptr;
	cudaTextureObject_t* dynamicTriangleVerticesTex = nullptr;
	TriangleData** dynamicTriangleDatas = nullptr;
	uint32_t numDynBvhs = 0;*/
};

void cudaRayTrace(const CudaTracerParams& params) noexcept;

void cudaCastRayTest(const CudaTracerParams& params) noexcept;

void cudaHeatmapTrace(const CudaTracerParams& params) noexcept;

void initCurand(const CudaTracerParams& params, unsigned long long seed) noexcept;

void cudaClearSurface(const cudaSurfaceObject_t& surface, const vec2i& targetRes, const vec4& color) noexcept;

} // namespace phe
