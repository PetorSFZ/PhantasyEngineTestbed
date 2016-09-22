// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <sfz/math/Vector.hpp>

#include <phantasy_engine/RayTracerCommon.hpp>
#include <phantasy_engine/rendering/Material.hpp>

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
	PointLight* staticPointLights = nullptr;
	uint32_t numStaticPointLights = ~0u;

	// TODO: Dynamic geometry & dynamic light sources
};


void initCurand(const CudaTracerParams& params);

void clearSurface(const cudaSurfaceObject_t& surface, const vec2i& targetRes, const vec4& color);

void runCudaRayTracer(const CudaTracerParams& params) noexcept;

void runCudaDebugRayTracer(const CudaTracerParams& params) noexcept;

} // namespace phe
