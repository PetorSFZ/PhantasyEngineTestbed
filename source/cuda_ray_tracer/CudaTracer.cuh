// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include <cuda_runtime.h>

#include <sfz/math/Vector.hpp>

#include <phantasy_engine/RayTracerCommon.hpp>

namespace phe {

using sfz::vec2i;
using sfz::vec3;

struct StaticSceneCuda final {
	BVHNode* bvhNodes = nullptr;
	TriangleVertices* triangleVertices = nullptr;
	TriangleData* triangleDatas = nullptr;

	PointLight* pointLights = nullptr;
	uint32_t numPointLights = ~0u;

	cudaTextureObject_t* textures = nullptr;
};

void runCudaRayTracer(cudaSurfaceObject_t surface, vec2i surfaceRes, const CameraDef& cam,
                      const StaticSceneCuda& staticScene) noexcept;

} // namespace phe
