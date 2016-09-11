// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include <cuda_runtime.h>

#include <sfz/math/Vector.hpp>

// TODO: REMOVE
#include <phantasy_engine/renderers/cpu_ray_tracer/BVH.hpp>

#include <phantasy_engine/RayTracerCommon.hpp>

namespace phe {

using sfz::vec2i;
using sfz::vec3;

void runCudaRayTracer(cudaSurfaceObject_t surface, vec2i surfaceRes, const CameraDef& cam, BVHNode* bvhNodes, TriangleVertices* triangles) noexcept;

} // namespace phe
