// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include <cuda_runtime.h>

#include <sfz/math/Vector.hpp>

namespace phe {

using sfz::vec2i;
using sfz::vec3;

struct CameraDef final {
	vec3 origin;
	vec3 dir; // normalized
	vec3 up; // normalized and orthogonal to camDir
	vec3 right; // normalized and orthogonal to both camDir and camUp
	float vertFovRad;
};

void runCudaRayTracer(cudaSurfaceObject_t surface, vec2i surfaceRes, const CameraDef& cam) noexcept;

} // namespace phe
