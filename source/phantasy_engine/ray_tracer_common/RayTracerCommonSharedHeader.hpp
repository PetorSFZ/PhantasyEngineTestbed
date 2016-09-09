// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

namespace phe {

// This is the shared header for all ray_tracer_common files. It's job is to include things from
// both CUDA and C++ and wrap them in a way that makes it possible to run the code on both C++
// and CUDA.
//
// Code in .cpp files are meant to be C++ only and not available for CUDA and other backends.

// C++ defines
// ------------------------------------------------------------------------------------------------

#if !defined(__CUDAC__)

#define PHE_CUDA_AVAILABLE inline

// Vectors
#include <sfz/math/Vector.hpp>
using vec2_t = sfz::vec2;
using vec3_t = sfz::vec3;
using vec4_t = sfz::vec4;
using vec2i_t = sfz::vec2i;
using vec3i_t = sfz::vec3i;
using vec4i_t = sfz::vec4i;

#endif

// CUDA defines
// ------------------------------------------------------------------------------------------------

#if defined(__CUDACC__)

#define PHE_CUDA_AVAILABLE inline __host__ __device__

// Vectors
using vec2_t = float2;
using vec3_t = float3;
using vec4_t = float4;
using vec2i_t = int2;
using vec3i_t = int3;
using vec4i_t = int4;

#define vec3_t F

#if defined(__CUDA_ARCH__) // Check if building for device or host
//#error "Building for device""
#endif

#else

#endif

} // namespace phe
