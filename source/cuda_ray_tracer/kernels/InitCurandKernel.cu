// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#include "kernels/InitCurandKernel.hpp"

#include <curand_kernel.h>
#include <device_launch_parameters.h>

#include <phantasy_engine/ray_tracer_common/BVHNode.hpp>

#include "CudaHelpers.hpp"

namespace phe {

using sfz::vec2;
using sfz::vec2i;
using sfz::vec3;
using sfz::vec3i;
using sfz::vec4;
using sfz::vec4i;

// InitCurand kernel
// ------------------------------------------------------------------------------------------------

static __global__ void initCurand(vec2i res, curandState* randStates)
{
	// Calculate surface coordinates
	vec2i loc = vec2i(blockIdx.x * blockDim.x + threadIdx.x,
	                  blockIdx.y * blockDim.y + threadIdx.y);
	if (loc.x >= res.x || loc.y >= res.y) return;

	uint32_t id = loc.x + loc.y * res.x;
	curand_init((id + 1) * 200, 0, 0, &randStates[id]);
}

void launchInitCurandKernel(vec2i res, curandState* randStates) noexcept
{
	// Calculate number of threads and blocks to run
	dim3 threadsPerBlock(8, 8);
	dim3 numBlocks((res.x + threadsPerBlock.x - 1) / threadsPerBlock.x,
	               (res.y + threadsPerBlock.y - 1) / threadsPerBlock.y);

	// Run cuda ray tracer kernel
	initCurand<<<numBlocks, threadsPerBlock>>>(res, randStates);
	CHECK_CUDA_ERROR(cudaGetLastError());
	CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}
} // namespace phe
