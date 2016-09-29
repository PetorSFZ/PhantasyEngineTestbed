// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#include "kernels/ProcessGBufferKernel.hpp"

#include "CudaHelpers.hpp"
#include "CudaSfzVectorCompatibility.cuh"

namespace phe {

using sfz::vec4;

static __global__ void tempWriteColorKernel(cudaSurfaceObject_t surface, vec2i res)
{
	// Calculate surface coordinates
	vec2i loc = vec2i(blockIdx.x * blockDim.x + threadIdx.x,
	                  blockIdx.y * blockDim.y + threadIdx.y);
	if (loc.x >= res.x || loc.y >= res.y) return;

	vec4 color = vec4(1.0f, 0.0f, 0.0f, 1.0f);
	surf2Dwrite(toFloat4(color), surface, loc.x * sizeof(float4), loc.y);
}

void launchTempWriteColorKernel(cudaSurfaceObject_t surface, vec2i res) noexcept
{
	// Calculate number of threads and blocks to run
	dim3 threadsPerBlock(8, 8);
	dim3 numBlocks((res.x + threadsPerBlock.x - 1) / threadsPerBlock.x,
	               (res.y + threadsPerBlock.y - 1) / threadsPerBlock.y);

	// Run cuda ray tracer kernel
	tempWriteColorKernel<<<numBlocks, threadsPerBlock>>>(surface, res);
	CHECK_CUDA_ERROR(cudaGetLastError());
	CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

} // namespace phe
