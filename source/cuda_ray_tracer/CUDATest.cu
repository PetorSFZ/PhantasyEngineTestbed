// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#include "CUDATest.cuh"

__global__ void writeBlauImpl(cudaSurfaceObject_t surf, int width, int height)
{
	// Calculate surface coordinates
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < width && y < height) {
		float4 data = make_float4(x / float(width), y / float(height), 1.0f, 1.0f);
		surf2Dwrite(data, surf, x * 4 * 4, y);
	}
}

namespace phe {

void writeBlau(cudaSurfaceObject_t surf, vec2i surfRes, vec2i currRes) noexcept
{
	int width = surfRes.x;
	int height = surfRes.y;
	
	dim3 threadsPerBlock(16, 16);
	dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
	               (height + threadsPerBlock.y  - 1) / threadsPerBlock.y);

	writeBlauImpl<<<numBlocks, threadsPerBlock>>>(surf, width, height);
}

}
