#include "renderers/cuda_ray_tracer/CUDATest.cuh"

__global__ void writeBlauCuda(cudaSurfaceObject_t surf, int width, int height)
{
	int x = threadIdx.x;
	int y = threadIdx.y;
	if (x < width && y < height) {
		float4 blau = {0.0f, 0.25f, 1.0f, 1.0f};
		surf2Dwrite(blau, surf, x, y);
	}	
}

namespace sfz {

void writeBlau(cudaSurfaceObject_t surf, vec2i surfRes, vec2i currRes) noexcept
{
	dim3 numThreads = dim3((uint32_t)currRes.x, (uint32_t)currRes.y, 1U);
	int numBlocks = 128;
	writeBlauCuda<<<numBlocks, numThreads>>>(surf, currRes.x, currRes.y);
	cudaDeviceSynchronize();
}

}
