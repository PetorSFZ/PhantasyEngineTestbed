#include "renderers/cuda_ray_tracer/CUDATest.cuh"

#include <stdio.h>

__device__ const char *STR = "HELLO WORLD!";
const char STR_LENGTH = 12;

__global__ void hello()
{
	printf("%c\n", STR[threadIdx.x % STR_LENGTH]);
}

void doSomething() noexcept
{
	int num_threads = STR_LENGTH;
	int num_blocks = 1;
	hello<<<num_blocks, num_threads>>>();
	cudaDeviceSynchronize();
}
