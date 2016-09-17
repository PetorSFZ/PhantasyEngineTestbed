// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#include "CudaTracer.cuh"

#include <math.h>

#include <sfz/math/Vector.hpp>

#include "CudaHelpers.hpp"
#include "CudaSfzVectorCompatibility.cuh"

namespace phe {

using namespace sfz;

inline __device__ vec4 readTexture(cudaTextureObject_t texture, vec2 coord) noexcept
{
	uchar4 res = tex2D<uchar4>(texture, coord.x, coord.y);
	return vec4(float(res.x), float(res.y), float(res.z), float(res.w)) / 256.0f;
}

inline __device__ void writeSurface(const cudaSurfaceObject_t& surface, vec2i loc, const vec4& data) noexcept
{
	float4 dataFloat4 = toFloat4(data);
	surf2Dwrite(dataFloat4, surface, loc.x * sizeof(float4), loc.y);
}

inline __device__ vec3 calculateRayDir(const CameraDef& cam, vec2 loc, vec2 surfaceRes) noexcept
{
	vec2 locNormalized = loc / surfaceRes; // [0, 1]
	vec2 centerOffsCoord = locNormalized * 2.0f - vec2(1.0f); // [-1.0, 1.0]
	centerOffsCoord.y = -centerOffsCoord.y;
	vec3 nonNormRayDir = cam.dir + cam.dX * centerOffsCoord.x + cam.dY * centerOffsCoord.y;
	return normalize(nonNormRayDir);
}

__global__ void cudaRayTracerKernel(cudaSurfaceObject_t surface, vec2i surfaceRes, CameraDef cam,
                                    StaticSceneCuda staticScene)
{
	// Calculate surface coordinates
	vec2i loc = vec2i(blockIdx.x * blockDim.x + threadIdx.x,
	                  blockIdx.y * blockDim.y + threadIdx.y);
	if (loc.x >= surfaceRes.x || loc.y >= surfaceRes.y) return;

	// Calculate ray direction
	vec3 rayDir = calculateRayDir(cam, vec2(loc), vec2(surfaceRes));
	Ray ray(cam.origin, rayDir);

	// Ray cast against BVH
	RayCastResult hit = castRay(staticScene.bvhNodes, staticScene.triangleVertices, ray);
	if (hit.triangleIndex == ~0u) {
		writeSurface(surface, loc, vec4(0.0f));
		return;
	}

	HitInfo info = interpretHit(staticScene.triangleDatas, hit, ray);
	vec3 color = info.normal;


	/*cudaTextureObject_t texture = staticScene.textures[10];

	vec3 lightPos = staticScene.pointLights[0].pos;
	vec3 l = -normalize(info.pos - lightPos);
	float diffuseFactor = max(dot(l, info.normal), 0.0f);

	vec4 texRes = readTexture(texture, info.uv);
	vec3 color = texRes.xyz * diffuseFactor;*/
	writeSurface(surface, loc, vec4(color, 1.0));
}

void runCudaRayTracer(cudaSurfaceObject_t surface, vec2i surfaceRes, const CameraDef& cam,
                      const StaticSceneCuda& staticScene) noexcept
{	
	// Calculate number of threads and blocks to run
	dim3 threadsPerBlock(8, 8);
	dim3 numBlocks((surfaceRes.x + threadsPerBlock.x - 1) / threadsPerBlock.x,
	               (surfaceRes.y + threadsPerBlock.y  - 1) / threadsPerBlock.y);

	// Run cuda ray tracer kernel
	cudaRayTracerKernel<<<numBlocks, threadsPerBlock>>>(surface, surfaceRes, cam, staticScene);
	CHECK_CUDA_ERROR(cudaGetLastError());
	CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

} // namespace phe
