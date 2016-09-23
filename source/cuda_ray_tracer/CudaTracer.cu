// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#include "CudaTracer.cuh"

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <sfz/math/Vector.hpp>

#include "BVHTraversal.cuh"
#include "CudaHelpers.hpp"
#include "CudaSfzVectorCompatibility.cuh"

namespace phe {

using namespace sfz;

// Material access helpers
// ------------------------------------------------------------------------------------------------

__device__ float readMaterialTextureGray(cudaTextureObject_t texture, vec2 coord) noexcept {
	uchar1 res = tex2D<uchar1>(texture, coord.x, coord.y);
	return float(res.x) / 255.0f;
}

__device__ vec4 readMaterialTextureRGBA(cudaTextureObject_t texture, vec2 coord) noexcept
{
	uchar4 res = tex2D<uchar4>(texture, coord.x, coord.y);
	return vec4(float(res.x), float(res.y), float(res.z), float(res.w)) / 255.0f;
}

// Surface manipulation helpers
// ------------------------------------------------------------------------------------------------

inline __device__ void writeToSurface(const cudaSurfaceObject_t& surface, vec2i loc, const vec4& data) noexcept
{
	float4 dataFloat4 = toFloat4(data);
	surf2Dwrite(dataFloat4, surface, loc.x * sizeof(float4), loc.y);
}

inline __device__ void addToSurface(const cudaSurfaceObject_t& surface, vec2i loc, const vec4& data) noexcept {
	float4 existing;
	surf2Dread(&existing, surface, loc.x * sizeof(float4), loc.y);
	float4 newData = toFloat4(toSFZ(existing) + data);
	surf2Dwrite(newData, surface, loc.x * sizeof(float4), loc.y);
}

// Primary ray helpers
// ------------------------------------------------------------------------------------------------

inline __device__ vec3 calculatePrimaryRayDir(const CameraDef& cam, vec2 loc, vec2 surfaceRes) noexcept
{
	vec2 locNormalized = loc / surfaceRes; // [0, 1]
	vec2 centerOffsCoord = locNormalized * 2.0f - vec2(1.0f); // [-1.0, 1.0]
	vec3 nonNormRayDir = cam.dir + cam.dX * centerOffsCoord.x + cam.dY * centerOffsCoord.y;
	return normalize(nonNormRayDir);
}

inline __device__ vec3 calculateRandomizedPrimaryRayDir(const CameraDef& cam, vec2 loc, vec2 surfaceRes, curandState& randState) noexcept
{
	vec2 locNormalized = loc / surfaceRes; // [0, 1]
	vec2 centerOffsCoord = locNormalized * 2.0f - vec2(1.0f); // [-1.0, 1.0]

	float r1 = curand_uniform(&randState);
	float r2 = curand_uniform(&randState);
	vec2 noiseOffset = (2.0f * vec2(r1, r2) - vec2(1.0f)) / surfaceRes;
	vec2 randomizedCoord = noiseOffset + centerOffsCoord;

	vec3 nonNormRayDir = cam.dir + cam.dX * randomizedCoord.x + cam.dY * randomizedCoord.y;
	return normalize(nonNormRayDir);
}

// Main ray tracing kernel
// ------------------------------------------------------------------------------------------------

__global__ void cudaRayTracerKernel(CudaTracerParams params)
{
	// Calculate surface coordinates
	vec2i loc = vec2i(blockIdx.x * blockDim.x + threadIdx.x,
	                  blockIdx.y * blockDim.y + threadIdx.y);
	if (loc.x >= params.targetRes.x || loc.y >= params.targetRes.y) return;

	// Find identifier for this pixel, to use as random seed 
	uint32_t id = loc.x + loc.y * params.targetRes.x;

	// Copy RNG state to local memory
	curandState randState = params.curandStates[id];

	// Find initial ray from camera
	vec3 rayDir = calculateRandomizedPrimaryRayDir(params.cam, vec2(loc), vec2(params.targetRes), randState);
	Ray ray(params.cam.origin, rayDir);

	// Initialize the final color value
	vec3 color = vec3(0.0f);

	// Initialize mask which later will color future light contributions, based on materials
	// having been passed along the path
	vec3 mask = vec3(1.0f);

	const uint32_t PATH_LENGTH = 2;
	for (int pathDepth = 0; pathDepth < PATH_LENGTH; pathDepth++) {
		RayCastResult hit = cudaCastRay(params.staticBvhNodesTex, params.staticTriangleVerticesTex, ray);
		if (hit.triangleIndex == ~0u) {
			break;
		}

		HitInfo info = interpretHit(params.staticTriangleDatas, hit, ray);

		if (info.materialIndex == UINT32_MAX) {
			break;
		}

		const Material& material = params.materials[info.materialIndex];
		vec4 albedoColor = material.albedoValue;
		if (params.materials[info.materialIndex].albedoIndex != UINT32_MAX) {
			cudaTextureObject_t albedoTexture = params.textures[material.albedoIndex];
			albedoColor = readMaterialTextureRGBA(albedoTexture, info.uv);
		}
		albedoColor.xyz = vec3(1, 1, 1);

		float metallic = material.metallicValue;
		if (params.materials[info.materialIndex].metallicIndex != UINT32_MAX) {
			cudaTextureObject_t metallicTexture = params.textures[material.metallicIndex];
			metallic = readMaterialTextureGray(metallicTexture, info.uv);
		}

		float roughness = material.roughnessValue;
		if (params.materials[info.materialIndex].roughnessIndex != UINT32_MAX) {
			cudaTextureObject_t roughnessTexture = params.textures[material.roughnessIndex];
			roughness = readMaterialTextureGray(roughnessTexture, info.uv);
		}

		// Any later contrubutions are colored by this material's albedo
		mask *= albedoColor.xyz;

		vec3 offsetHitPos = info.pos + info.normal * 0.01f;

		// Check if directly illuminated by light source by casting light ray
		SphereLight& light = params.staticSphereLights[0];
		vec3 lightPos = light.pos;
		vec3 lightDir = lightPos - offsetHitPos; // Intentionally not normalized!

		vec3 tmpVec = lightDir.x > 0.01f ? vec3(0.0f, 1.0f, 0.0f) : vec3(1.0f, 0.0f, 0.0f);
		vec3 u = normalize(cross(tmpVec, lightDir));
		vec3 v = normalize(cross(u, lightDir));

		float r1 = curand_uniform(&randState);
		float r2 = (2.0f * light.radius * curand_uniform(&randState)) - light.radius;
		float azimuthAngle = 2.0f * PI() * r1;

		vec3 lightPosOffset = u * cos(azimuthAngle) * r2 +
			v * sin(azimuthAngle) * r2;

		RayCastResult lightHit = castRay(params.staticBvhNodes, params.staticTriangleVertices, Ray(offsetHitPos, lightDir + lightPosOffset), 0.0001f, 1.0f);

		// If there was no intersection, the point is directly illuminated
		if (lightHit.triangleIndex == UINT32_MAX) {
			// Add contribution of light source
			vec3 l = -normalize(info.pos - lightPos);
			float diffuseFactor = max(dot(l, info.normal), 0.0f);
			color += mask * diffuseFactor * vec3(0.6f);
		}

		// No need to find next ray at last bounce
		// TODO: Restructure loop to make more elegant
		if (pathDepth == PATH_LENGTH - 1) {
			break;
		}

		// TODO: Use BRDF
		if (roughness < 0.2f) {
			// Let next ray be perfect mirror reflection
			rayDir = rayDir - 2 * (dot(rayDir, info.normal)) * info.normal;
		}
		else {
			// Treat material as fully diffuse. Take cosine-weighted sample over the hemisphere
			// and trace in that direction.

			float r1 = curand_uniform(&randState);
			float r2 = curand_uniform(&randState);
			float azimuthAngle = 2.0f * PI() * r1;
			float altitudeFactor = sqrtf(r2);

			// Find surface vectors u and v orthogonal to normal, using a tempVector not parallel
			// to normal
			vec3 tempVector = abs(info.normal.x) > 0.01f ? vec3(0.0f, 1.0f, 0.0f) : vec3(1.0f, 0.0f, 0.0f);
			vec3 u = cross(tempVector, info.normal);
			vec3 v = cross(info.normal, u);

			rayDir = u * cos(azimuthAngle) * altitudeFactor +
			         v * sin(azimuthAngle) * altitudeFactor +
			         info.normal * sqrtf(1 - r2);
		}
		ray = Ray(offsetHitPos, rayDir);
	}

	addToSurface(params.targetSurface, loc, vec4(color, 1.0));

	// Copy back updated RNG state
	params.curandStates[id] = randState;
}

void cudaRayTrace(const CudaTracerParams& params) noexcept
{
	vec2i surfaceRes = params.targetRes;

	// Calculate number of threads and blocks to run
	dim3 threadsPerBlock(8, 8);
	dim3 numBlocks((surfaceRes.x + threadsPerBlock.x - 1) / threadsPerBlock.x,
	               (surfaceRes.y + threadsPerBlock.y - 1) / threadsPerBlock.y);

	// Run cuda ray tracer kernel
	cudaRayTracerKernel<<<numBlocks, threadsPerBlock>>>(params);
	CHECK_CUDA_ERROR(cudaGetLastError());
	CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

// Simple castRay kernel
// ------------------------------------------------------------------------------------------------

__global__ void castRayTestKernel(CudaTracerParams params)
{
	// Calculate surface coordinates
	vec2i loc = vec2i(blockIdx.x * blockDim.x + threadIdx.x,
	                  blockIdx.y * blockDim.y + threadIdx.y);
	if (loc.x >= params.targetRes.x || loc.y >= params.targetRes.y) return;

	// Find initial ray from camera
	vec3 rayDir = calculatePrimaryRayDir(params.cam, vec2(loc), vec2(params.targetRes));
	Ray ray(params.cam.origin, rayDir);

	RayCastResult hit = cudaCastRay(params.staticBvhNodesTex, params.staticTriangleVerticesTex, ray);

	vec3 color(hit.u, hit.v, hit.t);

	writeToSurface(params.targetSurface, loc, vec4(color, 1.0));
}

void cudaCastRayTest(const CudaTracerParams& params) noexcept
{
	// Calculate number of threads and blocks to run
	dim3 threadsPerBlock(8, 8);
	dim3 numBlocks((params.targetRes.x + threadsPerBlock.x - 1) / threadsPerBlock.x,
	               (params.targetRes.y + threadsPerBlock.y - 1) / threadsPerBlock.y);

	// Run cuda ray tracer kernel
	castRayTestKernel<<<numBlocks, threadsPerBlock>>>(params);
	CHECK_CUDA_ERROR(cudaGetLastError());
	CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

// Heatmap kernel
// ------------------------------------------------------------------------------------------------

__global__ void heatmapKernel(CudaTracerParams params)
{
	// Calculate surface coordinates
	vec2i loc = vec2i(blockIdx.x * blockDim.x + threadIdx.x,
	                  blockIdx.y * blockDim.y + threadIdx.y);
	if (loc.x >= params.targetRes.x || loc.y >= params.targetRes.y) return;

	// Calculate ray direction
	vec3 rayDir = calculatePrimaryRayDir(params.cam, vec2(loc), vec2(params.targetRes));
	Ray ray(params.cam.origin, rayDir);

	// Ray cast against BVH
	DebugRayCastData debugData;
	RayCastResult hit = castDebugRay(params.staticBvhNodes, params.staticTriangleVertices, ray, &debugData);

	const float NODES_VISITED_CUTOFF = 150.0f;
	const vec3 WHITE(1.0f);
	const vec3 RED(1.0f, 0.0f, 0.0f);

	// Set color according to the number of nodes needed to be visited, so red marks an area which
	// takes longer to trace.
	vec3 color = sfz::lerp(WHITE, RED, float(debugData.nodesVisited) / NODES_VISITED_CUTOFF);

	writeToSurface(params.targetSurface, loc, vec4(color, 1.0));
}

void cudaHeatmapTrace(const CudaTracerParams& params) noexcept
{
	// Calculate number of threads and blocks to run
	dim3 threadsPerBlock(8, 8);
	dim3 numBlocks((params.targetRes.x + threadsPerBlock.x - 1) / threadsPerBlock.x,
	               (params.targetRes.y + threadsPerBlock.y - 1) / threadsPerBlock.y);

	// Run cuda ray tracer kernel
	heatmapKernel<<<numBlocks, threadsPerBlock>>>(params);
	CHECK_CUDA_ERROR(cudaGetLastError());
	CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

// Curand initialization kernel
// ------------------------------------------------------------------------------------------------

__global__ void initCurandKernel(CudaTracerParams params, unsigned long long seed)
{
	// Calculate surface coordinates
	vec2i loc = vec2i(blockIdx.x * blockDim.x + threadIdx.x,
	                  blockIdx.y * blockDim.y + threadIdx.y);
	if (loc.x >= params.targetRes.x || loc.y >= params.targetRes.y) return;

	uint32_t id = loc.x + loc.y * params.targetRes.x;
	curand_init(seed, id, 0, &params.curandStates[id]);
}

void initCurand(const CudaTracerParams& params, unsigned long long seed) noexcept
{
	// Calculate number of threads and blocks to run
	dim3 threadsPerBlock(8, 8);
	dim3 numBlocks((params.targetRes.x + threadsPerBlock.x - 1) / threadsPerBlock.x,
	               (params.targetRes.y + threadsPerBlock.y - 1) / threadsPerBlock.y);

	// Initialize all curand states
	initCurandKernel<<<numBlocks, threadsPerBlock>>>(params, seed);
	CHECK_CUDA_ERROR(cudaGetLastError());
	CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

// Clear surface kernel
// ------------------------------------------------------------------------------------------------

__global__ void clearSurfaceKernel(cudaSurfaceObject_t targetSurface, vec2i targetRes, vec4 color)
{
	vec2i loc = vec2i(blockIdx.x * blockDim.x + threadIdx.x,
	                  blockIdx.y * blockDim.y + threadIdx.y);
	if (loc.x >= targetRes.x || loc.y >= targetRes.y) return;

	writeToSurface(targetSurface, loc, color);
}

void cudaClearSurface(const cudaSurfaceObject_t& targetSurface, const vec2i& targetRes, const vec4& color) noexcept
{
	vec2i surfaceRes = targetRes;

	// Calculate number of threads and blocks to run
	dim3 threadsPerBlock(8, 8);
	dim3 numBlocks((surfaceRes.x + threadsPerBlock.x - 1) / threadsPerBlock.x,
	               (surfaceRes.y + threadsPerBlock.y - 1) / threadsPerBlock.y);

	// Run cuda ray tracer kernel
	clearSurfaceKernel<<<numBlocks, threadsPerBlock>>>(targetSurface, surfaceRes, color);
	CHECK_CUDA_ERROR(cudaGetLastError());
	CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

} // namespace phe
