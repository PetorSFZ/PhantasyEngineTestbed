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

inline __device__ vec3 vectorPow(const vec3& vector, float exponent) noexcept
{
	vec3 result;
	result.x = std::pow(vector.x, exponent);
	result.y = std::pow(vector.y, exponent);
	result.z = std::pow(vector.z, exponent);
	return result;
}

inline __device__ vec3 linearize(vec3 rgbGamma)
{
	return vectorPow(rgbGamma, 2.2f);
}

inline __device__ float linearize(float value)
{
	return std::pow(value, 2.2f);
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

__device__ vec3 shadeHit(const CudaTracerParams& params, curandState& randState, const Ray& ray,
                         const RayCastResult& hit, const HitInfo& info, const vec3& offsetHitPos,
                         const vec3& albedoColor, float metallic, float roughness) noexcept
{
	vec3 color = vec3(0.0f);

	for (uint32_t i = 0; i < params.numStaticSphereLights; i++) {
		SphereLight light = params.staticSphereLights[i];

		if (light.staticShadows) {
			vec3 lightPos = light.pos;
			vec3 lightDir = normalize(lightPos - offsetHitPos);

			// Slightly offset light ray to get stochastic soft shadows
			vec3 circleU;
			if (abs(info.normal.z) > 0.01f) {
				circleU = normalize(vec3(0.0f, -info.normal.z, info.normal.y));
			}
			else {
				circleU = normalize(vec3(-info.normal.y, info.normal.x, 0.0f));
			}
			vec3 circleV = cross(circleU, lightDir);

			float r1 = curand_uniform(&randState);
			float r2 = (2.0f * light.radius * curand_uniform(&randState)) - light.radius;
			float azimuthAngle = 2.0f * PI() * r1;

			vec3 lightPosOffset = circleU * cos(azimuthAngle) * r2 +
			                      circleV * sin(azimuthAngle) * r2;

			vec3 offsetLightDiff = lightPos + lightPosOffset - offsetHitPos;
			vec3 offsetLightDir = normalize(offsetLightDiff);

			Ray lightRay(offsetHitPos, offsetLightDir);

			RayCastResult lightHit = cudaCastRay(params.staticBvhNodesTex, params.staticTriangleVerticesTex,
				lightRay.origin, lightRay.dir, 0.0001f, length(offsetLightDiff), true);

			if (lightHit.triangleIndex != UINT32_MAX) {
				// In shadow, do not add light contribution
				continue;
			}
		}

		vec3 toLight = light.pos - info.pos;
		float toLightDist = length(toLight);

		// Early exit if light is out of range
		if (toLightDist > light.range) {
			continue;
		}

		vec3 l = toLight / toLightDist;
		vec3 v = normalize(-ray.dir);
		vec3 h = normalize(l + v);

		float nDotL = dot(info.normal, l);
		if (nDotL <= 0.0f) {
			continue;
		}

		float nDotV = dot(info.normal, v);

		nDotV = std::max(0.001f, nDotV);

		// Lambert diffuse
		vec3 diffuse = albedoColor / sfz::PI();

		// Cook-Torrance specular
		// Normal distribution function
		float nDotH = std::max(sfz::dot(info.normal, h), 0.0f); // max() should be superfluous here
		float ctD = ggx(nDotH, roughness * roughness);

		// Geometric self-shadowing term
		float k = pow(roughness + 1.0, 2) / 8.0;
		float ctG = geometricSchlick(nDotL, nDotV, k);

		// Fresnel function
		// Assume all dielectrics have a f0 of 0.04, for metals we assume f0 == albedo
		vec3 f0 = sfz::lerp(vec3(0.04f), albedoColor, metallic);
		vec3 ctF = fresnelSchlick(nDotL, f0);

		// Calculate final Cook-Torrance specular value
		vec3 specular = ctD * ctF * ctG / (4.0f * nDotL * nDotV);

		// Calculates light strength
		float fallofNumerator = pow(sfz::clamp(1.0f - std::pow(toLightDist / light.range, 4.0f), 0.0f, 1.0f), 2);
		float fallofDenominator = (toLightDist * toLightDist + 1.0);
		float falloff = fallofNumerator / fallofDenominator;
		vec3 lighting = falloff * light.strength;

		color += (diffuse + specular) * lighting * nDotL;
	}
	return color;
}

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
		RayCastResult hit = cudaCastRay(params.staticBvhNodesTex, params.staticTriangleVerticesTex, ray.origin, ray.dir);

		if (hit.triangleIndex == ~0u) {
			break;
		}

		HitInfo info = interpretHit(params.staticTriangleDatas, hit, ray);
		
		for (int i = 0; i < params.numDynBvhs; i++) {
			RayCastResult newHit = cudaCastRay(params.dynamicBvhNodesTex[i], params.dynamicTriangleVerticesTex[i], ray.origin, ray.dir);
			if (newHit.t < hit.t) {
				hit = newHit;
				if (hit.triangleIndex == ~0u) {
					break;
				}
				info = interpretHit(params.dynamicTriangleDatas[i], hit, ray);
			}
		}
		
		if (info.materialIndex == UINT32_MAX) {
			break;
		}

		const Material& material = params.materials[info.materialIndex];
		vec4 albedoValue = material.albedoValue();
		if (params.materials[info.materialIndex].albedoTexIndex() >= 0) {
			cudaTextureObject_t albedoTexture = params.textures[material.albedoTexIndex()];
			albedoValue = readMaterialTextureRGBA(albedoTexture, info.uv);
		}
		vec3 albedoColor = albedoValue.xyz;
		albedoColor = linearize(albedoColor);

		float metallic = material.metallicValue();
		if (params.materials[info.materialIndex].metallicTexIndex() >= 0) {
			cudaTextureObject_t metallicTexture = params.textures[material.metallicTexIndex()];
			metallic = readMaterialTextureGray(metallicTexture, info.uv);
		}
		metallic = linearize(metallic);

		float roughness = material.roughnessValue();
		if (params.materials[info.materialIndex].roughnessTexIndex() >= 0) {
			cudaTextureObject_t roughnessTexture = params.textures[material.roughnessTexIndex()];
			roughness = readMaterialTextureGray(roughnessTexture, info.uv);
		}
		roughness = linearize(roughness);

		vec3 offsetHitPos = info.pos + info.normal * 0.01f;

		color += mask * shadeHit(params, randState, ray, hit, info, offsetHitPos, albedoColor, metallic, roughness);

		// No need to find next ray at last bounce
		// TODO: Restructure loop to make more elegant
		if (pathDepth == PATH_LENGTH - 1) {
			break;
		}

		// Any later contrubutions are colored by this material's albedo
		mask *= albedoColor;
		// Temporarily amplify the ambiance effect
		// TODO: Weigh this properly using BRDF
		mask *= 2.0f;

		// To get ambient light, take cosine-weighted sample over the hemisphere
		// and trace in that direction.

		float r1 = curand_uniform(&randState);
		float r2 = curand_uniform(&randState);
		float azimuthAngle = 2.0f * PI() * r1;
		float altitudeFactor = sqrt(r2);

		// Find surface vectors u and v orthogonal to normal
		vec3 u;
		if (abs(info.normal.z) > 0.01f) {
			u = normalize(vec3(0.0f, -info.normal.z, info.normal.y));
		} else {
			u = normalize(vec3(-info.normal.y, info.normal.x, 0.0f));
		}
		vec3 v = cross(info.normal, u);

		rayDir = u * cos(azimuthAngle) * altitudeFactor +
		         v * sin(azimuthAngle) * altitudeFactor +
		         info.normal * sqrt(1 - r2);
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

	RayCastResult hit = cudaCastRay(params.staticBvhNodesTex, params.staticTriangleVerticesTex, ray.origin, ray.dir);

	for (int i = 0; i < params.numDynBvhs; i++) {
		RayCastResult newHit = cudaCastRay(params.dynamicBvhNodesTex[i], params.dynamicTriangleVerticesTex[i], ray.origin, ray.dir);
		if (newHit.t < hit.t) hit = newHit;
	}

	vec3 color;
	if (hit.triangleIndex != UINT32_MAX) {
		color = vec3(hit.u, hit.v, hit.t);
	} else {
		color = vec3(0.0f);
	}

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
	curand_init((id + 1) * seed, 0, 0, &params.curandStates[id]);
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
