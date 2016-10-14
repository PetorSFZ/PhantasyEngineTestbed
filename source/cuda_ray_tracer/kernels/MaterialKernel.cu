// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#include "kernels/MaterialKernel.hpp"

#include <phantasy_engine/level/SphereLight.hpp>
#include <phantasy_engine/ray_tracer_common/BVHNode.hpp>
#include <phantasy_engine/ray_tracer_common/Triangle.hpp>
#include <phantasy_engine/ray_tracer_common/Shading.hpp>
#include <phantasy_engine/rendering/Material.hpp>
#include <sfz/math/MathHelpers.hpp>

#include "CudaHelpers.hpp"
#include "CudaDeviceHelpers.cuh"
#include "CudaSfzVectorCompatibility.cuh"
#include "GBufferRead.cuh"

namespace phe {

using sfz::vec2;
using sfz::vec2i;
using sfz::vec3;
using sfz::vec3i;
using sfz::vec4;
using sfz::vec4i;


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

inline __device__ float linearize(float value)
{
	return std::pow(value, 2.2f);
}

struct HitInfo final {
	vec3 pos;
	vec3 normal;
	vec2 uv;
	uint32_t materialIndex;
};

SFZ_CUDA_CALLABLE HitInfo interpretHit(const TriangleData* triDatas, const RayHit& result,
	const RayIn& ray) noexcept
{
	const TriangleData& data = triDatas[result.triangleIndex];
	float u = result.u;
	float v = result.v;

	// Retrieving position
	HitInfo info;
	info.pos = ray.origin() + result.t * ray.dir();

	// Interpolating normal
	vec3 n0 = data.n0;
	vec3 n1 = data.n1;
	vec3 n2 = data.n2;
	info.normal = normalize(n0 + (n1 - n0) * u + (n2 - n0) * v);

	 // Interpolating uv coordinate
	vec2 uv0 = data.uv0;
	vec2 uv1 = data.uv1;
	vec2 uv2 = data.uv2;
	info.uv = uv0 + (uv1 - uv0) * u + (uv2 - uv0) * v;

	// Material index
	info.materialIndex = data.materialIndex;

	return info;
}

__device__ void shadeHit(PathState& pathState, curandState& randState, RayIn& shadowRay,
	const vec3& normal, const vec3& toCamera, const vec3& pos,
	const vec3& albedo, float metallic, float roughness,
	const SphereLight* sphereLights, uint32_t numSphereLights) noexcept
{
	vec3 offsetHitPos = pos + 0.01f * normal;

	vec3 p = pos;
	vec3 n = normal;

	vec3 v = toCamera; // to view

										  // Interpolation of normals sometimes makes them face away from the camera. Clamp
										  // these to almost zero, to not break shading calculations.
	float nDotV = fmaxf(0.001f, dot(n, v));

	// TEMP: Restrict to single light source
	numSphereLights = 2;
	for (uint32_t i = 1; i < numSphereLights; i++) {
		// Retrieve light source
		SphereLight light = sphereLights[i];
		vec3 toLight = light.pos - p;
		float toLightDist = length(toLight);

		// Check if surface is in range of light source
		if (toLightDist > light.range) continue;

		// Shading parameters
		vec3 l = toLight / toLightDist; // to light (normalized)
		vec3 h = normalize(l + v); // half vector (normal of microfacet)

								   // If nDotL is <= 0 then the light source is not in the hemisphere of the surface, i.e.
								   // no shading needs to be performed
		float nDotL = dot(n, l);
		if (nDotL <= 0.0f) continue;

		// Lambert diffuse
		vec3 diffuse = albedo / float(sfz::PI());

		// Cook-Torrance specular
		// Normal distribution function
		float nDotH = fmaxf(0.0f, dot(n, h)); // max() should be superfluous here
		float ctD = ggx(nDotH, roughness * roughness);

		// Geometric self-shadowing term
		float k = powf(roughness + 1.0f, 2.0f) / 8.0f;
		float ctG = geometricSchlick(nDotL, nDotV, k);

		// Fresnel function
		// Assume all dielectrics have a f0 of 0.04, for metals we assume f0 == albedo
		vec3 f0 = lerp(vec3(0.04f), albedo, metallic);
		vec3 ctF = fresnelSchlick(nDotL, f0);

		// Calculate final Cook-Torrance specular value
		vec3 specular = ctD * ctF * ctG / (4.0f * nDotL * nDotV);

		// Calculates light strength
		float fallofNumerator = powf(clamp(1.0f - powf(toLightDist / light.range, 4.0f), 0.0f, 1.0f), 2.0f);
		float fallofDenominator = (toLightDist * toLightDist + 1.0);
		float falloff = fallofNumerator / fallofDenominator;
		vec3 lightContrib = falloff * light.strength;

		// "Solves" reflectance equation under the assumption that the light source is a point light
		// and that there is no global illumination.
		vec3 color = pathState.throughput * (diffuse + specular) * lightContrib * nDotL;

		if (light.staticShadows) {
			// Slightly offset light ray to get stochastic soft shadows
			vec3 circleU;
			if (abs(normal.z) > 0.01f) {
				circleU = normalize(vec3(0.0f, -normal.z, normal.y));
			}
			else {
				circleU = normalize(vec3(-normal.y, normal.x, 0.0f));
			}
			vec3 circleV = cross(circleU, toLight);

			float r1 = curand_uniform(&randState);
			float r2 = curand_uniform(&randState);
			float centerDistance = light.radius * (2.0f * r2 - 1.0f);
			float azimuthAngle = 2.0f * sfz::PI() * r1;

			vec3 lightPosOffset = circleU * cos(azimuthAngle) * centerDistance +
			                      circleV * sin(azimuthAngle) * centerDistance;

			vec3 offsetLightDiff = light.pos + lightPosOffset - offsetHitPos;
			vec3 offsetLightDir = normalize(offsetLightDiff);

			shadowRay.setOrigin(offsetHitPos);
			shadowRay.setDir(offsetLightDir);
			shadowRay.setMaxDist(length(offsetLightDiff));
			shadowRay.setNoResultOnlyHit(true);

			pathState.pendingLightContribution = color;
		}
		else {
			pathState.finalColor += color;
		}
	}
}

static __global__ void materialKernel(
	vec2i res,
	RayIn* shadowRays,
	PathState* pathStates,
	curandState* randStates,
	const RayIn* rays,
	const RayHit* rayHits,
	const TriangleData* staticTriangleDatas,
	const Material* materials,
	const cudaTextureObject_t* textures,
	const SphereLight* sphereLights,
	uint32_t numSphereLights)
{
	// Calculate surface coordinates
	vec2i loc = vec2i(blockIdx.x * blockDim.x + threadIdx.x,
	                  blockIdx.y * blockDim.y + threadIdx.y);
	if (loc.x >= res.x || loc.y >= res.y) return;

	uint32_t id = loc.y * res.x + loc.x;
	PathState& pathState = pathStates[id];
	RayIn& shadowRay = shadowRays[id];
	curandState randState = randStates[id];

	RayHit hit = rayHits[id];
	RayIn ray = rays[id];

	if (hit.triangleIndex == ~0u) {
		return;
	}

	HitInfo info = interpretHit(staticTriangleDatas, hit, ray);

	const Material& material = materials[info.materialIndex];
	vec4 albedoValue = material.albedoValue();
	if (materials[info.materialIndex].albedoTexIndex() != UINT32_MAX) {
		cudaTextureObject_t albedoTexture = textures[material.albedoTexIndex()];
		albedoValue = readMaterialTextureRGBA(albedoTexture, info.uv);
	}
	vec3 albedoColor = albedoValue.xyz;
	albedoColor = linearize(albedoColor);

	float metallic = material.metallicValue();
	if (materials[info.materialIndex].metallicTexIndex() != UINT32_MAX) {
		cudaTextureObject_t metallicTexture = textures[material.metallicTexIndex()];
		metallic = readMaterialTextureGray(metallicTexture, info.uv);
	}
	metallic = linearize(metallic);

	float roughness = material.roughnessValue();
	if (materials[info.materialIndex].roughnessTexIndex() != UINT32_MAX) {
		cudaTextureObject_t roughnessTexture = textures[material.roughnessTexIndex()];
		roughness = readMaterialTextureGray(roughnessTexture, info.uv);
	}
	roughness = linearize(roughness);

	shadeHit(pathState, randState, shadowRay, info.normal, -ray.dir(), info.pos, albedoColor, metallic, roughness, sphereLights, numSphereLights);
	randStates[id] = randState;
}

void launchMaterialKernel(const MaterialKernelInput& input) noexcept
{
	// Calculate number of threads and blocks to run
	dim3 threadsPerBlock(8, 8);
	dim3 numBlocks((input.res.x + threadsPerBlock.x - 1) / threadsPerBlock.x,
	               (input.res.y + threadsPerBlock.y - 1) / threadsPerBlock.y);

	// Run cuda ray tracer kernel
	materialKernel<<<numBlocks, threadsPerBlock>>>(input.res, input.shadowRays, input.pathStates, input.randStates, input.rays, input.rayHits, input.staticTriangleDatas, input.materials, input.textures, input.sphereLights, input.numSphereLights);
	CHECK_CUDA_ERROR(cudaGetLastError());
	CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

static __global__ void gBufferMaterialKernel(
	vec2i res,
	vec3 camPos,
	RayIn* extensionRays,
	RayIn* shadowRays,
	PathState* pathStates,
	curandState* randStates,
	cudaSurfaceObject_t posTex,
	cudaSurfaceObject_t normalTex,
	cudaSurfaceObject_t albedoTex,
	cudaSurfaceObject_t materialTex,
	const SphereLight* sphereLights,
	uint32_t numSphereLights)
{
	// Calculate surface coordinates
	vec2i loc = vec2i(blockIdx.x * blockDim.x + threadIdx.x,
		blockIdx.y * blockDim.y + threadIdx.y);
	if (loc.x >= res.x || loc.y >= res.y) return;

	GBufferValue gBufferValue = readGBuffer(posTex, normalTex, albedoTex, materialTex, loc);

	uint32_t id = loc.y * res.x + loc.x;
	PathState& pathState = pathStates[id];
	RayIn& shadowRay = shadowRays[id];
	curandState randState = randStates[id];

	vec3 toCamera = normalize(camPos - gBufferValue.pos);

	shadeHit(pathState, randState, shadowRay, gBufferValue.normal, toCamera, gBufferValue.pos, gBufferValue.albedo, gBufferValue.metallic, gBufferValue.roughness, sphereLights, numSphereLights);

	pathState.throughput *= gBufferValue.albedo;
	pathState.throughput *= 0.5f;

	// To get ambient light, take cosine-weighted sample over the hemisphere
	// and trace in that direction.

	RayIn& extensionRay = extensionRays[id];

	float r1 = curand_uniform(&randState);
	float r2 = curand_uniform(&randState);
	float azimuthAngle = 2.0f * sfz::PI() * r1;
	float altitudeFactor = sqrt(r2);

	// Find surface vectors u and v orthogonal to normal
	vec3 u;
	if (abs(gBufferValue.normal.z) > 0.01f) {
		u = normalize(vec3(0.0f, -gBufferValue.normal.z, gBufferValue.normal.y));
	}
	else {
		u = normalize(vec3(-gBufferValue.normal.y, gBufferValue.normal.x, 0.0f));
	}
	vec3 v = cross(gBufferValue.normal, u);

	vec3 rayDir = u * cos(azimuthAngle) * altitudeFactor +
	              v * sin(azimuthAngle) * altitudeFactor +
	              gBufferValue.normal * sqrt(1 - r2);

	extensionRay.setOrigin(offsetHitPos);
	extensionRay.setDir(rayDir);
	extensionRay.setMaxDist(FLT_MAX);
	extensionRay.setNoResultOnlyHit(false);

	randStates[id] = randState;
}

void launchGBufferMaterialKernel(const GBufferMaterialKernelInput& input) noexcept
{
	// Calculate number of threads and blocks to run
	dim3 threadsPerBlock(8, 8);
	dim3 numBlocks((input.res.x + threadsPerBlock.x - 1) / threadsPerBlock.x,
	               (input.res.y + threadsPerBlock.y - 1) / threadsPerBlock.y);

	// Run cuda ray tracer kernel
	gBufferMaterialKernel<<<numBlocks, threadsPerBlock>>>(input.res, input.camPos, input.extensionRays, input.shadowRays, input.pathStates, input.randState, input.posTex, input.normalTex, input.albedoTex, input.materialTex, input.sphereLights, input.numSphereLights);

	CHECK_CUDA_ERROR(cudaGetLastError());
	CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

static __global__ void initPathStates(vec2i res, PathState* pathStates)
{
	// Calculate surface coordinates
	vec2i loc = vec2i(blockIdx.x * blockDim.x + threadIdx.x,
	                  blockIdx.y * blockDim.y + threadIdx.y);
	if (loc.x >= res.x || loc.y >= res.y) return;

	// Read rayhit from array
	uint32_t id = loc.y * res.x + loc.x;

	PathState& pathState = pathStates[id];
	pathState.finalColor = vec3(0.0f);
	pathState.pathLength = 0;
	pathState.pendingLightContribution = vec3(0.0f);
	pathState.throughput = vec3(1.0f);
}

void launchInitPathStatesKernel(vec2i res, PathState* pathStates) noexcept
{
	// Calculate number of threads and blocks to run
	dim3 threadsPerBlock(8, 8);
	dim3 numBlocks((res.x + threadsPerBlock.x - 1) / threadsPerBlock.x,
	               (res.y + threadsPerBlock.y - 1) / threadsPerBlock.y);

	// Run cuda ray tracer kernel
	initPathStates<<<numBlocks, threadsPerBlock>>>(res, pathStates);
	CHECK_CUDA_ERROR(cudaGetLastError());
	CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

static __global__ void shadowLogicKernel(vec2i res, const bool* shadowRayHits, PathState* pathStates)
{
	// Calculate surface coordinates
	vec2i loc = vec2i(blockIdx.x * blockDim.x + threadIdx.x,
	                  blockIdx.y * blockDim.y + threadIdx.y);
	if (loc.x >= res.x || loc.y >= res.y) return;

	uint32_t id = loc.y * res.x + loc.x;

	PathState& pathState = pathStates[id];
	const bool inLight = shadowRayHits[id];
	if (inLight) {
		pathState.finalColor += pathState.pendingLightContribution;
	}
}

void launchShadowLogicKernel(vec2i res, const bool* shadowRayHits, PathState* pathStates) noexcept
{
	// Calculate number of threads and blocks to run
	dim3 threadsPerBlock(8, 8);
	dim3 numBlocks((res.x + threadsPerBlock.x - 1) / threadsPerBlock.x,
	               (res.y + threadsPerBlock.y - 1) / threadsPerBlock.y);

	// Run cuda ray tracer kernel
	shadowLogicKernel<<<numBlocks, threadsPerBlock>>>(res, shadowRayHits, pathStates);
	CHECK_CUDA_ERROR(cudaGetLastError());
	CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

static __global__ void writeResultKernel(cudaSurfaceObject_t surface, vec2i res, PathState* pathStates)
{
	// Calculate surface coordinates
	vec2i loc = vec2i(blockIdx.x * blockDim.x + threadIdx.x,
	                  blockIdx.y * blockDim.y + threadIdx.y);
	if (loc.x >= res.x || loc.y >= res.y) return;

	uint32_t id = loc.y * res.x + loc.x;

	PathState& pathState = pathStates[id];

	vec4 color4 = vec4(pathState.finalColor, 1.0f);
	surf2Dwrite(toFloat4(color4), surface, loc.x * sizeof(float4), loc.y);
}

void launchWriteResultKernel(const WriteResultKernelInput& input) noexcept
{
	// Calculate number of threads and blocks to run
	dim3 threadsPerBlock(8, 8);
	dim3 numBlocks((input.res.x + threadsPerBlock.x - 1) / threadsPerBlock.x,
	               (input.res.y + threadsPerBlock.y - 1) / threadsPerBlock.y);

	// Run cuda ray tracer kernel
	writeResultKernel<<<numBlocks, threadsPerBlock>>>(input.surface, input.res, input.pathStates);
	CHECK_CUDA_ERROR(cudaGetLastError());
	CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

} // namespace phe
