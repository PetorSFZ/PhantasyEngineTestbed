// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#include "kernels/MaterialKernel.hpp"

#include <phantasy_engine/level/SphereLight.hpp>

#include "CudaHelpers.hpp"
#include "CudaDeviceHelpers.cuh"
#include "CudaPbr.cuh"
#include "CudaSfzVectorCompatibility.cuh"
#include "GBufferRead.cuh"

namespace phe {

using sfz::vec2;
using sfz::vec2i;
using sfz::vec3;
using sfz::vec3i;
using sfz::vec4;
using sfz::vec4i;

// Surface manipulation helpers
// ------------------------------------------------------------------------------------------------

inline __device__ void writeToSurface(const cudaSurfaceObject_t& surface, vec2i loc, const vec4& data) noexcept
{
	float4 dataFloat4 = toFloat4(data);
	surf2Dwrite(dataFloat4, surface, loc.x * sizeof(float4), loc.y);
}

inline __device__ vec4 readFromSurface(const cudaSurfaceObject_t& surface, vec2i loc) noexcept {
	float4 data;
	surf2Dread(&data, surface, loc.x * sizeof(float4), loc.y);
	return toSFZ(data);
}

// Shading
// ------------------------------------------------------------------------------------------------

/// Set ray to not hit anything and be cheap to evaluate in ray casting.
static __device__ void setToDummyRay(RayIn& ray)
{
	ray.setOrigin(vec3(1000000.0f));
	ray.setDir(vec3(0.0f, 1.0f, 0.0f));
	ray.setMinDist(0.001f);
	ray.setMaxDist(0.001f);
}

static __device__ void shadeHit(uint32_t id, const vec3& mask, curandState& randState,
                                RayIn* shadowRays, vec3* lightContributions,
                                const vec3& normal, const vec3& toCamera, const vec3& pos,
                                const vec3& albedo, float metallic, float roughness,
                                const SphereLight* staticSphereLights,
                                uint32_t numStaticSphereLights) noexcept
{
	vec3 p = pos;
	vec3 n = normal;

	vec3 v = toCamera; // to view

	// Interpolation of normals sometimes makes them face away from the camera. Clamp
	// these to almost zero, to not break shading calculations.
	float nDotV = fmaxf(0.001f, dot(n, v));

	for (uint32_t i = 0; i < numStaticSphereLights; i++) {
		SphereLight light = staticSphereLights[i];

		vec3 color = vec3(0.0f);

		// Initialize shadow ray to dummy. A future improvement would be to only queue up as many
		// RayIns as are actually needed
		RayIn shadowRay;
		setToDummyRay(shadowRay);

		vec3 toLight = light.pos - p;
		float toLightDist = length(toLight);

		// Check if surface is in range of light source
		if (toLightDist <= light.range) {
			// Shading parameters
			vec3 l = toLight / toLightDist; // to light (normalized)
			vec3 h = normalize(l + v); // half vector (normal of microfacet)

			// If nDotL is <= 0 then the light source is not in the hemisphere of the surface, i.e.
			// no shading needs to be performed
			float nDotL = dot(n, l);
			if (nDotL > 0.0f) {
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
				color = mask * (diffuse + specular) * lightContrib * nDotL;

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

					vec3 offsetLightDiff = light.pos + lightPosOffset - pos;
					vec3 offsetLightDir = normalize(offsetLightDiff);

					shadowRay.setOrigin(pos);
					shadowRay.setDir(offsetLightDir);
					shadowRay.setMinDist(0.001f);
					shadowRay.setMaxDist(length(offsetLightDiff));
				}
			}
		}
		uint32_t shadowRayID = id * numStaticSphereLights + i;
		shadowRays[shadowRayID] = shadowRay;
		lightContributions[shadowRayID] = color;
	}
}

// Main material kernels
// ------------------------------------------------------------------------------------------------

static __global__ void gBufferMaterialKernel(GBufferMaterialKernelInput input)
{
	// Calculate surface coordinates
	vec2i loc = vec2i(blockIdx.x * blockDim.x + threadIdx.x,
	                  blockIdx.y * blockDim.y + threadIdx.y);
	if (loc.x >= input.res.x || loc.y >= input.res.y) return;

	GBufferValue gBufferValue = readGBuffer(input.posTex, input.normalTex, input.albedoTex,
	                                        input.materialTex, loc);

	uint32_t id = loc.y * input.res.x + loc.x;

	curandState randState = input.randStates[id];

	vec3 toCamera = normalize(input.camPos - gBufferValue.pos);

	shadeHit(id, vec3(1.0f), randState, input.shadowRays, input.lightContributions,
	         gBufferValue.normal, toCamera, gBufferValue.pos, gBufferValue.albedo,
	         gBufferValue.metallic, gBufferValue.roughness,
	         input.staticSphereLights, input.numStaticSphereLights);

	input.randStates[id] = randState;
}

void launchGBufferMaterialKernel(const GBufferMaterialKernelInput& input) noexcept
{
	// Calculate number of threads and blocks to run
	dim3 threadsPerBlock(8, 8);
	dim3 numBlocks((input.res.x + threadsPerBlock.x - 1) / threadsPerBlock.x,
	               (input.res.y + threadsPerBlock.y - 1) / threadsPerBlock.y);

	gBufferMaterialKernel<<<numBlocks, threadsPerBlock>>>(input);

	CHECK_CUDA_ERROR(cudaGetLastError());
	CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

static __global__ void materialKernel(MaterialKernelInput input)
{
	// Calculate surface coordinates
	vec2i loc = vec2i(blockIdx.x * blockDim.x + threadIdx.x,
	                  blockIdx.y * blockDim.y + threadIdx.y);
	if (loc.x >= input.res.x || loc.y >= input.res.y) return;

	uint32_t id = loc.y * input.res.x + loc.x;

	RayHitInfo hitInfo = input.rayHitInfos[id];

	if (!hitInfo.wasHit()) {
		// If nothing was hit, return black. Set shadow ray to dummy. A future improvement would
		// be to only queue up as many RayIns as are actually needed
		RayIn dummyRay;
		setToDummyRay(dummyRay);
		for (uint32_t i = 0; i < input.numStaticSphereLights; i++) {
			uint32_t shadowRayID = id * input.numStaticSphereLights + i;
			input.shadowRays[shadowRayID] = dummyRay;
			input.lightContributions[shadowRayID] = vec3(0.0f);
		}
		return;
	}

	PathState& pathState = input.pathStates[id];
	curandState randState = input.randStates[id];
	RayIn ray = input.rays[id];

	shadeHit(id, pathState.throughput, randState, input.shadowRays, input.lightContributions,
	         hitInfo.normal(), -ray.dir(), hitInfo.position(), hitInfo.albedo(), hitInfo.metallic(),
	         hitInfo.roughness(), input.staticSphereLights, input.numStaticSphereLights);
	input.randStates[id] = randState;
}

void launchMaterialKernel(const MaterialKernelInput& input) noexcept
{
	// Calculate number of threads and blocks to run
	dim3 threadsPerBlock(8, 8);
	dim3 numBlocks((input.res.x + threadsPerBlock.x - 1) / threadsPerBlock.x,
	               (input.res.y + threadsPerBlock.y - 1) / threadsPerBlock.y);

	materialKernel<<<numBlocks, threadsPerBlock>>>(input);
	CHECK_CUDA_ERROR(cudaGetLastError());
	CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

// Importance sampling helpers
// ------------------------------------------------------------------------------------------------

inline __device__ vec3 hemisphereToWorld(float phi, float cosTheta, float sinTheta, const vec3& normal)
{
	vec3 hemisphereCoord;
	hemisphereCoord.x = sinTheta * cos(phi);
	hemisphereCoord.y = sinTheta * sin(phi);
	hemisphereCoord.z = cosTheta;

	vec3 upVector = abs(normal.z) < 0.999f ? vec3(0.0f, 0.0f, 1.0f) : vec3(1.0f, 0.0f, 0.0f);
	vec3 u = sfz::normalize(sfz::cross(upVector, normal));
	vec3 v = sfz::cross(normal, u);

	return u * hemisphereCoord.x + v * hemisphereCoord.y + normal * hemisphereCoord.z;
}

/// Move vector facing into a surface up to the tangent plane of the surface.
inline __device__ vec3 clampToHemisphere(const vec3& vector, const vec3& surfaceNormal)
{
	float nDotV = dot(vector, surfaceNormal);
	float normalOffsetDist = -std::min(0.0f, nDotV);
	return sfz::normalize(vector + normalOffsetDist * surfaceNormal);
}

/// Return a GGX distributed sample around the reflection vector.
inline __device__ vec3 sampleGgx(const vec2& randVec, float a, const vec3& reflection, const vec3& surfaceNormal)
{
	float phi = 2.0f * sfz::PI() * randVec.x;
	float cosTheta = sqrt((1.0f - randVec.y) / (1.0f + (a * a - 1.0f) * randVec.y));
	float sinTheta = sqrt(1.0f - cosTheta * cosTheta);

	return clampToHemisphere(hemisphereToWorld(phi, cosTheta, sinTheta, reflection), surfaceNormal);
}

/// Return cosine-weighted sample around the hemisphere.
inline __device__ vec3 sampleLambertian(const vec2& randVec, const vec3& normal)
{
	float phi = 2.0f * sfz::PI() * randVec.x;
	float sinTheta = sqrt(randVec.y);
	float cosTheta = sqrt(1.0f - randVec.y);

	return hemisphereToWorld(phi, cosTheta, sinTheta, normal);
}

// Ray generation kernel
// ------------------------------------------------------------------------------------------------

static __global__ void createSecondaryRaysKernel(CreateSecondaryRaysKernelInput input)
{
	// Calculate surface coordinates
	vec2i loc = vec2i(blockIdx.x * blockDim.x + threadIdx.x,
	                  blockIdx.y * blockDim.y + threadIdx.y);
	if (loc.x >= input.res.x || loc.y >= input.res.y) return;

	uint32_t id = loc.y * input.res.x + loc.x;

	curandState randState = input.randStates[id];
	PathState& pathState = input.pathStates[id];

	vec2i doubleLoc = loc * 2;
	uint32_t random = curand(&randState);

	uint32_t pixel = random % 4;
	vec2i offset(pixel % 2, pixel / 2);
	vec2i doubleLocRandomized = doubleLoc + offset;

	GBufferValue gBufferValue = readGBuffer(input.posTex, input.normalTex, input.albedoTex, input.materialTex, doubleLocRandomized);

	pathState.throughput *= gBufferValue.albedo;

	float r1 = curand_uniform(&randState);
	float r2 = curand_uniform(&randState);

	// Probabilistically pick type of sample using russian roulette with roughness as threshold
	float diffuseOrSpecularSample = curand_uniform(&randState);
	vec3 rayDir;
	if (diffuseOrSpecularSample < gBufferValue.roughness) {
		rayDir = sampleLambertian(vec2(r1, r2), gBufferValue.normal);

		// TODO: Weigh sample according to pdf
		pathState.throughput *= 0.8f;
	}
	else {
		vec3 fromCamera = normalize(gBufferValue.pos - input.camPos);
		vec3 reflection = reflect(fromCamera, gBufferValue.normal);
		float a = gBufferValue.roughness * gBufferValue.roughness;
		rayDir = sampleGgx(vec2(r1, r2), a, reflection, gBufferValue.normal);

		// TODO: Weigh sample according to pdf
		pathState.throughput *= 0.8f;
	}
	RayIn extensionRay;
	extensionRay.setOrigin(gBufferValue.pos);
	extensionRay.setDir(rayDir);
	extensionRay.setMinDist(0.001f);
	extensionRay.setMaxDist(FLT_MAX);

	input.extensionRays[id] = extensionRay;
	input.randStates[id] = randState;
}

void launchCreateSecondaryRaysKernel(const CreateSecondaryRaysKernelInput& input) noexcept
{
	// Calculate number of threads and blocks to run
	dim3 threadsPerBlock(8, 8);
	dim3 numBlocks((input.res.x + threadsPerBlock.x - 1) / threadsPerBlock.x,
	               (input.res.y + threadsPerBlock.y - 1) / threadsPerBlock.y);

	createSecondaryRaysKernel<<<numBlocks, threadsPerBlock>>>(input);

	CHECK_CUDA_ERROR(cudaGetLastError());
	CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

// Path initialization kernel
// ------------------------------------------------------------------------------------------------

static __global__ void initPathStates(vec2i res, PathState* pathStates)
{
	// Calculate surface coordinates
	vec2i loc = vec2i(blockIdx.x * blockDim.x + threadIdx.x,
	                  blockIdx.y * blockDim.y + threadIdx.y);
	if (loc.x >= res.x || loc.y >= res.y) return;

	// Read rayhit from array
	uint32_t id = loc.y * res.x + loc.x;

	PathState& pathState = pathStates[id];
	pathState.pathLength = 0;
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

// Shadow logic kernel
// ------------------------------------------------------------------------------------------------

static __global__ void shadowLogicKernel(ShadowLogicKernelInput input)
{
	// Calculate surface coordinates
	vec2i loc = vec2i(blockIdx.x * blockDim.x + threadIdx.x,
	                  blockIdx.y * blockDim.y + threadIdx.y);
	if (loc.x >= input.res.x || loc.y >= input.res.y) return;

	uint32_t id = loc.y * input.res.x + loc.x;
	vec2i scaledLoc = loc / int32_t(input.resolutionScale);
	vec2i scaledRes = input.res / int32_t(input.resolutionScale);
	uint32_t scaledID = scaledLoc.y * scaledRes.x + scaledLoc.x;

	vec3 color(0.0f);
	if (input.addToSurface) {
		vec4 color4 = readFromSurface(input.surface, loc);
		color = color4.xyz;
	}

	for (int i = 0; i < input.numStaticSphereLights; i++) {
		uint32_t shadowRayID = scaledID * input.numStaticSphereLights + i;
		const bool inLight = input.shadowRayHits[shadowRayID];
		if (inLight) {
			color += input.lightContributions[shadowRayID];
		}
	}
	writeToSurface(input.surface, loc, vec4(color, 1.0f));
}

void launchShadowLogicKernel(const ShadowLogicKernelInput& input) noexcept
{
	// Calculate number of threads and blocks to run
	dim3 threadsPerBlock(8, 8);
	dim3 numBlocks((input.res.x + threadsPerBlock.x - 1) / threadsPerBlock.x,
	               (input.res.y + threadsPerBlock.y - 1) / threadsPerBlock.y);

	shadowLogicKernel<<<numBlocks, threadsPerBlock>>>(input);
	CHECK_CUDA_ERROR(cudaGetLastError());
	CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

} // namespace phe
