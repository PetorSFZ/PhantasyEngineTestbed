// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#include "kernels/InterpretRayHitKernel.hpp"

#include "CudaHelpers.hpp"

namespace phe {

// Helper functions
// ------------------------------------------------------------------------------------------------

static __device__ void retrieveTriData(const TriangleData* __restrict__ triDatas,
                                       const RayIn& ray, const RayHit& hit,
                                       vec3& pos, vec3& normal, vec2& uv,
                                       uint32_t& materialIndex) noexcept
{
	const TriangleData& data = triDatas[hit.triangleIndex];

	// Retrieving position
	pos = ray.origin() + ray.dir() * hit.t;

	// Interpolating normal
	vec3 n0 = data.n0;
	vec3 n1 = data.n1;
	vec3 n2 = data.n2;
	normal = normalize(n0 + (n1 - n0) * hit.u + (n2 - n0) * hit.v); // TODO: FMA lerp?

	// Interpolating uv coordinate
	vec2 uv0 = data.uv0;
	vec2 uv1 = data.uv1;
	vec2 uv2 = data.uv2;
	uv = uv0 + (uv1 - uv0) * hit.u + (uv2 - uv0) * hit.v; // TODO: FMA lerp?

	// Material index
	materialIndex = data.materialIndex;
}

static __device__ vec4 linearize(vec4 rgbaGamma) noexcept
{
	rgbaGamma.x = powf(rgbaGamma.x, 2.2f);
	rgbaGamma.y = powf(rgbaGamma.y, 2.2f);
	rgbaGamma.z = powf(rgbaGamma.z, 2.2f);
	return rgbaGamma;
}

// InterpretRayHitKernel
// ------------------------------------------------------------------------------------------------

static __global__ void interpretRayHitKernel(InterpretRayHitKernelInput input, 
                                             RayHitInfo* __restrict__ outInfos)
{
	// Calculate ray index in array
	uint32_t rayIdx = blockDim.x * blockIdx.x + threadIdx.x;
	if (rayIdx >= input.numRays) return;
	
	RayHit hit = input.rayHits[rayIdx];
	RayHitInfo info;

	if (hit.triangleIndex != ~0u) {

		RayIn ray = input.rays[rayIdx];

		// Retrieve triangle information
		vec3 pos, normal;
		vec2 uv;
		uint32_t materialIndex;
		retrieveTriData(input.staticTriangleDatas, ray, hit, pos, normal, uv, materialIndex);		
		info.setPosition(pos);
		info.setNormal(normal);

		// Retrieve material
		const Material& mat = input.materials[materialIndex];

		// Albedo
		vec4 albedo = mat.albedoValue();
		if (mat.albedoTexIndex() >= 0) {
			cudaTextureObject_t albedoTex = input.textures[mat.albedoTexIndex()];
			uchar4 albedoTmp = tex2D<uchar4>(albedoTex, uv.x, uv.y);
			albedo = linearize(vec4(albedoTmp.x, albedoTmp.y, albedoTmp.z, albedoTmp.w) / 255.0f);
		}
		info.setAlbedo(albedo.xyz);
		info.setAlpha(albedo.w);

		// Roughness
		float roughness = mat.roughnessValue();
		if (mat.roughnessTexIndex() >= 0) {
			cudaTextureObject_t roughnessTex = input.textures[mat.roughnessTexIndex()];
			uchar1 roughnessTmp = tex2D<uchar1>(roughnessTex, uv.x, uv.y);
			roughness = float(roughnessTmp.x) / 255.0f;
		}
		info.setRoughness(roughness);

		// Metallic
		float metallic = mat.metallicValue();
		if (mat.metallicTexIndex() >= 0) {
			cudaTextureObject_t metallicTex = input.textures[mat.metallicTexIndex()];
			uchar1 metallicTmp = tex2D<uchar1>(metallicTex, uv.x, uv.y);
			metallic = float(metallicTmp.x) / 255.0f;
		}
		info.setMetallic(metallic);

		info.setHitStatus(true);
	}
	else {
		info.setAlpha(1.0f);
		info.setHitStatus(false);
	}

	outInfos[rayIdx] = info;
}

// InterpretRayHitKernel launch function
// ------------------------------------------------------------------------------------------------

void launchInterpretRayHitKernel(const InterpretRayHitKernelInput& input,
                                 RayHitInfo* __restrict__ outInfos,
                                 const cudaDeviceProp& deviceProperties) noexcept
{
	const uint32_t numThreadsPerBlock = 256;
	uint32_t numBlocks = (input.numRays / numThreadsPerBlock) + 1;

	interpretRayHitKernel<<<numBlocks, numThreadsPerBlock>>>(input, outInfos);
	CHECK_CUDA_ERROR(cudaGetLastError());
	CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

} // namespace phe