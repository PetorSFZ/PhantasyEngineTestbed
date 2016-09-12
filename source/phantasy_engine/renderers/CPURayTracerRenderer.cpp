// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#include "phantasy_engine/renderers/CPURayTracerRenderer.hpp"

#include <chrono>
#include <future>

#include <sfz/gl/IncludeOpenGL.hpp>
#include <sfz/math/MathHelpers.hpp>
#include <sfz/math/Matrix.hpp>
#include <sfz/math/MatrixSupport.hpp>

#include "phantasy_engine/RayTracerCommon.hpp"

namespace phe {

using namespace sfz;

// Statics
// ------------------------------------------------------------------------------------------------



// CPURayTracerRenderer: Constructors & destructors
// ------------------------------------------------------------------------------------------------

CPURayTracerRenderer::CPURayTracerRenderer() noexcept
{
	
}

// CPURayTracerRenderer: Virtual methods from BaseRenderer interface
// ------------------------------------------------------------------------------------------------

RenderResult CPURayTracerRenderer::render(Framebuffer& resultFB) noexcept
{
	using time_point = std::chrono::high_resolution_clock::time_point;
	time_point before = std::chrono::high_resolution_clock::now();

	// Calculate camera def in order to generate first rays
	vec2 resultRes = vec2(mTargetResolution);
	CameraDef cam = generateCameraDef(mMatrices.position, mMatrices.forward, mMatrices.up,
	                                  mMatrices.vertFovRad, resultRes);

	int nThreads = 10;
	int rowsPerThread = mTargetResolution.y / nThreads;

	// Spawn threads for ray tracing
	for (int i = 0; i < nThreads; i++) {
		mThreads.add(std::thread{ [this, i, resultRes, cam, rowsPerThread, nThreads]() {
			
			// Calculate the which row is the last one that the current thread is responsible for
			int yEnd = i >= nThreads - 1 ? this->mTargetResolution.y : rowsPerThread * (i + 1);
			
			for (int y = i * rowsPerThread; y < yEnd; y++) {
				int rowStartIndex = this->mTargetResolution.x * y;

				for (int x = 0; x < this->mTargetResolution.x; x++) {
					
					// Calculate ray dir
					vec2 loc = vec2(float(x), float(y));
					vec2 locNormalized = loc / resultRes; // [0, 1]
					vec2 centerOffsCoord = locNormalized * 2.0f - vec2(1.0f); // [-1.0, 1.0]
					vec3 rayDir = normalize(cam.dir + centerOffsCoord.x * cam.dX + centerOffsCoord.y * cam.dY);
					Ray ray(cam.origin, rayDir);

					BVHNode* nodes = this->mBVH.nodes.data();
					TriangleVertices* triangles = this->mBVH.triangles.data();
					TriangleData* datas = this->mBVH.triangleDatas.data();
					
					// Ray cast against BVH
					RayCastResult hit = castRay(nodes, triangles, ray);
					if (hit.triangleIndex == ~0u) {
						this->mTexture[x + rowStartIndex] = vec4(0.0f);
						continue;
					}

					HitInfo info = interpretHit(datas, hit, ray);

					this->mTexture[x + rowStartIndex] = vec4(info.normal, 1.0);


					// Trace ray ODL
					//
					//this->mTexture[x + rowStartIndex] = tracePrimaryRays(ray);
				}
			}
		} });
	}

	// Wait for all threads to finish
	for (std::thread& thread : mThreads) {
		thread.join();
	}
	mThreads.clear();

	// Transfer result to resultFB
	glBindTexture(GL_TEXTURE_2D, resultFB.texture(0));
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, mTargetResolution.x, mTargetResolution.y, 0, GL_RGBA, GL_FLOAT, mTexture.data());
	
	RenderResult tmp;
	tmp.renderedRes = mTargetResolution;

	time_point after = std::chrono::high_resolution_clock::now();
	using FloatSecond = std::chrono::duration<float>;
	float delta = std::chrono::duration_cast<FloatSecond>(after - before).count();
	printf("Render time (CPU tracer): %.3f seconds\n", delta);

	return tmp;
}

// CPURayTracerRenderer: Protected virtual methods from BaseRenderer interface
// ------------------------------------------------------------------------------------------------

void CPURayTracerRenderer::staticSceneChanged() noexcept
{
	{
		using time_point = std::chrono::high_resolution_clock::time_point;
		time_point before = std::chrono::high_resolution_clock::now();

		mBVH.buildStaticFrom(*mStaticScene.get());

		time_point after = std::chrono::high_resolution_clock::now();
		using FloatSecond = std::chrono::duration<float>;
		float delta = std::chrono::duration_cast<FloatSecond>(after - before).count();
		printf("CPU Ray Tracer: Time spent building BVH: %.3f seconds\n", delta);
	}
}

void CPURayTracerRenderer::targetResolutionUpdated() noexcept
{
	using gl::FBTextureFiltering;
	using gl::FBTextureFormat;
	using gl::FramebufferBuilder;

	mTexture.ensureCapacity(mTargetResolution.x * mTargetResolution.y);
	mTexture.setSize(mTargetResolution.x * mTargetResolution.y);
}

// CPURayTracerRenderer: Private methods
// ------------------------------------------------------------------------------------------------

const uint8_t* CPURayTracerRenderer::sampleImage(const RawImage& image, const vec2 uv) const noexcept
{
	vec2 texDim = vec2(image.dim);

	// Convert from triangle UV to texture coordinates
	vec2 scaledUV = uv * texDim;
	scaledUV.x = std::fmod(scaledUV.x, texDim.x);
	scaledUV.y = std::fmod(scaledUV.y, texDim.y);
	scaledUV += texDim;
	scaledUV.x = std::fmod(scaledUV.x, texDim.x);
	scaledUV.y = std::fmod(scaledUV.y, texDim.y);

	vec2i texCoord = vec2i(std::floor(scaledUV.x), std::floor(scaledUV.y));

	sfz_assert_debug(texCoord.x >= 0);
	sfz_assert_debug(texCoord.y >= 0);
	sfz_assert_debug(texCoord.x < image.dim.x);
	sfz_assert_debug(texCoord.y < image.dim.y);

	return image.getPixelPtr(texCoord);
}

// PBR shading functions
// ------------------------------------------------------------------------------------------------
/*


vec4 CPURayTracerRenderer::tracePrimaryRays(const Ray& ray) const noexcept
{
	RaycastResult result = mAabbTree.raycast(ray);
	if (!result.intersection.intersected) {
		return vec4{0.0f, 0.0f, 0.0f, 1.0f};
	}

	vec3 n0 = result.rawGeometryTriangle.v0->normal;
	vec3 n1 = result.rawGeometryTriangle.v1->normal;
	vec3 n2 = result.rawGeometryTriangle.v2->normal;

	vec2 uv0 = result.rawGeometryTriangle.v0->uv;
	vec2 uv1 = result.rawGeometryTriangle.v1->uv;
	vec2 uv2 = result.rawGeometryTriangle.v2->uv;

	float t = result.intersection.t;

	float u = result.intersection.u;
	float v = result.intersection.v;

	vec3 normal = normalize(n0 + (n1 - n0) * u + (n2 - n0) * v);
	vec2 textureUV = uv0 + (uv1 - uv0) * u + (uv2 - uv0) * v;

	const Material& material = result.rawGeometryTriangle.component->material;
	const DynArray<RawImage>& images = mStaticScene->images;

	vec3 albedoColor = material.albedoValue;

	if (material.albedoIndex != UINT32_MAX) {
		const RawImage& albedoImage = images[material.albedoIndex];
		if (albedoImage.bytesPerPixel == 3 ||
		    albedoImage.bytesPerPixel == 4) {
			Vector<uint8_t, 3> intColor = Vector<uint8_t, 3>(sampleImage(albedoImage, textureUV));
			albedoColor = vec3(intColor) / 255.0f;
		}
	}
	// Linearize
	albedoColor.x = std::pow(albedoColor.x, 2.2);
	albedoColor.y = std::pow(albedoColor.y, 2.2);
	albedoColor.z = std::pow(albedoColor.z, 2.2);

	float roughness = material.roughnessValue;
	float metallic = material.metallicValue;

	if (material.roughnessIndex != UINT32_MAX) {
		const RawImage& image = images[material.roughnessIndex];
		uint8_t intColor = sampleImage(image, textureUV)[0];
		roughness = intColor / 255.0f;
	}
	if (material.metallicIndex != UINT32_MAX) {
		const RawImage& image = images[material.metallicIndex];
		uint8_t intColor = sampleImage(image, textureUV)[0];
		metallic = intColor / 255.0f;
	}

	vec3 pos = ray.origin + ray.dir * t;
	vec3 reflectionDir = reflect(ray.dir, normal);

	vec3 color = vec3(0.0f);

	for (PointLight& light : mStaticScene.get()->pointLights) {
		vec3 toLight = light.pos - pos;
		float toLightDist = length(toLight);
		vec3 l = toLight / toLightDist;
		vec3 v = normalize(-ray.dir);
		vec3 h = normalize(l + v);

		float nDotL = dot(normal, l);
		if (nDotL <= 0.0f) {
			continue;
		}
		
		float nDotV = dot(normal, v);

		nDotV = std::max(0.001f, nDotV);

		// Lambert diffuse
		vec3 diffuse = albedoColor / sfz::PI();

		// Cook-Torrance specular
		// Normal distribution function
		float nDotH = std::max(sfz::dot(normal, h), 0.0f); // max() should be superfluous here
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

	//vec4 bouncyBounce = traceSecondaryRays({ pos, reflectionDir });

	return vec4(color, 1.0f);
}*/

} // namespace phe