// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#include "phantasy_engine/renderers/cpu_ray_tracer/CPURayTracerRenderer.hpp"

#include <chrono>

#include <sfz/gl/IncludeOpenGL.hpp>
#include <sfz/math/MathHelpers.hpp>
#include <sfz/math/Matrix.hpp>
#include <sfz/math/MatrixSupport.hpp>

namespace phe {

using namespace sfz;

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

	resultFB.bindViewportClearColorDepth(vec4(0.0f, 1.0f, 0.0f, 1.0f), 0.0f);
	
	mat4 invProjectionMatrix = inverse(mMatrices.projMatrix);
	
	mat4 viewMatrix = mMatrices.headMatrix * mMatrices.originMatrix;
	mat4 invViewMatrix = inverse(viewMatrix);

	mat4 invViewProjectionMatrix = invViewMatrix * invProjectionMatrix;

	// Clip space corners
	vec4 min{ -1.0f, -1.0f, 0.0f, 1.0f };
	vec4 max{ +1.0f, +1.0f, 0.0f, 1.0f };

	// Calculate ray directions of the top left and bottom right corner of the screen respectively
	vec4 topLeftDir = invViewProjectionMatrix * min;
	topLeftDir /= topLeftDir.w;
	vec4 bottomRightDir = invViewProjectionMatrix * max;
	bottomRightDir /= bottomRightDir.w;

	// Get view frustum vectors in world space
	vec3 down = -mMatrices.up;
	vec3 forward = mMatrices.forward;
	vec3 right = normalize(cross(down, forward));

	// Project the lerp step size projected onto the screen's X and Y axes
	vec3 dirDifference = bottomRightDir.xyz - topLeftDir.xyz;
	vec3 projOnX = dot(dirDifference, right) * right;
	vec3 projOnY = dot(dirDifference, down) * down;

	vec3 dX = projOnX / float(mTargetResolution.x);
	vec3 dY = projOnY / float(mTargetResolution.y);

	int nThreads = 10;
	int rowsPerThread = mTargetResolution.y / nThreads;

	// Spawn threads for ray tracing
	for (int i = 0; i < nThreads; i++) {
		mThreads.add(std::thread{ [this, i, topLeftDir, dX, dY, rowsPerThread, nThreads]() {
			// Calculate the which row is the last one that the current thread is responsible for
			int yEnd = i >= nThreads - 1 ? this->mTargetResolution.y : rowsPerThread * (i + 1);
			for (int y = i * rowsPerThread; y < yEnd; y++) {
				// Add the Y-component of the ray direction
				vec3 yLerped = topLeftDir.xyz + dY * float(y);
				int rowStartIndex = this->mTargetResolution.x * y;
				for (int x = 0; x < this->mTargetResolution.x; x++) {
					// Final ray direction
					vec3 rayDir{ dX * float(x) + yLerped};
					rayDir = normalize(rayDir);
					Ray ray(this->mMatrices.position, rayDir);
					this->mTexture[x + rowStartIndex] = tracePrimaryRays(ray);
				}
			}
		} });
	}
	for (std::thread& thread : mThreads) {
		thread.join();
	}
	mThreads.clear();

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
	mAabbTree = AabbTree();

	{
		using time_point = std::chrono::high_resolution_clock::time_point;
		time_point before = std::chrono::high_resolution_clock::now();

		mAabbTree.constructFrom(mStaticScene->opaqueRenderables);

		time_point after = std::chrono::high_resolution_clock::now();
		using FloatSecond = std::chrono::duration<float>;
		float delta = std::chrono::duration_cast<FloatSecond>(after - before).count();
		printf("Time spent building BVH: %.3f seconds\n", delta);
	}

	{
		using time_point = std::chrono::high_resolution_clock::time_point;
		time_point before = std::chrono::high_resolution_clock::now();

		vec3 origin ={-0.25f, 2.25f, 0.0f};
		vec3 dir ={0.0f, 0.0f, -1.0f};
		Ray ray(origin, dir);
		RaycastResult result = mAabbTree.raycast(ray);

		time_point after = std::chrono::high_resolution_clock::now();
		using FloatSecond = std::chrono::duration<float>;
		float delta = std::chrono::duration_cast<FloatSecond>(after - before).count();
		printf("Time to find ray intersection: %f seconds\n", delta);
		if (result.intersection.intersected) {
			printf("Test ray intersected at t=%f, %s\n", result.intersection.t, toString(origin + result.intersection.t * dir).str);
		}
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

vec4 CPURayTracerRenderer::traceSecondaryRays(const Ray& ray) const noexcept
{
	return vec4(0.0f);
}

// Reflect u in v, v must be normalized
vec3 reflect(const vec3 u, const vec3 v) noexcept
{
	return u - 2 * dot(u, v) * v;
}

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

	vec2i texCoord = vec2i(std::round(scaledUV.x), std::round(scaledUV.y));

	return image.getPixelPtr(texCoord);
}

// PBR shading functions
// ------------------------------------------------------------------------------------------------

// References used:
// https://de45xmedrsdbp.cloudfront.net/Resources/files/2013SiggraphPresentationsNotes-26915738.pdf
// http://blog.selfshadow.com/publications/s2016-shading-course/
// http://www.codinglabs.net/article_physically_based_rendering_cook_torrance.aspx
// http://graphicrants.blogspot.se/2013/08/specular-brdf-reference.html

// Normal distribution function, GGX/Trowbridge-Reitz
// a = roughness^2, UE4 parameterization
// dot(n,h) term should be clamped to 0 if negative
float ggx(float nDotH, float a)
{
	float a2 = a * a;
	float div = sfz::PI() * pow(nDotH * nDotH * (a2 - 1.0f) + 1.0f, 2);
	return a2 / div;
}

// Schlick's model adjusted to fit Smith's method
// k = a/2, where a = roughness^2, however, for analytical light sources (non image based)
// roughness is first remapped to roughness = (roughnessOrg + 1) / 2.
// Essentially, for analytical light sources:
// k = (roughness + 1)^2 / 8
// For image based lighting:
// k = roughness^2 / 2
float geometricSchlick(float nDotL, float nDotV, float k)
{
	float g1 = nDotL / (nDotL * (1.0f - k) + k);
	float g2 = nDotV / (nDotV * (1.0f - k) + k);
	return g1 * g2;
}

// Schlick's approximation. F0 should typically be 0.04 for dielectrics
vec3 fresnelSchlick(float nDotL, vec3 f0)
{
	return f0 + (vec3(1.0f) - f0) * sfz::clamp(std::pow(1.0f - nDotL, 5.0f), 0.0f, 1.0f);
}

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

	const Renderable& renderable = *result.rawGeometryTriangle.renderable;
	const Material& material = result.rawGeometryTriangle.component->material;
	const DynArray<RawImage>& images = renderable.images;

	vec3 albedoColour = material.albedoValue;

	if (material.albedoIndex != UINT32_MAX) {
		const RawImage& albedoImage = images[material.albedoIndex];
		if (albedoImage.bytesPerPixel == 3 ||
		    albedoImage.bytesPerPixel == 4) {
			Vector<uint8_t, 3> intColor = Vector<uint8_t, 3>(sampleImage(albedoImage, textureUV));
			albedoColour = vec3(intColor) / 255.0f;
		}
	}
	// Linearize
	albedoColour.x = std::pow(albedoColour.x, 2.2);
	albedoColour.y = std::pow(albedoColour.y, 2.2);
	albedoColour.z = std::pow(albedoColour.z, 2.2);

	float roughness = material.roughnessValue;
	float metallic = material.metallicValue;

	if (material.roughnessIndex != UINT32_MAX) {
		const RawImage& image = images[material.roughnessIndex];
		uint8_t intColour = sampleImage(image, textureUV)[0];
		roughness = intColour / 255.0f;
	}
	if (material.metallicIndex != UINT32_MAX) {
		const RawImage& image = images[material.metallicIndex];
		uint8_t intColour = sampleImage(image, textureUV)[0];
		metallic = intColour / 255.0f;
	}

	vec3 pos = ray.origin + ray.dir * t;
	vec3 reflectionDir = reflect(ray.dir, normal);

	vec3 colour = vec3(0.0f);

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
		vec3 diffuse = albedoColour / sfz::PI();

		// Cook-Torrance specular
		// Normal distribution function
		float nDotH = std::max(sfz::dot(normal, h), 0.0f); // max() should be superfluous here
		float ctD = ggx(nDotH, roughness * roughness);

		// Geometric self-shadowing term
		float k = pow(roughness + 1.0, 2) / 8.0;
		float ctG = geometricSchlick(nDotL, nDotV, k);

		// Fresnel function
		// Assume all dielectrics have a f0 of 0.04, for metals we assume f0 == albedo
		vec3 f0 = sfz::lerp(vec3(0.04f), albedoColour, metallic);
		vec3 ctF = fresnelSchlick(nDotL, f0);

		// Calculate final Cook-Torrance specular value
		vec3 specular = ctD * ctF * ctG / (4.0f * nDotL * nDotV);

		// Calculates light strength
		float fallofNumerator = pow(sfz::clamp(1.0f - std::pow(toLightDist / light.range, 4.0f), 0.0f, 1.0f), 2);
		float fallofDenominator = (toLightDist * toLightDist + 1.0);
		float falloff = fallofNumerator / fallofDenominator;
		vec3 lighting = falloff * light.strength;

		colour += (diffuse + specular) * lighting * nDotL;
	}

	//vec4 bouncyBounce = traceSecondaryRays({ pos, reflectionDir });

	return vec4(colour, 1.0f);
}

} // namespace phe
