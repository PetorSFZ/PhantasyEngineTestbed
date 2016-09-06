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

	vec3 albedoColor = material.albedoValue;

	if (material.albedoIndex != UINT32_MAX) {
		const RawImage& albedoImage = images[material.albedoIndex];

		vec2 texDim = vec2(albedoImage.dim);

		// Convert from triangle UV to texture coordinates
		vec2 scaledUV = textureUV * texDim;
		scaledUV.x = std::fmod(scaledUV.x, texDim.x);
		scaledUV.y = std::fmod(scaledUV.y, texDim.y);
		scaledUV += texDim;
		scaledUV.x = std::fmod(scaledUV.x, texDim.x);
		scaledUV.y = std::fmod(scaledUV.y, texDim.y);

		vec2i texCoord = vec2i(std::round(scaledUV.x), std::round(scaledUV.y));

		if (albedoImage.bytesPerPixel == 3 ||
		    albedoImage.bytesPerPixel == 4) {
			Vector<uint8_t, 3> intColor = Vector<uint8_t, 3>(albedoImage.getPixelPtr(texCoord));
			albedoColor = vec3(intColor) / 255.0f;
		}
	}


	vec3 pos = ray.origin + ray.dir * t;
	vec3 reflectionDir = reflect(ray.dir, normal);

	vec3 color = vec3(0.0f);

	//for (PointLight light : mStaticScene.get()->pointLights) {
	PointLight light = mStaticScene.get()->pointLights[2];
	{
		vec3 toLight = light.pos - pos;
		float toLightDist = length(toLight);
		vec3 l = toLight / toLightDist;
		vec3 v = normalize(-pos);
		vec3 h = normalize(l + v);

		float nDotL = dot(normal, l);
		if (nDotL <= 0.0f) {

		}
		else {
			float nDotV = dot(normal, v);

			nDotV = std::max(0.001f, nDotV);

			vec3 diffuse = albedoColor / sfz::PI();

			vec3 specular = vec3(0.0f);

			// Calculates light strength
			float fallofNumerator = pow(sfz::clamp(1.0f - pow(toLightDist / light.range, 4), 0.0f, 1.0f), 2);
			float fallofDenominator = (toLightDist * toLightDist + 1.0);
			float falloff = fallofNumerator / fallofDenominator;
			vec3 lighting = falloff * light.strength;

			color += (diffuse + specular) * lighting * nDotV;
		}
	}

	vec4 bouncyBounce = traceSecondaryRays({ pos, reflectionDir });

	return vec4(color, 1.0f) + bouncyBounce;
}

} // namespace phe
