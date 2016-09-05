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
					this->mTexture[x + rowStartIndex] = tracePrimaryRays(this->mMatrices.position, rayDir);
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
		RaycastResult result = mAabbTree.raycast(origin, dir);

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

vec4 CPURayTracerRenderer::tracePrimaryRays(vec3 origin, vec3 dir) const noexcept
{
	RaycastResult result = mAabbTree.raycast(origin, dir);
	if (!result.intersection.intersected) {
		return vec4{0.0f, 0.0f, 0.0f, 1.0f};
	}
	return vec4(vec3(result.intersection.t * 50.0f), 1.0f);
}

} // namespace phe
