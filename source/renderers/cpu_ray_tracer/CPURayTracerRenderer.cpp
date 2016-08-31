#include "CPURayTracerRenderer.hpp"

#include <sfz/gl/IncludeOpenGL.hpp>
#include <sfz/math/Matrix.hpp>
#include <sfz/math/MatrixSupport.hpp>
#include <sfz/math/MathHelpers.hpp>

namespace sfz {

struct Intersection {
	bool intersected;
	float u, v;
};

struct Triangle {
	vec3 p0, p1, p2;
};

Intersection intersects(const Triangle& triangle, vec3 origin, vec3 dir)
{
	const vec3 p0 = triangle.p0;
	const vec3 p1 = triangle.p1;
	const vec3 p2 = triangle.p2;

	const float EPS = 0.00001f;

	vec3 e1 = p1 - p0;
	vec3 e2 = p2 - p0;
	vec3 q = cross(dir, e2);
	float a = dot(e1, q);
	if (-EPS < a && a < EPS) return { false, 0.0f, 0.0f };

	float f = 1.0f / a;
	vec3 s = origin - p0;
	float u = f * dot(s, q);
	if (u < 0.0f) return { false, 0.0f, 0.0f };

	vec3 r = cross(s, e1);
	float v = f * dot(dir, r);
	if (v < 0.0f || (u + v) > 1.0f) return { false, 0.0f, 0.0f };

	float t = f * dot(e2, r);
	if (t < 0.0f) return { false, 0.0f, 0.0f }; // only trace the ray forward
	return { true, u, v };
}

vec4 tracePrimaryRays(Triangle triangle, vec3 origin, vec3 dir)
{
	if (intersects(triangle, origin, dir).intersected) return vec4{ 1.0f, 0.0f, 0.0f, 1.0f };
	return vec4{ 0.0f, 0.0f, 0.0f, 1.0f};
}

// CPURayTracerRenderer: Constructors & destructors
// ------------------------------------------------------------------------------------------------

CPURayTracerRenderer::CPURayTracerRenderer() noexcept
{

}

// CPURayTracerRenderer: Virtual methods from BaseRenderer interface
// ------------------------------------------------------------------------------------------------

RenderResult CPURayTracerRenderer::render(const DynArray<DrawOp>& operations, const DynArray<PointLight>& pointLights) noexcept
{
	mResult.bindViewportClearColorDepth(vec4(0.0f, 1.0f, 0.0f, 1.0f), 0.0f);
	
	mat4 invProjectionMatrix = inverse(mMatrices.projMatrix);
	
	mat4 viewMatrix = mMatrices.headMatrix * mMatrices.originMatrix;
	mat4 invViewMatrix = inverse(viewMatrix);

	mat4 invViewProjectionMatrix = invViewMatrix * invProjectionMatrix;

	Triangle tri{
		vec3{ -0.5f, 2.0f, -2.0f },
		vec3{ 0.0f, 2.0f, -2.0f },
		vec3{ 0.0f, 2.5f, -2.0f }
	};

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

	vec3 dX = projOnX / float(mResolution.x);
	vec3 dY = projOnY / float(mResolution.y);

	// Lerp the ray direction and trace the rays
	for (int y = 0; y < mResolution.y; y++) {
		vec3 yLerped = topLeftDir.xyz + dY * float(y);
		for (int x = 0; x < mResolution.x; x++) {
			vec3 rayDir{ dX * float(x) + yLerped };
			// rayDir is not normalized
			mTexture[x + mResolution.x * y] = tracePrimaryRays(tri, mMatrices.position, rayDir);
		}
	}

	glBindTexture(GL_TEXTURE_2D, mResult.texture(0));
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, mMaxResolution.x, mMaxResolution.y, 0, GL_RGBA, GL_FLOAT, mTexture.get());
	
	RenderResult tmp;
	tmp.colorTex = mResult.texture(0);
	tmp.colorTexRes = mResult.dimensions();
	tmp.colorTexRenderedRes = mResolution;
	return tmp;
}

// CPURayTracerRenderer: Protected virtual methods from BaseRenderer interface
// ------------------------------------------------------------------------------------------------

void CPURayTracerRenderer::maxResolutionUpdated() noexcept
{
	using gl::FBTextureFiltering;
	using gl::FBTextureFormat;
	using gl::FramebufferBuilder;

	mTexture = std::unique_ptr<vec4[]>{ new vec4[mMaxResolution.x * mMaxResolution.y] };
	mResult = FramebufferBuilder(mMaxResolution)
	          .addTexture(0, FBTextureFormat::RGB_U8, FBTextureFiltering::LINEAR)
	          .build();
}

void CPURayTracerRenderer::resolutionUpdated() noexcept
{

}

} // namespace sfz
