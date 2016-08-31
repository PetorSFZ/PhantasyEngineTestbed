#include "CPURayTracerRenderer.hpp"

#include <sfz/gl/IncludeOpenGL.hpp>
#include <sfz/math/Matrix.hpp>
#include <sfz/math/MatrixSupport.hpp>

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

	for (int y = 0; y < mResolution.y; y++) {
		for (int x = 0; x < mResolution.x; x++) {

			// Calculate the position of the pixel in [0,1]
			vec2 texCoord{ float(x) / mResolution.x, 1.0f - float(y) / mResolution.y };

			// Calculate ray direction
			texCoord.y = 1.0f - texCoord.y; // Need to convert coord from GL to D3D clip space
			vec4 clipSpacePos = vec4(2.0f * texCoord - vec2(1.0f), 0.0f, 1.0f);
			vec4 posTmp = invViewProjectionMatrix * clipSpacePos;
			posTmp.xyz /= posTmp.w;

			mTexture[x + mResolution.x * y] = tracePrimaryRays(tri, mMatrices.position, posTmp.xyz);
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
