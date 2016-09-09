// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#include "phantasy_engine/renderers/cpu_ray_tracer/CPURayTracerRenderer.hpp"

#include <chrono>

#include <sfz/gl/IncludeOpenGL.hpp>
#include <sfz/math/MathHelpers.hpp>
#include <sfz/math/Matrix.hpp>
#include <sfz/math/MatrixSupport.hpp>

#include "phantasy_engine/renderers/cpu_ray_tracer/RayTracerCommon.hpp"

namespace phe {

using namespace sfz;

// Statics
// ------------------------------------------------------------------------------------------------

struct AABBHit final {
	bool hit;
	float t;
};

static AABBHit intersects(const Ray& ray, const vec3& min, const vec3& max) noexcept
{
	vec3 t1 = (min - ray.origin) * ray.invDir;
	vec3 t2 = (max - ray.origin) * ray.invDir;

	float tmin = sfz::maxElement(sfz::min(t1, t2));
	float tmax = sfz::minElement(sfz::max(t1, t2));

	AABBHit tmp;
	tmp.hit = tmax >= tmin;
	tmp.t = tmin;
	return tmp;
}

// DONT CHANGE STUPIDS
struct TriangleHit final {
	bool hit;
	float t, u, v;
};

// See page 750 in Real-Time Rendering 3
inline TriangleHit intersects(const TriangleVertices& tri, vec3 origin, vec3 dir) noexcept
{
	const float EPS = 0.00001f;
	vec3 p0 = vec3(tri.v0);
	vec3 p1 = vec3(tri.v1);
	vec3 p2 = vec3(tri.v2);

	vec3 e1 = p1 - p0;
	vec3 e2 = p2 - p0;
	vec3 q = cross(dir, e2);
	float a = dot(e1, q);
	if (-EPS < a && a < EPS) return {false, 0.0f, 0.0f, 0.0f};

	// Backface culling here?
	// dot(cross(e1, e2), dir) <= 0.0 ??

	float f = 1.0f / a;
	vec3 s = origin - p0;
	float u = f * dot(s, q);
	if (u < 0.0f) return {false, 0.0f, 0.0f, 0.0f};

	vec3 r = cross(s, e1);
	float v = f * dot(dir, r);
	if (v < 0.0f || (u + v) > 1.0f) return {false, 0.0f, 0.0f, 0.0f};

	float t = f * dot(e2, r);
	return {true, u, v, t};
}

struct RayCastResult final {
	uint32_t index = ~0u;
	
	// Amount to go in ray direction
	float t = FLT_MAX;

	// Hit position on triangle
	float u = FLT_MAX;
	float v = FLT_MAX;
};

static RayCastResult castRay(BVHNode* nodes, TriangleVertices* triangles, const Ray& ray, float tMin = 0.0001f, float tMax = FLT_MAX) noexcept
{
	// Create local stack
	const uint32_t STACK_MAX_SIZE = 196u;
	uint32_t stack[STACK_MAX_SIZE];
	for (uint32_t& s : stack) s = ~0u;
	
	// Place initial node on stack
	stack[0] = 0u;
	uint32_t stackSize = 1u;

	// Traverse through the tree
	RayCastResult closest;
	while (stackSize > 0u) {
		
		// Retrieve node on top of stack
		stackSize -= 1;
		BVHNode node = nodes[stack[stackSize]];

		// Node is a leaf
		if (isLeaf(node)) {
			uint32_t triCount = numTriangles(node);
			TriangleVertices* triList = triangles + triangleListIndex(node);

			for (uint32_t i = 0; i < triCount; i++) {
				TriangleVertices& tri = triList[i];
				TriangleHit hit = intersects(tri, ray.origin, ray.dir);

				if (hit.hit && hit.t < closest.t && tMin <= hit.t && hit.t <= tMax) {
					closest.index = (triList - triangles) + i;
					closest.t = hit.t;
					closest.u = hit.u;
					closest.v = hit.v;

					// Possible early exit
					// if (hit.t == tMin) return closest;
				}
			
			}

		}

		// Node is a not leaf
		else {
			AABBHit hit = intersects(ray, aabbMin(node), aabbMax(node));
			if (hit.hit && hit.t <= closest.t && hit.t <= tMax) {
				
				stack[stackSize] = leftChildIndex(node);
				stack[stackSize + 1] = rightChildIndex(node);
				stackSize += 2;
			}
		}

	}

	return closest;
}

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
					
					// Ray cast against BVH
					RayCastResult hit = castRay(nodes, triangles, ray);
					if (hit.index == ~0u) {
						this->mTexture[x + rowStartIndex] = vec4(0.0f);
						continue;
					}

					// Draw depth
					this->mTexture[x + rowStartIndex] = vec4(vec3(hit.t / 10.0f), 1.0);


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
	mAabbTree = AabbTree();

	{
		using time_point = std::chrono::high_resolution_clock::time_point;
		time_point before = std::chrono::high_resolution_clock::now();

		//mAabbTree.constructFrom(mStaticScene->opaqueRenderables);
		mBVH.buildStaticFrom(*mStaticScene.get());

		time_point after = std::chrono::high_resolution_clock::now();
		using FloatSecond = std::chrono::duration<float>;
		float delta = std::chrono::duration_cast<FloatSecond>(after - before).count();
		printf("Time spent building BVH: %.3f seconds\n", delta);
	}

	/*{
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
	}*/
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

	vec2i texCoord = vec2i(std::floor(scaledUV.x), std::floor(scaledUV.y));

	sfz_assert_debug(texCoord.x >= 0);
	sfz_assert_debug(texCoord.y >= 0);
	sfz_assert_debug(texCoord.x < image.dim.x);
	sfz_assert_debug(texCoord.y < image.dim.y);

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
}

} // namespace phe
