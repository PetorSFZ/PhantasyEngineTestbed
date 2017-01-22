// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#include "phantasy_engine/deferred_renderer/DeferredRenderer.hpp"

#include <sfz/gl/IncludeOpenGL.hpp>
#include <sfz/gl/Program.hpp>
#include <sfz/math/ProjectionMatrices.hpp>
#include <sfz/memory/New.hpp>
#include <sfz/strings/StackString.hpp>
#include <sfz/util/IO.hpp>

#include "phantasy_engine/deferred_renderer/GLModel.hpp"
#include "phantasy_engine/deferred_renderer/GLTexture.hpp"
#include "phantasy_engine/deferred_renderer/ShadowCubeMap.hpp"
#include "phantasy_engine/deferred_renderer/SSBO.hpp"
#include "phantasy_engine/level/SphereLight.hpp"
#include "phantasy_engine/rendering/FullscreenTriangle.hpp"

namespace phe {

using namespace sfz;
using gl::Framebuffer;
using gl::Program;

// Statics
// ------------------------------------------------------------------------------------------------

static const uint32_t GBUFFER_NORMAL = 0;
static const uint32_t GBUFFER_ALBEDO = 1;
static const uint32_t GBUFFER_MATERIAL = 2;
static const uint32_t GBUFFER_VELOCITY = 3;

// DeferredRendererImpl
// ------------------------------------------------------------------------------------------------

class DeferredRendererImpl final {
public:
	// Shaders
	Program shadowMapGenShader, gbufferGenShader, shadingShader;
	
	// Framebuffers
	Framebuffer resultFB;
	Framebuffer gbuffer;

	// Allmighty fullscreen triangle
	FullscreenTriangle fullscreenTriangle;

	// Textures & materials
	DynArray<GLTexture> textures;
	SSBO texturesSSBO;
	SSBO materialSSBO;

	// Static scene
	DynArray<GLModel> staticGLModels;
	DynArray<SphereLight> staticSphereLights;
	DynArray<ShadowCubeMap> staticShadowMaps;

	// Dynamic scene
	DynArray<GLModel> dynamicGLModels;

	~DeferredRendererImpl() noexcept
	{

	}
};

// DeferredRenderer: Constructors & destructors
// ------------------------------------------------------------------------------------------------

DeferredRenderer::DeferredRenderer() noexcept
{
	this->mImpl = sfzNewDefault<DeferredRendererImpl>();

	StackString128 shadersPath;
	shadersPath.printf("%sresources/shaders/deferred_renderer/", basePath());

	mImpl->shadowMapGenShader = Program::fromFile(shadersPath.str, "shadow_map_gen.vert",
	                                              "shadow_map_gen.geom", "shadow_map_gen.frag",
	[](uint32_t shaderProgram) {
		glBindAttribLocation(shaderProgram, 0, "inPosition");
	});

	mImpl->gbufferGenShader = Program::fromFile(shadersPath.str, "gbuffer_gen.vert", "gbuffer_gen.frag",
	[](uint32_t shaderProgram) {
		glBindAttribLocation(shaderProgram, 0, "inPosition");
		glBindAttribLocation(shaderProgram, 1, "inNormal");
		glBindAttribLocation(shaderProgram, 2, "inUV");
		glBindAttribLocation(shaderProgram, 3, "inMaterialId");
	});

	mImpl->shadingShader = Program::postProcessFromFile(shadersPath.str, "shading.frag");
}

DeferredRenderer::DeferredRenderer(DeferredRenderer&& other) noexcept
{
	std::swap(this->mImpl, other.mImpl);
}

DeferredRenderer& DeferredRenderer::operator= (DeferredRenderer&& other) noexcept
{
	std::swap(this->mImpl, other.mImpl);
	return *this;
}

DeferredRenderer::~DeferredRenderer() noexcept
{
	sfzDeleteDefault(mImpl);
}

// DeferredRenderer: Virtual methods from BaseRenderer interface
// ------------------------------------------------------------------------------------------------

void DeferredRenderer::setMaterialsAndTextures(const DynArray<Material>& materials,
                                               const DynArray<RawImage>& textures) noexcept
{
	// Destroy old values
	mImpl->textures.clear();
	mImpl->texturesSSBO.destroy();
	mImpl->materialSSBO.destroy();

	// Allocate SSBO memory and upload compact materials
	uint32_t numMaterialBytes = materials.size() * sizeof(Material);
	mImpl->materialSSBO.create(numMaterialBytes);
	mImpl->materialSSBO.uploadData(materials.data(), numMaterialBytes);

	// Create GLTextures
	DynArray<uint64_t> tmpBindlessTextureHandles;
	tmpBindlessTextureHandles.setCapacity(textures.size());
	mImpl->textures.setCapacity(textures.size());
	for (const RawImage& img : textures) {
		mImpl->textures.add(GLTexture(img));
		tmpBindlessTextureHandles.add(mImpl->textures.last().bindlessHandle());
	}

	// Allocate SSBO memory and upload bindless texture handles
	uint32_t numBindlessTextureHandleBytes = tmpBindlessTextureHandles.size() * sizeof(uint64_t);
	mImpl->texturesSSBO.create(numBindlessTextureHandleBytes);
	mImpl->texturesSSBO.uploadData(tmpBindlessTextureHandles.data(), numBindlessTextureHandleBytes);
}

void DeferredRenderer::addTexture(const RawImage& texture) noexcept
{
	sfz::error("DeferredRenderer: addTexture() not implemented");
}

void DeferredRenderer::addMaterial(const Material& material) noexcept
{
	sfz::error("DeferredRenderer: addMaterial() not implemented");
}

void DeferredRenderer::setStaticScene(const StaticScene& staticScene) noexcept
{
	DynArray<GLModel>& glModels = mImpl->staticGLModels;
	glModels.clear();

	for (const RawMesh& mesh : staticScene.meshes) {
		glModels.add(GLModel(mesh));
	}

	mImpl->staticSphereLights = staticScene.sphereLights;

	// Static shadow maps
	mImpl->staticShadowMaps.clear();
	for (const SphereLight& light : mImpl->staticSphereLights) {
		ShadowCubeMap shadowMap(2096u);

		// Note: We don't use reverse-z here, unecessary since we overwrite the depth anyway

		glDisable(GL_BLEND);
		glEnable(GL_CULL_FACE);
		glEnable(GL_DEPTH_TEST);
		glDepthFunc(GL_LESS);

		glBindFramebuffer(GL_FRAMEBUFFER, shadowMap.fbo());
		glViewport(0, 0, shadowMap.resolution().x, shadowMap.resolution().y);
		glClearDepth(1.0f);
		glClear(GL_DEPTH_BUFFER_BIT);
		
		mImpl->shadowMapGenShader.useProgram();

		// Model matrix is identity since static scene is defined in world space
		gl::setUniform(mImpl->shadowMapGenShader, "uModelMatrix", mat4::identity());

		// View and projection matrices for each face
		const mat4 projMatrix = mat4::scaling3(1.0f, -1.0f, 1.0f) * sfz::perspectiveProjectionVkD3d(90.0f, 1.0f, 0.01f, light.range);
		mat4 viewProjMatrices[6];
		viewProjMatrices[0] = projMatrix * sfz::viewMatrixGL(light.pos, vec3(1.0f, 0.0f, 0.0f), vec3(0.0f, -1.0f, 0.0f)); // right
		viewProjMatrices[1] = projMatrix * sfz::viewMatrixGL(light.pos, vec3(-1.0f, 0.0f, 0.0f), vec3(0.0f, -1.0f, 0.0f)); // left
		viewProjMatrices[2] = projMatrix * sfz::viewMatrixGL(light.pos, vec3(0.0f, 1.0f, 0.0f), vec3(0.0f, 0.0f, 1.0f)); // top
		viewProjMatrices[3] = projMatrix * sfz::viewMatrixGL(light.pos, vec3(0.0f, -1.0f, 0.0f), vec3(0.0f, 0.0f, -1.0f)); // bottom
		viewProjMatrices[4] = projMatrix * sfz::viewMatrixGL(light.pos, vec3(0.0f, 0.0f, 1.0f), vec3(0.0f, -1.0f, 0.0f)); // near
		viewProjMatrices[5] = projMatrix * sfz::viewMatrixGL(light.pos, vec3(0.0f, 0.0f, -1.0f), vec3(0.0f, -1.0f, 0.0f)); // far
		gl::setUniform(mImpl->shadowMapGenShader, "uViewProjMatrices", viewProjMatrices, 6);

		// Light information
		gl::setUniform(mImpl->shadowMapGenShader, "uLightPosWS", light.pos);
		gl::setUniform(mImpl->shadowMapGenShader, "uLightRange", light.range);

		for (const GLModel& model : mImpl->staticGLModels) {
			model.draw();
		}

		mImpl->staticShadowMaps.add(std::move(shadowMap));
	}
}
	
void DeferredRenderer::setDynamicMeshes(const DynArray<RawMesh>& meshes) noexcept
{
	DynArray<GLModel>& glModels = mImpl->dynamicGLModels;
	glModels.clear();

	for (const RawMesh& mesh : meshes) {
		glModels.add(GLModel(mesh));
	}
}

void DeferredRenderer::addDynamicMesh(const RawMesh& mesh) noexcept
{
	sfz::error("DeferredRenderer: addDynamicMeshes() not implemented");
}

RenderResult DeferredRenderer::render(const RenderComponent* renderComponents, uint32_t numComponents,
                                      const DynArray<SphereLight>& lights) noexcept
{
	auto& gbufferGenShader = mImpl->gbufferGenShader;
	auto& shadingShader = mImpl->shadingShader;
	auto& resultFb = mImpl->resultFB;
	auto& gbuffer = mImpl->gbuffer;

	const mat4 viewMatrix = mCamera.viewMatrix();
	const mat4 invViewMatrix = inverse(viewMatrix);
	const mat4 projMatrix = mCamera.projMatrix(mTargetResolution);
	const mat4 invProjMatrix = inverse(projMatrix);

	// GBuffer generation
	// --------------------------------------------------------------------------------------------

	glDisable(GL_BLEND);
	glEnable(GL_CULL_FACE);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_GREATER); // reversed-z

	gbuffer.bindViewportClearColorDepth(vec2i(0), mTargetResolution, vec4(0.0f), 0.0f);
	gbufferGenShader.useProgram();

	const mat4 modelMatrix = mat4::identity();
	gl::setUniform(gbufferGenShader, "uProjMatrix", projMatrix);
	gl::setUniform(gbufferGenShader, "uViewMatrix", viewMatrix);

	// Bind SSBOs
	mImpl->materialSSBO.bind(0);
	mImpl->texturesSSBO.bind(1);

	const int modelMatrixLoc = glGetUniformLocation(gbufferGenShader.handle(), "uModelMatrix");
	const int normalMatrixLoc = glGetUniformLocation(gbufferGenShader.handle(), "uNormalMatrix");
	const int worldVelocityLoc = glGetUniformLocation(gbufferGenShader.handle(), "uWorldVelocity");

	// For the static scene the model matrix should be identity
	gl::setUniform(modelMatrixLoc, mat4::identity());
	gl::setUniform(normalMatrixLoc, inverse(transpose(viewMatrix)));

	// Set velocity of static geometry to 0
	gl::setUniform(worldVelocityLoc, vec3(0.0f));
	for (const GLModel& model : mImpl->staticGLModels) {
		model.draw();
	}

	for (uint32_t i = 0; i < numComponents; i++) {
		const RenderComponent& obj = renderComponents[i];

		gl::setUniform(modelMatrixLoc, obj.transform);
		gl::setUniform(normalMatrixLoc, inverse(transpose(viewMatrix * obj.transform)));
		gl::setUniform(worldVelocityLoc, obj.velocity);
		const GLModel& model = mImpl->dynamicGLModels[obj.meshIndex];
		model.draw();
	}

	// Shading
	// --------------------------------------------------------------------------------------------

	glDisable(GL_BLEND);
	glEnable(GL_CULL_FACE);
	glDisable(GL_DEPTH_TEST);

	glEnable(GL_BLEND);
	glBlendEquation(GL_FUNC_ADD);
	glBlendFunc(GL_ONE, GL_ONE);

	resultFb.bindViewportClearColorDepth(vec2i(0.0), mTargetResolution, vec4(0.0f), 0.0f);
	shadingShader.useProgram();

	gl::setUniform(shadingShader, "uInvProjMatrix", invProjMatrix);
	gl::setUniform(shadingShader, "uInvViewMatrix", invViewMatrix);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, gbuffer.depthTexture());
	gl::setUniform(shadingShader, "uDepthTexture", 0);

	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, gbuffer.texture(GBUFFER_NORMAL));
	gl::setUniform(shadingShader, "uNormalTexture", 1);

	glActiveTexture(GL_TEXTURE2);
	glBindTexture(GL_TEXTURE_2D, gbuffer.texture(GBUFFER_ALBEDO));
	gl::setUniform(shadingShader, "uAlbedoTexture", 2);

	glActiveTexture(GL_TEXTURE3);
	glBindTexture(GL_TEXTURE_2D, gbuffer.texture(GBUFFER_MATERIAL));
	gl::setUniform(shadingShader, "uMaterialTexture", 3);

	const int lightPosVSLoc = glGetUniformLocation(shadingShader.handle(), "uLightPosVS");
	const int lightPosWSLoc = glGetUniformLocation(shadingShader.handle(), "uLightPosWS");
	const int lightStrengthLoc = glGetUniformLocation(shadingShader.handle(), "uLightStrength");
	const int lightRangeLoc = glGetUniformLocation(shadingShader.handle(), "uLightRange");

	// Prepare for binding shadow map
	gl::setUniform(shadingShader, "uShadowMap", 4);
	glActiveTexture(GL_TEXTURE4);

	// Static lights
	for (uint32_t i = 0; i < mImpl->staticSphereLights.size(); i++) {
		const SphereLight& sphereLight = mImpl->staticSphereLights[i];
		const ShadowCubeMap& shadowMap = mImpl->staticShadowMaps[i];

		// Set uniforms
		const vec3 lightPosVS = transformPoint(viewMatrix, sphereLight.pos);
		gl::setUniform(lightPosVSLoc, lightPosVS);
		gl::setUniform(lightPosWSLoc, sphereLight.pos);
		gl::setUniform(lightStrengthLoc, sphereLight.strength);
		gl::setUniform(lightRangeLoc, sphereLight.range);

		// Bind shadow map
		glBindTexture(GL_TEXTURE_CUBE_MAP, shadowMap.shadowCubeMap());

		// Run light pass
		mImpl->fullscreenTriangle.render();
	}

	// Unbind shadow map
	glBindTexture(GL_TEXTURE_CUBE_MAP, 0);

	// Dynamic lights
	/*for (const SphereLight& sphereLight : lights) {
		const vec3 lightPosVS = transformPoint(viewMatrix, sphereLight.pos);
		gl::setUniform(lightPosVSLoc, lightPosVS);
		gl::setUniform(lightStrengthLoc, sphereLight.strength);
		gl::setUniform(lightRangeLoc, sphereLight.range);

		mImpl->fullscreenTriangle.render();
	}*/

	RenderResult result;
	result.renderedRes = mTargetResolution;
	result.depthTexture = gbuffer.depthTexture();
	result.colorTexture = mImpl->resultFB.texture(0);
	result.materialTexture = mImpl->gbuffer.texture(GBUFFER_MATERIAL);
	result.velocityTexture = gbuffer.texture(GBUFFER_VELOCITY);
	return result;
}

// DeferredRenderer: Protected virtual methods from BaseRenderer interface
// ------------------------------------------------------------------------------------------------

void DeferredRenderer::targetResolutionUpdated() noexcept
{
	using gl::FBDepthFormat;
	using gl::FBTextureFiltering;
	using gl::FBTextureFormat;
	using gl::FramebufferBuilder;

	mImpl->resultFB = FramebufferBuilder(mTargetResolution)
	    .addTexture(0, FBTextureFormat::RGBA_F16, FBTextureFiltering::LINEAR)
	    .addDepthTexture(FBDepthFormat::F32, FBTextureFiltering::NEAREST)
	    .build();

	mImpl->gbuffer = FramebufferBuilder(mTargetResolution)
	    .addDepthTexture(FBDepthFormat::F32, FBTextureFiltering::NEAREST)
	    .addTexture(GBUFFER_NORMAL, FBTextureFormat::RGB_F16, FBTextureFiltering::LINEAR)
	    .addTexture(GBUFFER_ALBEDO, FBTextureFormat::RGB_U8, FBTextureFiltering::LINEAR)
	    .addTexture(GBUFFER_MATERIAL, FBTextureFormat::RG_U8, FBTextureFiltering::LINEAR) // Roughness, metallic
	    .addTexture(GBUFFER_VELOCITY, FBTextureFormat::RGB_F32, FBTextureFiltering::LINEAR)
	    .build();
}

} // namespace phe
