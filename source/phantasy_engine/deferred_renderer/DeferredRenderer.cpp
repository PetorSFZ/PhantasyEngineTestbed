// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#include "phantasy_engine/deferred_renderer/DeferredRenderer.hpp"

#include <sfz/containers/StackString.hpp>
#include <sfz/gl/IncludeOpenGL.hpp>
#include <sfz/gl/Program.hpp>
#include <sfz/memory/New.hpp>
#include <sfz/util/IO.hpp>

#include "phantasy_engine/deferred_renderer/GLModel.hpp"
#include "phantasy_engine/deferred_renderer/GLTexture.hpp"
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
	Program gbufferGenShader, shadingShader;
	
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
	this->mImpl = sfz_new<DeferredRendererImpl>();

	StackString128 shadersPath;
	shadersPath.printf("%sresources/shaders/deferred_renderer/", basePath());

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
	sfz_delete(mImpl);
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

RenderResult DeferredRenderer::render(const DynArray<DynObject>& objects,
                                      const DynArray<SphereLight>& lights) noexcept
{
	auto& gbufferGenShader = mImpl->gbufferGenShader;
	auto& shadingShader = mImpl->shadingShader;
	auto& resultFb = mImpl->resultFB;
	auto& gbuffer = mImpl->gbuffer;

	const mat4 viewMatrix = mCamera.viewMatrix();
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

	const mat4 modelMatrix = identityMatrix4<float>();
	gl::setUniform(gbufferGenShader, "uProjMatrix", projMatrix);
	gl::setUniform(gbufferGenShader, "uViewMatrix", viewMatrix);

	// Bind SSBOs
	mImpl->materialSSBO.bind(0);
	mImpl->texturesSSBO.bind(1);

	const int modelMatrixLoc = glGetUniformLocation(gbufferGenShader.handle(), "uModelMatrix");
	const int normalMatrixLoc = glGetUniformLocation(gbufferGenShader.handle(), "uNormalMatrix");
	const int worldVelocityLoc = glGetUniformLocation(gbufferGenShader.handle(), "uWorldVelocity");

	// For the static scene the model matrix should be identity
	gl::setUniform(modelMatrixLoc, identityMatrix4<float>());
	gl::setUniform(normalMatrixLoc, inverse(transpose(viewMatrix)));

	// Set velocity of static geometry to 0
	gl::setUniform(worldVelocityLoc, vec3(0.0f));
	for (const GLModel& model : mImpl->staticGLModels) {
		model.draw();
	}

	for (const DynObject& obj : objects) {
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

	const int lightPosLoc = glGetUniformLocation(shadingShader.handle(), "uLightPos");
	const int lightStrengthLoc = glGetUniformLocation(shadingShader.handle(), "uLightStrength");
	const int lightRangeLoc = glGetUniformLocation(shadingShader.handle(), "uLightRange");

	// Static lights
	for (const SphereLight& sphereLight : mImpl->staticSphereLights) {
		const vec3 lightPosVS = transformPoint(viewMatrix, sphereLight.pos);
		gl::setUniform(lightPosLoc, lightPosVS);
		gl::setUniform(lightStrengthLoc, sphereLight.strength);
		gl::setUniform(lightRangeLoc, sphereLight.range);

		mImpl->fullscreenTriangle.render();
	}

	// Dynamic lights
	for (const SphereLight& sphereLight : lights) {
		const vec3 lightPosVS = transformPoint(viewMatrix, sphereLight.pos);
		gl::setUniform(lightPosLoc, lightPosVS);
		gl::setUniform(lightStrengthLoc, sphereLight.strength);
		gl::setUniform(lightRangeLoc, sphereLight.range);

		mImpl->fullscreenTriangle.render();
	}

	RenderResult tmp;
	tmp.renderedRes = mTargetResolution;
	tmp.depthTexture = gbuffer.depthTexture();
	tmp.colorTexture = mImpl->resultFB.texture(0);
	tmp.velocityTexture = gbuffer.texture(3);
	return tmp;
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
