// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#include "phantasy_engine/deferred_renderer/DeferredRenderer.hpp"

#include <sfz/containers/StackString.hpp>
#include <sfz/gl/IncludeOpenGL.hpp>
#include <sfz/gl/Program.hpp>
#include <sfz/memory/New.hpp>
#include <sfz/util/IO.hpp>

#include "phantasy_engine/deferred_renderer/GLModel.hpp"
#include "phantasy_engine/level/PointLight.hpp"
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

// DeferredRendererImpl
// ------------------------------------------------------------------------------------------------

class DeferredRendererImpl final {
public:
	Program gbufferGenShader, shadingShader;
	Framebuffer gbuffer;
	FullscreenTriangle fullscreenTriangle;

	// Static scene
	DynArray<GLModel> staticGLModels;
	DynArray<PointLight> statigPointLights;

	DeferredRendererImpl() noexcept
	{

	}
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

void DeferredRenderer::bakeMaterials(const DynArray<RawImage>& textures,
                                     const DynArray<Material>& materials) noexcept
{
	
}

void DeferredRenderer::addMaterial(RawImage& texture, Material& material) noexcept
{
	sfz::error("DeferredRenderer: addMaterial() not implemented");
}

void DeferredRenderer::bakeStaticScene(const StaticScene& staticScene) noexcept
{
	DynArray<GLModel>& glModels = mImpl->staticGLModels;

	for (const RawMesh& mesh : staticScene.meshes) {
		glModels.add(GLModel(mesh));
	}

	mImpl->statigPointLights = staticScene.pointLights;
}

RenderResult DeferredRenderer::render(Framebuffer& resultFB) noexcept
{
	auto& gbufferGenShader = mImpl->gbufferGenShader;
	auto& shadingShader = mImpl->shadingShader;
	auto& gbuffer = mImpl->gbuffer;

	const mat4 viewMatrix = mMatrices.headMatrix * mMatrices.originMatrix;
	const mat4 projMatrix = mMatrices.projMatrix;
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

	gl::setUniform(gbufferGenShader, "uAlbedoTexture", 0);
	gl::setUniform(gbufferGenShader, "uRoughnessTexture", 1);
	gl::setUniform(gbufferGenShader, "uMetallicTexture", 2);
	gl::setUniform(gbufferGenShader, "uSpecularTexture", 3);

	// For the static scene the model matrix should be identity
	gl::setUniform(gbufferGenShader, "uModelMatrix", identityMatrix4<float>());
	gl::setUniform(gbufferGenShader, "uNormalMatrix", inverse(transpose(viewMatrix)));

	//const int modelMatrixLoc = glGetUniformLocation(mGBufferGenShader.handle(), "uModelMatrix");
	//const int normalMatrixLoc = glGetUniformLocation(mGBufferGenShader.handle(), "uNormalMatrix");

	const int hasAlbedoTextureLoc = glGetUniformLocation(gbufferGenShader.handle(), "uHasAlbedoTexture");
	const int albedoValueLoc = glGetUniformLocation(gbufferGenShader.handle(), "uAlbedoValue");

	const int hasRoughnessTextureLoc = glGetUniformLocation(gbufferGenShader.handle(), "uHasRoughnessTexture");
	const int rougnessValueLoc = glGetUniformLocation(gbufferGenShader.handle(), "uRoughnessValue");

	const int hasMetallicTextureLoc = glGetUniformLocation(gbufferGenShader.handle(), "uHasMetallicTexture");
	const int metallicValueLoc = glGetUniformLocation(gbufferGenShader.handle(), "uMetallicValue");

	for (const GLModel& model : mImpl->staticGLModels) {
		

		model.draw();
	}

	/*for (const RenderableComponent& component : mStaticScene->opaqueComponents) {

		const Material& m = component.material;

		// Set albedo
		if (m.albedoIndex != uint32_t(~0)) {
			gl::setUniform(hasAlbedoTextureLoc, 1);
			glActiveTexture(GL_TEXTURE0);
			sfz_assert_debug(m.albedoIndex < mStaticScene->textures.size());
			sfz_assert_debug(mStaticScene->textures[m.albedoIndex].isValid());
			glBindTexture(GL_TEXTURE_2D, mStaticScene->textures[m.albedoIndex].handle());
		} else {
			gl::setUniform(hasAlbedoTextureLoc, 0);
			gl::setUniform(albedoValueLoc, m.albedoValue);
		}

		// Set roughness
		if (m.roughnessIndex != uint32_t(~0)) {
			gl::setUniform(hasRoughnessTextureLoc, 1);
			glActiveTexture(GL_TEXTURE1);
			sfz_assert_debug(m.roughnessIndex < mStaticScene->textures.size());
			sfz_assert_debug(mStaticScene->textures[m.roughnessIndex].isValid());
			glBindTexture(GL_TEXTURE_2D, mStaticScene->textures[m.roughnessIndex].handle());
		} else {
			gl::setUniform(hasRoughnessTextureLoc, 0);
			gl::setUniform(rougnessValueLoc, m.roughnessValue);
		}

		// Set metallic
		if (m.metallicIndex != uint32_t(~0)) {
			gl::setUniform(hasMetallicTextureLoc, 1);
			glActiveTexture(GL_TEXTURE2);
			sfz_assert_debug(m.metallicIndex < mStaticScene->textures.size());
			sfz_assert_debug(mStaticScene->textures[m.metallicIndex].isValid());
			glBindTexture(GL_TEXTURE_2D, mStaticScene->textures[m.metallicIndex].handle());
		} else {
			gl::setUniform(hasMetallicTextureLoc, 0);
			gl::setUniform(metallicValueLoc, m.metallicValue);
		}

		// Render model
		component.glModel.draw();
	}*/

	// Shading
	// --------------------------------------------------------------------------------------------

	glDisable(GL_BLEND);
	glEnable(GL_CULL_FACE);
	glDisable(GL_DEPTH_TEST);

	glEnable(GL_BLEND);
	glBlendEquation(GL_FUNC_ADD);
	glBlendFunc(GL_ONE, GL_ONE);

	resultFB.bindViewportClearColorDepth(vec2i(0.0), mTargetResolution, vec4(0.0f), 0.0f);
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

	for (const PointLight& pointLight : mImpl->statigPointLights) {
		const vec3 lightPosVS = transformPoint(viewMatrix, pointLight.pos);
		gl::setUniform(lightPosLoc, lightPosVS);
		gl::setUniform(lightStrengthLoc, pointLight.strength);
		gl::setUniform(lightRangeLoc, pointLight.range);

		mImpl->fullscreenTriangle.render();
	}
	
	RenderResult tmp;
	tmp.renderedRes = mTargetResolution;
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

	mImpl->gbuffer = FramebufferBuilder(mTargetResolution)
	    .addDepthTexture(FBDepthFormat::F32, FBTextureFiltering::NEAREST)
	    .addTexture(GBUFFER_NORMAL, FBTextureFormat::RGB_F16, FBTextureFiltering::LINEAR)
	    .addTexture(GBUFFER_ALBEDO, FBTextureFormat::RGB_U8, FBTextureFiltering::LINEAR)
	    .addTexture(GBUFFER_MATERIAL, FBTextureFormat::RG_U8, FBTextureFiltering::LINEAR) // Roughness, metallic
	    .build();
}

} // namespace phe
