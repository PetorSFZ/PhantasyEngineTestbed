// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#include <sfz/containers/StackString.hpp>
#include <sfz/gl/IncludeOpenGL.hpp>
#include <sfz/util/IO.hpp>

#include "phantasy_engine/renderers/DeferredRenderer.hpp"
#include "phantasy_engine/level/PointLight.hpp"

namespace phe {

using namespace sfz;

// Statics
// ------------------------------------------------------------------------------------------------

static const uint32_t GBUFFER_NORMAL = 0;
static const uint32_t GBUFFER_ALBEDO = 1;
static const uint32_t GBUFFER_MATERIAL = 2;

// DeferredRenderer: Constructors & destructors
// ------------------------------------------------------------------------------------------------

DeferredRenderer::DeferredRenderer() noexcept
{
	StackString128 shadersPath;
	shadersPath.printf("%sresources/shaders/", basePath());

	mGBufferGenShader = Program::fromFile(shadersPath.str, "gbuffer_gen.vert", "gbuffer_gen.frag",
	[](uint32_t shaderProgram) {
		glBindAttribLocation(shaderProgram, 0, "inPosition");
		glBindAttribLocation(shaderProgram, 1, "inNormal");
		glBindAttribLocation(shaderProgram, 2, "inUV");
	});

	mShadingShader = Program::postProcessFromFile(shadersPath.str, "shading.frag");
}

// DeferredRenderer: Virtual methods from BaseRenderer interface
// ------------------------------------------------------------------------------------------------

RenderResult DeferredRenderer::render(Framebuffer& resultFB) noexcept
{
	const mat4 viewMatrix = mMatrices.headMatrix * mMatrices.originMatrix;
	const mat4 projMatrix = mMatrices.projMatrix;
	const mat4 invProjMatrix = inverse(projMatrix);

	// GBuffer generation
	// --------------------------------------------------------------------------------------------

	glDisable(GL_BLEND);
	glEnable(GL_CULL_FACE);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_GREATER); // reversed-z

	mGBuffer.bindViewportClearColorDepth(vec2i(0), mTargetResolution, vec4(0.0f), 0.0f);
	mGBufferGenShader.useProgram();

	const mat4 modelMatrix = identityMatrix4<float>();
	gl::setUniform(mGBufferGenShader, "uProjMatrix", projMatrix);
	gl::setUniform(mGBufferGenShader, "uViewMatrix", viewMatrix);

	gl::setUniform(mGBufferGenShader, "uAlbedoTexture", 0);
	gl::setUniform(mGBufferGenShader, "uRoughnessTexture", 1);
	gl::setUniform(mGBufferGenShader, "uMetallicTexture", 2);
	gl::setUniform(mGBufferGenShader, "uSpecularTexture", 3);

	// For the static scene the model matrix should be identity
	gl::setUniform(mGBufferGenShader, "uModelMatrix", identityMatrix4<float>());
	gl::setUniform(mGBufferGenShader, "uNormalMatrix", inverse(transpose(viewMatrix)));

	//const int modelMatrixLoc = glGetUniformLocation(mGBufferGenShader.handle(), "uModelMatrix");
	//const int normalMatrixLoc = glGetUniformLocation(mGBufferGenShader.handle(), "uNormalMatrix");

	const int hasAlbedoTextureLoc = glGetUniformLocation(mGBufferGenShader.handle(), "uHasAlbedoTexture");
	const int albedoValueLoc = glGetUniformLocation(mGBufferGenShader.handle(), "uAlbedoValue");

	const int hasRoughnessTextureLoc = glGetUniformLocation(mGBufferGenShader.handle(), "uHasRoughnessTexture");
	const int rougnessValueLoc = glGetUniformLocation(mGBufferGenShader.handle(), "uRoughnessValue");

	const int hasMetallicTextureLoc = glGetUniformLocation(mGBufferGenShader.handle(), "uHasMetallicTexture");
	const int metallicValueLoc = glGetUniformLocation(mGBufferGenShader.handle(), "uMetallicValue");

	for (const Renderable& renderable : mStaticScene->opaqueRenderables) {

		for (const RenderableComponent& comp : renderable.components) {

			const Material& m = comp.material;

			// Set albedo
			if (m.albedoIndex != uint32_t(~0)) {
				gl::setUniform(hasAlbedoTextureLoc, 1);
				glActiveTexture(GL_TEXTURE0);
				sfz_assert_debug(m.albedoIndex < renderable.textures.size());
				sfz_assert_debug(renderable.textures[m.albedoIndex].isValid());
				glBindTexture(GL_TEXTURE_2D, renderable.textures[m.albedoIndex].handle());
			} else {
				gl::setUniform(hasAlbedoTextureLoc, 0);
				gl::setUniform(albedoValueLoc, m.albedoValue);
			}

			// Set roughness
			if (m.roughnessIndex != uint32_t(~0)) {
				gl::setUniform(hasRoughnessTextureLoc, 1);
				glActiveTexture(GL_TEXTURE1);
				sfz_assert_debug(m.roughnessIndex < renderable.textures.size());
				sfz_assert_debug(renderable.textures[m.roughnessIndex].isValid());
				glBindTexture(GL_TEXTURE_2D, renderable.textures[m.roughnessIndex].handle());
			} else {
				gl::setUniform(hasRoughnessTextureLoc, 0);
				gl::setUniform(rougnessValueLoc, m.roughnessValue);
			}

			// Set metallic
			if (m.metallicIndex != uint32_t(~0)) {
				gl::setUniform(hasMetallicTextureLoc, 1);
				glActiveTexture(GL_TEXTURE2);
				sfz_assert_debug(m.metallicIndex < renderable.textures.size());
				sfz_assert_debug(renderable.textures[m.metallicIndex].isValid());
				glBindTexture(GL_TEXTURE_2D, renderable.textures[m.metallicIndex].handle());
			} else {
				gl::setUniform(hasMetallicTextureLoc, 0);
				gl::setUniform(metallicValueLoc, m.metallicValue);
			}

			// Render model
			comp.glModel.draw();
		}
	}

	// Shading
	// --------------------------------------------------------------------------------------------

	glDisable(GL_BLEND);
	glEnable(GL_CULL_FACE);
	glDisable(GL_DEPTH_TEST);

	glEnable(GL_BLEND);
	glBlendEquation(GL_FUNC_ADD);
	glBlendFunc(GL_ONE, GL_ONE);

	resultFB.bindViewportClearColorDepth(vec2i(0.0), mTargetResolution, vec4(0.0f), 0.0f);
	mShadingShader.useProgram();

	gl::setUniform(mShadingShader, "uInvProjMatrix", invProjMatrix);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, mGBuffer.depthTexture());
	gl::setUniform(mShadingShader, "uDepthTexture", 0);

	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, mGBuffer.texture(GBUFFER_NORMAL));
	gl::setUniform(mShadingShader, "uNormalTexture", 1);

	glActiveTexture(GL_TEXTURE2);
	glBindTexture(GL_TEXTURE_2D, mGBuffer.texture(GBUFFER_ALBEDO));
	gl::setUniform(mShadingShader, "uAlbedoTexture", 2);

	glActiveTexture(GL_TEXTURE3);
	glBindTexture(GL_TEXTURE_2D, mGBuffer.texture(GBUFFER_MATERIAL));
	gl::setUniform(mShadingShader, "uMaterialTexture", 3);

	const int lightPosLoc = glGetUniformLocation(mShadingShader.handle(), "uLightPos");
	const int lightStrengthLoc = glGetUniformLocation(mShadingShader.handle(), "uLightStrength");
	const int lightRangeLoc = glGetUniformLocation(mShadingShader.handle(), "uLightRange");

	for (const PointLight& pointLight : mStaticScene->pointLights) {
		const vec3 lightPosVS = transformPoint(viewMatrix, pointLight.pos);
		gl::setUniform(lightPosLoc, lightPosVS);
		gl::setUniform(lightStrengthLoc, pointLight.strength);
		gl::setUniform(lightRangeLoc, pointLight.range);

		mFullscreenTriangle.render();
	}

	RenderResult tmp;
	tmp.renderedRes = mTargetResolution;
	return tmp;
}

// DeferredRenderer: Protected virtual methods from BaseRenderer interface
// ------------------------------------------------------------------------------------------------

void DeferredRenderer::staticSceneChanged() noexcept
{
	// TODO: Bake shadow maps for all static lights
}

void DeferredRenderer::targetResolutionUpdated() noexcept
{
	using gl::FBDepthFormat;
	using gl::FBTextureFiltering;
	using gl::FBTextureFormat;
	using gl::FramebufferBuilder;

	mGBuffer = FramebufferBuilder(mTargetResolution)
	          .addDepthTexture(FBDepthFormat::F32, FBTextureFiltering::NEAREST)
	          .addTexture(GBUFFER_NORMAL, FBTextureFormat::RGB_F16, FBTextureFiltering::LINEAR)
	          .addTexture(GBUFFER_ALBEDO, FBTextureFormat::RGB_U8, FBTextureFiltering::LINEAR)
	          .addTexture(GBUFFER_MATERIAL, FBTextureFormat::RG_U8, FBTextureFiltering::LINEAR) // Roughness, metallic
	          .build();
}

} // namespace phe
