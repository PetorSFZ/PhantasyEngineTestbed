// Copyright (c) Peter Hillerström (skipifzero.com, peter@hstroem.se)

#include "renderers/DeferredRenderer.hpp"

#include <sfz/containers/StackString.hpp>
#include <sfz/gl/IncludeOpenGL.hpp>
#include <sfz/util/IO.hpp>

namespace sfz {

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

void DeferredRenderer::render(const DynArray<DrawOp>& operations) noexcept
{
	const mat4 viewMatrix = mMatrices.headMatrix * mMatrices.originMatrix;
	const mat4 projMatrix = mMatrices.projMatrix;
	const mat4 invProjMatrix = inverse(projMatrix);

	// GBuffer generation
	// --------------------------------------------------------------------------------------------

	glDisable(GL_BLEND);
	glEnable(GL_CULL_FACE);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);

	mGBuffer.bindViewportClearColorDepth(vec2i(0), mResolution);
	mGBufferGenShader.useProgram();

	const mat4 modelMatrix = identityMatrix4<float>();
	gl::setUniform(mGBufferGenShader, "uProjMatrix", projMatrix);
	gl::setUniform(mGBufferGenShader, "uViewMatrix", viewMatrix);

	gl::setUniform(mGBufferGenShader, "uAlbedoTexture", 0);
	gl::setUniform(mGBufferGenShader, "uRoughnessTexture", 1);
	gl::setUniform(mGBufferGenShader, "uMetallicTexture", 2);
	gl::setUniform(mGBufferGenShader, "uSpecularTexture", 3);

	const int modelMatrixLoc = glGetUniformLocation(mGBufferGenShader.handle(), "uModelMatrix");
	const int normalMatrixLoc = glGetUniformLocation(mGBufferGenShader.handle(), "uNormalMatrix");

	const int hasAlbedoTextureLoc = glGetUniformLocation(mGBufferGenShader.handle(), "uHasAlbedoTexture");
	const int albedoValueLoc = glGetUniformLocation(mGBufferGenShader.handle(), "uAlbedoValue");

	const int hasRoughnessTextureLoc = glGetUniformLocation(mGBufferGenShader.handle(), "uHasRoughnessTexture");
	const int rougnessValueLoc = glGetUniformLocation(mGBufferGenShader.handle(), "uRoughnessValue");

	const int hasMetallicTextureLoc = glGetUniformLocation(mGBufferGenShader.handle(), "uHasMetallicTexture");
	const int metallicValueLoc = glGetUniformLocation(mGBufferGenShader.handle(), "uMetallicValue");

	const int hasSpecularTextureLoc = glGetUniformLocation(mGBufferGenShader.handle(), "uHasSpecularTexture");
	const int specularValueLoc = glGetUniformLocation(mGBufferGenShader.handle(), "uSpecularValue");

	for (const DrawOp& op : operations) {
		sfz_assert_debug(op.renderablePtr != nullptr);

		// Set model & normal matrix (= inverse(tranpose(modelViewMatrix)))
		gl::setUniform(modelMatrixLoc, op.transform);
		gl::setUniform(normalMatrixLoc, inverse(transpose(viewMatrix * op.transform)));

		// Set albedo
		if (op.renderablePtr->glMaterial.albedoTexture.isValid()) {
			gl::setUniform(hasAlbedoTextureLoc, 1);
			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_2D, op.renderablePtr->glMaterial.albedoTexture.handle());
		} else {
			gl::setUniform(hasAlbedoTextureLoc, 0);
			gl::setUniform(albedoValueLoc, op.renderablePtr->material.albedoValue);
		}

		// Set roughness
		if (op.renderablePtr->glMaterial.roughnessTexture.isValid()) {
			gl::setUniform(hasRoughnessTextureLoc, 1);
			glActiveTexture(GL_TEXTURE1);
			glBindTexture(GL_TEXTURE_2D, op.renderablePtr->glMaterial.roughnessTexture.handle());
		} else {
			gl::setUniform(hasRoughnessTextureLoc, 0);
			gl::setUniform(rougnessValueLoc, op.renderablePtr->material.roughnessValue);
		}

		// Set metallic
		if (op.renderablePtr->glMaterial.metallicTexture.isValid()) {
			gl::setUniform(hasMetallicTextureLoc, 1);
			glActiveTexture(GL_TEXTURE2);
			glBindTexture(GL_TEXTURE_2D, op.renderablePtr->glMaterial.metallicTexture.handle());
		} else {
			gl::setUniform(hasMetallicTextureLoc, 0);
			gl::setUniform(metallicValueLoc, op.renderablePtr->material.metallicValue);
		}

		// Set specular
		if (op.renderablePtr->glMaterial.specularTexture.isValid()) {
			gl::setUniform(hasSpecularTextureLoc, 1);
			glActiveTexture(GL_TEXTURE3);
			glBindTexture(GL_TEXTURE_2D, op.renderablePtr->glMaterial.specularTexture.handle());
		} else {
			gl::setUniform(hasSpecularTextureLoc, 0);
			gl::setUniform(specularValueLoc, op.renderablePtr->material.specularValue);
		}

		// Render model
		op.renderablePtr->glModel.draw();
	}

	// Shading
	// --------------------------------------------------------------------------------------------

	glDisable(GL_BLEND);
	glEnable(GL_CULL_FACE);
	glDisable(GL_DEPTH_TEST);

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glViewport(0, 0, mResolution.x, mResolution.y);
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClearDepth(1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

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

	mFullscreenQuad.render();
}

const Framebuffer& DeferredRenderer::getResult() const noexcept
{
	return mResult;
}

const Framebuffer& DeferredRenderer::getResultVR(uint32_t eye) const noexcept
{
	sfz_assert_debug(eye <= 1);
	return mResultVR[eye];
}

// DeferredRenderer: Protected virtual methods from BaseRenderer interface
// ------------------------------------------------------------------------------------------------

void DeferredRenderer::maxResolutionUpdated() noexcept
{
	using gl::FBDepthFormat;
	using gl::FBTextureFiltering;
	using gl::FBTextureFormat;
	using gl::FramebufferBuilder;

	mGBuffer = FramebufferBuilder(mMaxResolution)
	          .addDepthTexture(FBDepthFormat::F32, FBTextureFiltering::NEAREST)
	          .addTexture(GBUFFER_NORMAL, FBTextureFormat::RGB_S8, FBTextureFiltering::LINEAR)
	          .addTexture(GBUFFER_ALBEDO, FBTextureFormat::RGB_U8, FBTextureFiltering::LINEAR)
	          .addTexture(GBUFFER_MATERIAL, FBTextureFormat::RGB_U8, FBTextureFiltering::LINEAR) // Roughness, metallic, specular
	          .build();
}

void DeferredRenderer::resolutionUpdated() noexcept
{
	
}

} // namespace sfz
