// Copyright (c) Peter Hillerström (skipifzero.com, peter@hstroem.se)

#include "renderers/DeferredRenderer.hpp"

#include <sfz/containers/StackString.hpp>
#include <sfz/gl/IncludeOpenGL.hpp>
#include <sfz/util/IO.hpp>

namespace sfz {

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

	mShadingShader = Program::fromFile(shadersPath.str, "shading.vert", "shading.frag",
		[](uint32_t shaderProgram) {
		glBindAttribLocation(shaderProgram, 0, "inPosition");
		glBindAttribLocation(shaderProgram, 1, "inNormal");
		glBindAttribLocation(shaderProgram, 2, "inUV");
	});
}

// DeferredRenderer: Virtual methods from BaseRenderer interface
// ------------------------------------------------------------------------------------------------

void DeferredRenderer::render(const DynArray<DrawOp>& operations) noexcept
{
	const mat4 viewMatrix = mMatrices.headMatrix * mMatrices.originMatrix;
	const mat4 projMatrix = mMatrices.projMatrix;

	glEnable(GL_CULL_FACE);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glViewport(0, 0, mResolution.x, mResolution.y);
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClearDepth(1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	mGBufferGenShader.useProgram();

	const mat4 modelMatrix = identityMatrix4<float>();
	gl::setUniform(mGBufferGenShader, "uProjMatrix", projMatrix);
	gl::setUniform(mGBufferGenShader, "uViewMatrix", viewMatrix);

	const int modelMatrixLoc = glGetUniformLocation(mGBufferGenShader.handle(), "uModelMatrix");
	const int normalMatrixLoc = glGetUniformLocation(mGBufferGenShader.handle(), "uNormalMatrix");

	for (const DrawOp& op : operations) {
		sfz_assert_debug(op.renderablePtr != nullptr);

		gl::setUniform(modelMatrixLoc, op.transform);
		gl::setUniform(normalMatrixLoc, inverse(transpose(viewMatrix * op.transform))); // inverse(tranpose(modelViewMatrix))
		op.renderablePtr->glModel.draw();
	}
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

}

void DeferredRenderer::resolutionUpdated() noexcept
{

}

} // namespace sfz
