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
	StackString128 modelsPath;
	modelsPath.printf("%sresources/models/", basePath());
	StackString128 shadersPath;
	shadersPath.printf("%sresources/shaders/", basePath());

	mSnakeRenderable = tinyObjLoadRenderable(modelsPath.str, "head_d2u_f2.obj");

	mTempShader = Program::fromFile(shadersPath.str, "temp_shader.vert", "temp_shader.frag",
	[](uint32_t shaderProgram) {
		glBindAttribLocation(shaderProgram, 0, "inPosition");
		glBindAttribLocation(shaderProgram, 1, "inNormal");
		glBindAttribLocation(shaderProgram, 2, "inUV");
	});

	mCam = sfz::ViewFrustum(vec3(0.0f, 3.0f, -6.0f), normalize(vec3(0.0f, -0.25f, 1.0f)),
	                        normalize(vec3(0.0f, 1.0f, 0.0)), 60.0f, 1.0f, 0.01f, 100.0f);
}

// DeferredRenderer: Virtual methods from BaseRenderer interface
// ------------------------------------------------------------------------------------------------

void DeferredRenderer::render(const mat4& viewMatrix, const mat4& projMatrix) noexcept
{
	glEnable(GL_CULL_FACE);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClearDepth(1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	mTempShader.useProgram();

	const mat4 modelMatrix = identityMatrix4<float>();
	gl::setUniform(mTempShader, "uProjMatrix", mCam.projMatrix());
	gl::setUniform(mTempShader, "uViewMatrix", mCam.viewMatrix());
	gl::setUniform(mTempShader, "uModelMatrix", modelMatrix);
	gl::setUniform(mTempShader, "uNormalMatrix", inverse(transpose(mCam.viewMatrix() * modelMatrix))); // inverse(tranpose(modelViewMatrix))

	mSnakeRenderable.glModel.draw();
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

} // namespace sfz
