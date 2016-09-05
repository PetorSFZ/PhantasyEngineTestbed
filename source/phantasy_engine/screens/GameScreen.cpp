// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#include "phantasy_engine/screens/GameScreen.hpp"

#include <sfz/gl/IncludeOpenGL.hpp>
#include <sfz/util/IO.hpp>
#include <sfz/math/Vector.hpp>

#include "phantasy_engine/config/GlobalConfig.hpp"

namespace phe {

using namespace sfz;

// GameScreen: Constructors & destructors
// ------------------------------------------------------------------------------------------------

GameScreen::GameScreen(UniquePtr<GameLogic>&& gameLogicIn, UniquePtr<Level>&& levelIn,
                       UniquePtr<BaseRenderer>&& rendererIn) noexcept
:
	gameLogic(std::move(gameLogicIn)),
	level(std::move(levelIn)),
	renderer(std::move(rendererIn))
{
	auto& cfg = GlobalConfig::instance();

	cam = ViewFrustum(vec3(0.0f, 3.0f, -6.0f), normalize(vec3(0.0f, -0.25f, 1.0f)),
	                  normalize(vec3(0.0f, 1.0f, 0.0)), 60.0f, 1.0f, 0.01f, 10000.0f);
	
	// Load shaders
	StackString192 shadersPath;
	shadersPath.printf("%sresources/shaders/", basePath());

	mScalingShader = Program::postProcessFromFile(shadersPath.str, "scaling.frag");
	glUseProgram(mScalingShader.handle());
	gl::setUniform(mScalingShader, "uSrcTexture", 0);

	mGammaCorrectionShader = Program::postProcessFromFile(shadersPath.str, "gamma_correction.frag");
	glUseProgram(mGammaCorrectionShader.handle());
	gl::setUniform(mGammaCorrectionShader, "uLinearTexture", 0);

	// Set and bake static scene
	this->renderer->setAndBakeStaticScene(this->level->staticScene);
}

// GameScreen: Overriden methods from sfz::BaseScreen
// ------------------------------------------------------------------------------------------------

UpdateOp GameScreen::update(UpdateState& state)
{
	return gameLogic->update(*this, state);
}

void GameScreen::render(UpdateState& state)
{
	auto& cfg = GlobalConfig::instance();

	const vec2i drawableDim = state.window.drawableDimensions();
	const vec2i targetRes = cfg.graphcisCfg().getTargetResolution(drawableDim);

	// Check if framebuffers / renderer needs to be reloaded
	if (targetRes != renderer->targetResolution()) {
		printf("New target resolution: %s, reloading framebuffers\n", toString(targetRes).str);
		this->reloadFramebuffers(targetRes);
		renderer->setTargetResolution(targetRes);
		cam.setAspectRatio(float(targetRes.x) / float(targetRes.y));
	}

	// Render the level
	RenderResult res = renderer->render(mResultFB);

	glDisable(GL_BLEND);
	glEnable(GL_CULL_FACE);
	glDisable(GL_DEPTH_TEST);
	glActiveTexture(GL_TEXTURE0);

	// Apply gamma correction
	mGammaCorrectedFB.bindViewportClearColor();
	glUseProgram(mGammaCorrectionShader.handle());
	gl::setUniform(mGammaCorrectionShader, "uGamma", cfg.windowCfg().screenGamma->floatValue());
	glBindTexture(GL_TEXTURE_2D, mResultFB.texture(0));
	mFullscreenTriangle.render();

	// Scale result to screen
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glViewport(0, 0, drawableDim.x, drawableDim.y);
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClearDepth(0.0f); // Reverse depth buffer
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glUseProgram(mScalingShader.handle());
	glBindTexture(GL_TEXTURE_2D, mGammaCorrectedFB.texture(0));
	gl::setUniform(mScalingShader, "uViewportRes", vec2(res.renderedRes));
	gl::setUniform(mScalingShader, "uDstRes", vec2(drawableDim));

	mFullscreenTriangle.render();

	SDL_GL_SwapWindow(state.window.ptr());
}

// GameScreen: Private methods
// ------------------------------------------------------------------------------------------------

void GameScreen::reloadFramebuffers(vec2i internalRes) noexcept
{
	using gl::FBDepthFormat;
	using gl::FBTextureFiltering;
	using gl::FBTextureFormat;
	using gl::FramebufferBuilder;

	if (mResultFB.dimensions() != internalRes) {
		mResultFB = FramebufferBuilder(internalRes)
		            .addTexture(0, FBTextureFormat::RGBA_F16, FBTextureFiltering::LINEAR)
		            .addDepthTexture(FBDepthFormat::F32, FBTextureFiltering::NEAREST)
		            .build();
	}

	if (mGammaCorrectedFB.dimensions() != internalRes) {
		mGammaCorrectedFB = FramebufferBuilder(internalRes)
		                    .addTexture(0, FBTextureFormat::RGBA_F16, FBTextureFiltering::LINEAR)
		                    .build();
	}
}

void GameScreen::reloadShaders() noexcept
{
	mScalingShader.reload();
	if (mScalingShader.isValid()) {
		glUseProgram(mScalingShader.handle());
		gl::setUniform(mScalingShader, "uSrcTexture", 0);
	}

	mGammaCorrectionShader.reload();
	if (mGammaCorrectionShader.isValid()) {
		glUseProgram(mGammaCorrectionShader.handle());
		gl::setUniform(mGammaCorrectionShader, "uLinearTexture", 0);
	}
}

} // namespace phe
