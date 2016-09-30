// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#include "phantasy_engine/screens/GameScreen.hpp"

#include <imgui.h>
#include <imgui_impl_sdl_gl3.h>

#include <sfz/gl/IncludeOpenGL.hpp>
#include <sfz/util/IO.hpp>
#include <sfz/math/Vector.hpp>

#include "phantasy_engine/config/GlobalConfig.hpp"

namespace phe {

using namespace sfz;

// Constants
// ------------------------------------------------------------------------------------------------

static const vec2 HALTON_SEQ[] {
	{0.5f, 0.333333333333f},
	{0.25f, 0.666666666667f},
	{0.75f, 0.111111111111f},
	{0.125f, 0.444444444444f},
	{0.625f, 0.777777777778f},
	{0.375f, 0.222222222222f},
	{0.875f, 0.555555555556f},
	{0.0625f, 0.888888888889f},
	{0.5625f, 0.037037037037f},
	{0.3125f, 0.37037037037f},
	{0.8125f, 0.703703703704f},
	{0.1875f, 0.148148148148f},
	{0.6875f, 0.481481481481f},
	{0.4375f, 0.814814814815f},
	{0.9375f, 0.259259259259f},
	{0.03125f, 0.592592592593f}
};
static const uint32_t HALTON_LENGTH = sizeof(HALTON_SEQ) / sizeof(vec2);

// GameScreen: Constructors & destructors
// ------------------------------------------------------------------------------------------------

GameScreen::GameScreen(SharedPtr<GameLogic> gameLogicIn, SharedPtr<Level> levelIn,
                       SharedPtr<BaseRenderer> rendererIn) noexcept
:
	gameLogic(gameLogicIn),
	level(levelIn),
	renderer(rendererIn)
{
	auto& cfg = GlobalConfig::instance();

	cam = ViewFrustum(vec3(0.0f, 3.0f, -6.0f), normalize(vec3(1.0f, -0.25f, 0.0f)),
	                  normalize(vec3(0.0f, 1.0f, 0.0)), 60.0f, 1.0f, 0.01f, 10000.0f);
	mPreviousCamera = cam;

	// Load shaders
	StackString192 shadersPath;
	shadersPath.printf("%sresources/shaders/", basePath());

	mVelocityShader = Program::postProcessFromFile(shadersPath.str, "velocity_gen.frag");
	glUseProgram(mVelocityShader.handle());
	gl::setUniform(mVelocityShader, "uCurrDepthTexture", 0);

	mTaaShader = Program::postProcessFromFile(shadersPath.str, "taa.frag");
	glUseProgram(mTaaShader.handle());
	gl::setUniform(mTaaShader, "uSrcTexture", 0);
	gl::setUniform(mTaaShader, "uHistoryTexture", 1);
	gl::setUniform(mTaaShader, "uVelocityTexture", 2);
	gl::setUniform(mTaaShader, "uPrevVelocityTexture", 3);

	mScalingShader = Program::postProcessFromFile(shadersPath.str, "scaling.frag");
	glUseProgram(mScalingShader.handle());
	gl::setUniform(mScalingShader, "uSrcTexture", 0);

	mGammaCorrectionShader = Program::postProcessFromFile(shadersPath.str, "gamma_correction.frag");
	glUseProgram(mGammaCorrectionShader.handle());
	gl::setUniform(mGammaCorrectionShader, "uLinearTexture", 0);

	// Set and bake static scene
	this->renderer->setMaterialsAndTextures(this->level->materials, this->level->textures);
	this->renderer->setStaticScene(this->level->staticScene);
	this->renderer->setDynamicMeshes(this->level->meshes);
}

// GameScreen: Overriden methods from sfz::BaseScreen
// ------------------------------------------------------------------------------------------------

UpdateOp GameScreen::update(UpdateState& state)
{
	auto& cfg = GlobalConfig::instance();

	bool showDebugUI = cfg.debugCfg().showDebugUI->boolValue();

	// Intercept some key events at GameScreen level
	for (const SDL_Event& event : state.events) {
		switch (event.type) {
		case SDL_QUIT: return SCREEN_QUIT;
		case SDL_KEYUP:
			switch (event.key.keysym.sym) {
			case SDLK_F11:
				// Pressing F11 toggles the rendering of the debug UI
				showDebugUI = !showDebugUI;
				cfg.debugCfg().showDebugUI->setBool(showDebugUI);
				if (!showDebugUI) {
					debugUIActive = false;
					SDL_SetWindowGrab(state.window.ptr(), SDL_TRUE);
					SDL_SetRelativeMouseMode(SDL_TRUE);
					SDL_ShowCursor(SDL_FALSE);
				}
				break;
			default:
				// Do nothing
				break;
			}
			switch (event.key.keysym.scancode) {
			case SDL_SCANCODE_GRAVE:
				// Pressing the "grave" key (left of 1) toggles input focus between the game and the debug UI
				if (showDebugUI) {
					debugUIActive = !debugUIActive;
					SDL_SetWindowGrab(state.window.ptr(), !debugUIActive ? SDL_TRUE : SDL_FALSE);
					SDL_SetRelativeMouseMode(!debugUIActive ? SDL_TRUE : SDL_FALSE);
					SDL_ShowCursor(debugUIActive ? SDL_TRUE : SDL_FALSE);
				}
				break;
			default:
				// Do nothing
				break;
			}
			break;
		}
	}

	UpdateOp op = SCREEN_NO_OP;

	// When debug UI is active, prevent GameLogic update.
	// TODO: Perhaps only make GameLogic ignore input events when debug UI is active.
	if (!debugUIActive) {
		op = gameLogic->update(*this, state);
	} else {
		// Forward events to imgui
		for (DynArray<SDL_Event>* eventList : { &state.events, &state.controllerEvents, &state.mouseEvents }) {
			for (SDL_Event& event : *eventList) {
				ImGui_ImplSdlGL3_ProcessEvent(&event);
			}
		}
	}

	return op;
}

void GameScreen::render(UpdateState& state)
{
	auto& cfg = GlobalConfig::instance();

	bool showDebugUI = cfg.debugCfg().showDebugUI->boolValue();

	if (showDebugUI) {
		ImGui_ImplSdlGL3_NewFrame(state.window.ptr());

		if (!debugUIActive) {
			// It seems that there isn't a nicer way to make ImGui ignore mouse input
			ImGui::GetIO().MousePos = ImVec2(-1, -1);
		}
	}

	const vec2i drawableDim = state.window.drawableDimensions();
	const vec2i targetRes = cfg.graphcisCfg().getTargetResolution(drawableDim);

	// Check if framebuffers / renderer needs to be reloaded
	if (targetRes != renderer->targetResolution()) {
		printf("New target resolution: %s, reloading framebuffers\n", toString(targetRes).str);
		this->reloadFramebuffers(targetRes);
		renderer->setTargetResolution(targetRes);
		cam.setAspectRatio(float(targetRes.x) / float(targetRes.y));
	}

	bool taaEnabled = cfg.graphcisCfg().taa->boolValue();
	if (taaEnabled) {
		vec2 pixelOffset = HALTON_SEQ[mHaltonIndex] - vec2(0.5f);
		mHaltonIndex = (mHaltonIndex + 1) % HALTON_LENGTH;

		ViewFrustum jitteredCamera = cam;
		jitteredCamera.setPixelOffset(pixelOffset);
		renderer->updateCamera(jitteredCamera);
	} else {
		renderer->updateCamera(cam);
	}

	// Render the level
	uint32_t prevFBIndex = (mFBIndex + 1) % 2;
	Framebuffer& resultFB = mResultFB[mFBIndex];
	Framebuffer& prevResultFB = mResultFB[prevFBIndex];
	RenderResult res = renderer->render(resultFB, level->objects, level->sphereLights);

	glDisable(GL_BLEND);
	glEnable(GL_CULL_FACE);
	glDisable(GL_DEPTH_TEST);

	GLuint chainTexture = resultFB.texture(0);
	if (taaEnabled) {
		// Generate velocity buffer from depth
		mVelocityFB[mFBIndex].bindViewport();
		mVelocityShader.useProgram();
		gl::setUniform(mVelocityShader, "uCurrInvViewMatrix", inverse(cam.viewMatrix()));
		gl::setUniform(mVelocityShader, "uCurrInvProjMatrix", inverse(cam.projMatrix(targetRes)));
		gl::setUniform(mVelocityShader, "uPrevViewMatrix", mPreviousCamera.viewMatrix());
		gl::setUniform(mVelocityShader, "uPrevProjMatrix", mPreviousCamera.projMatrix(targetRes));

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, resultFB.depthTexture());
		glActiveTexture(GL_TEXTURE1);
		glBindTexture(GL_TEXTURE_2D, prevResultFB.depthTexture());
		mFullscreenTriangle.render();

		// Apply TAA
		mTaaFB[mFBIndex].bindViewport();
		glUseProgram(mTaaShader.handle());
		gl::setUniform(mTaaShader, "uResolution", mResultFB->dimensionsFloat());

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, chainTexture);
		glActiveTexture(GL_TEXTURE1);
		glBindTexture(GL_TEXTURE_2D, mTaaFB[prevFBIndex].texture(1));
		glActiveTexture(GL_TEXTURE2);
		glBindTexture(GL_TEXTURE_2D, mVelocityFB[mFBIndex].texture(0));
		glActiveTexture(GL_TEXTURE3);
		glBindTexture(GL_TEXTURE_2D, mVelocityFB[prevFBIndex].texture(0));

		mFullscreenTriangle.render();
		chainTexture = mTaaFB[mFBIndex].texture(0);
	}

	// Apply gamma correction
	mGammaCorrectedFB.bindViewportClearColor();
	glUseProgram(mGammaCorrectionShader.handle());
	gl::setUniform(mGammaCorrectionShader, "uGamma", cfg.windowCfg().screenGamma->floatValue());
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, chainTexture);
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

	if (showDebugUI) {
		// Change back to default clip space due to library incompatibility. This operation is
		// potentially expensive since it's intended to only be used once in the program.
		glClipControl(GL_LOWER_LEFT, GL_NEGATIVE_ONE_TO_ONE);

		renderDebugUI();

		// Reset to previous clip space
		glClipControl(GL_UPPER_LEFT, GL_ZERO_TO_ONE);
	}

	mFBIndex = (mFBIndex + 1) % 2;
	mPreviousCamera = cam;

	SDL_GL_SwapWindow(state.window.ptr());
}

void GameScreen::setRenderer(const SharedPtr<BaseRenderer>& renderer) noexcept
{
	this->renderer = renderer;

	for (int i = 0; i < 2; i++) {
		// Clear TAA history
		mTaaFB[i].bindViewportClearColor();
	}
}

// GameScreen: Private methods
// ------------------------------------------------------------------------------------------------

void GameScreen::reloadFramebuffers(vec2i internalRes) noexcept
{
	using gl::FBDepthFormat;
	using gl::FBTextureFiltering;
	using gl::FBTextureFormat;
	using gl::FramebufferBuilder;

	if (mResultFB[0].dimensions() != internalRes) {
		for (int i = 0; i < 2; i++) {
			mResultFB[i] = FramebufferBuilder(internalRes)
			               .addTexture(0, FBTextureFormat::RGBA_F16, FBTextureFiltering::LINEAR)
			               .addDepthTexture(FBDepthFormat::F32, FBTextureFiltering::NEAREST)
			               .build();
		}

		for (int i = 0; i < 2; i++) {
			mVelocityFB[i] = FramebufferBuilder(internalRes)
			                 .addTexture(0, FBTextureFormat::RGB_F16, FBTextureFiltering::LINEAR)
			                 .build();
		}
	}

	if (mTaaFB[0].dimensions() != internalRes) {
		for (int i = 0; i < 2; i++) {
			mTaaFB[i] = FramebufferBuilder(internalRes)
			            .addTexture(0, FBTextureFormat::RGBA_F16, FBTextureFiltering::LINEAR)
			            .addTexture(1, FBTextureFormat::RGBA_F16, FBTextureFiltering::LINEAR)
			            .build();
		}
	}

	if (mGammaCorrectedFB.dimensions() != internalRes) {
		mGammaCorrectedFB = FramebufferBuilder(internalRes)
		                    .addTexture(0, FBTextureFormat::RGBA_F16, FBTextureFiltering::LINEAR)
		                    .build();
	}
}

void GameScreen::reloadShaders() noexcept
{
	mTaaShader.reload();
	if (mTaaShader.isValid()) {
		glUseProgram(mTaaShader.handle());
		gl::setUniform(mTaaShader, "uSrcTexture", 0);
		gl::setUniform(mTaaShader, "uHistoryTexture", 1);
		gl::setUniform(mTaaShader, "uVelocityTexture", 2);
		gl::setUniform(mTaaShader, "uPrevVelocityTexture", 3);
	}

	mVelocityShader.reload();
	if (mVelocityShader.isValid()) {
		glUseProgram(mVelocityShader.handle());
		gl::setUniform(mVelocityShader, "uCurrDepthTexture", 0);
	}

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

void GameScreen::renderDebugUI() const noexcept
{
	using namespace ImGui;

	auto& cfg = GlobalConfig::instance();

	SetNextWindowPos(ImVec2(0, 0), ImGuiSetCond_FirstUseEver);
	SetNextWindowSize(ImVec2(400, 200), ImGuiSetCond_FirstUseEver);
	ShowMetricsWindow();

	SetNextWindowPos(ImVec2(0, 200), ImGuiSetCond_FirstUseEver);
	SetNextWindowSize(ImVec2(400, 460), ImGuiSetCond_FirstUseEver);
	SetNextWindowCollapsed(true, ImGuiSetCond_FirstUseEver);

	if (Begin("Config", nullptr, ImGuiWindowFlags_ShowBorders)) {
		DynArray<Setting*> settings;
		cfg.getSettings(settings);
		for (Setting* setting : settings) {

			// Each case has extra scope to be able to declare local variables inside
			switch (setting->type()) {
			case SettingType::INT:
				{
					int32_t tmp = setting->intValue();
					if (ImGui::InputInt(setting->key().str, &tmp)) {
						setting->setInt(tmp);
					}
				}
				break;
			case SettingType::FLOAT:
				{
					float tmp = setting->floatValue();
					if (ImGui::InputFloat(setting->key().str, &tmp)) {
						setting->setFloat(tmp);
					}
				}
				break;
			case SettingType::BOOL:
				{
					bool tmp = setting->boolValue();
					if (ImGui::Checkbox(setting->key().str, &tmp)) {
						setting->setBool(tmp);
					}
				}
				break;
			}
		}
	}
	End();

	Render();
}

} // namespace phe
