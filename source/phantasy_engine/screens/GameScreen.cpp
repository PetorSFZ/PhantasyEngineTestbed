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
	this->renderer->setMaterialsAndTextures(this->level->materials, this->level->textures);
	this->renderer->setStaticScene(this->level->staticScene);
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

	// Update renderer matrices
	mMatrices.headMatrix = cam.viewMatrix();
	mMatrices.projMatrix = cam.projMatrix();
	mMatrices.position = cam.pos();
	mMatrices.forward = cam.dir();
	mMatrices.up = cam.up();
	mMatrices.vertFovRad = cam.verticalFov() * sfz::DEG_TO_RAD();
	if (renderer != nullptr) {
		renderer->updateMatrices(mMatrices);
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

	// Render the level
	RenderResult res = renderer->render(mResultFB, level->objects, level->sphereLights);

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

	if (showDebugUI) {
		// Change back to default clip space due to library incompatibility. This operation is
		// potentially expensive since it's intended to only be used once in the program.
		glClipControl(GL_LOWER_LEFT, GL_NEGATIVE_ONE_TO_ONE);

		renderDebugUI();

		// Reset to previous clip space
		glClipControl(GL_UPPER_LEFT, GL_ZERO_TO_ONE);
	}

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
