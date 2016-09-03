// Copyright (c) Peter Hillerström (skipifzero.com, peter@hstroem.se)

#pragma once

#include <sfz/gl/Program.hpp>
#include <sfz/screens/BaseScreen.hpp>

#include "renderers/BaseRenderer.hpp"
#include "renderers/ViewFrustum.hpp"
#include "renderers/FullscreenTriangle.hpp"
#include "resources/Renderable.hpp"

#include "level/Scene.hpp"

namespace sfz {

using gl::Program;
using sdl::ButtonState;

class GameScreen : public sfz::BaseScreen {
public:
	// Constructors & destructors
	// --------------------------------------------------------------------------------------------

	GameScreen() noexcept;

	// Overriden methods from sfz::BaseScreen
	// --------------------------------------------------------------------------------------------

	virtual UpdateOp update(UpdateState& state) override final;
	virtual void render(UpdateState& state) override final;

private:
	// Private structs
	// --------------------------------------------------------------------------------------------

	struct EmulatedGameController {
		sdl::GameControllerState state;
		ButtonState leftStickUp = ButtonState::NOT_PRESSED;
		ButtonState leftStickDown = ButtonState::NOT_PRESSED;
		ButtonState leftStickLeft = ButtonState::NOT_PRESSED;
		ButtonState leftStickRight = ButtonState::NOT_PRESSED;
		ButtonState shiftPressed = ButtonState::NOT_PRESSED;
	};
	
	// Private methods
	// --------------------------------------------------------------------------------------------

	void reloadFramebuffers(vec2i maxResolution) noexcept;
	void reloadShaders() noexcept;
	void updateEmulatedController(const DynArray<SDL_Event>& events, const sdl::Mouse& rawMouse) noexcept;

	// Private members
	// --------------------------------------------------------------------------------------------

	UniquePtr<BaseRenderer> mRendererPtr;
	EmulatedGameController mEmulatedController;
	ViewFrustum mCam;
	CameraMatrices mMatrices;
	DynArray<DrawOp> mDrawOps;

	Framebuffer mGammaCorrectedFB;
	Program mScalingShader, mGammaCorrectionShader;
	FullscreenTriangle mFullscreenTriangle;

	// Temp
	Renderable mSponza;
	Renderable mSnakeRenderable;

	Scene scene;
};

} // namespace sfz
