// Copyright (c) Peter Hillerström (skipifzero.com, peter@hstroem.se)

#pragma once

#include <sfz/gl/FullscreenQuad.hpp>
#include <sfz/gl/Program.hpp>
#include <sfz/screens/BaseScreen.hpp>

#include "renderers/BaseRenderer.hpp"
#include "renderers/ViewFrustum.hpp"
#include "resources/Renderable.hpp"

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
	virtual void onQuit() override final;
	virtual void onResize(vec2 dimensions, vec2 drawableDimensions) override final;

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

	void reloadShaders() noexcept;
	void updateEmulatedController(const DynArray<SDL_Event>& events, const sdl::Mouse& rawMouse) noexcept;

	// Private members
	// --------------------------------------------------------------------------------------------

	UniquePtr<BaseRenderer> mRendererPtr;
	EmulatedGameController mEmulatedController;
	ViewFrustum mCam;
	CameraMatrices mMatrices;
	DynArray<DrawOp> mDrawOps;
	Program mScalingShader;
	gl::FullscreenQuad mFullscreenQuad;

	// Temp
	Renderable mSponza;
	Renderable mSnakeRenderable;
};

} // namespace sfz
