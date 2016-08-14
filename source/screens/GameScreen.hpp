// Copyright (c) Peter Hillerström (skipifzero.com, peter@hstroem.se)

#pragma once

#include <sfz/geometry/ViewFrustum.hpp>
#include <sfz/screens/BaseScreen.hpp>

#include "renderers/BaseRenderer.hpp"
#include "resources/Renderable.hpp"

namespace sfz {

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
		sdl::ButtonState leftStickUp = ButtonState::NOT_PRESSED;
		sdl::ButtonState leftStickDown = ButtonState::NOT_PRESSED;
		sdl::ButtonState leftStickLeft = ButtonState::NOT_PRESSED;
		sdl::ButtonState leftStickRight = ButtonState::NOT_PRESSED;
		sdl::ButtonState shiftPressed = ButtonState::NOT_PRESSED;
	};
	
	// Private methods
	// --------------------------------------------------------------------------------------------

	void updateEmulatedController(const DynArray<SDL_Event>& events, const sdl::Mouse& rawMouse) noexcept;

	// Private members
	// --------------------------------------------------------------------------------------------

	UniquePtr<BaseRenderer> mRendererPtr;
	EmulatedGameController mEmulatedController;
	ViewFrustum mCam;
	CameraMatrices mMatrices;
	DynArray<DrawOp> mDrawOps;

	// Temp
	DynArray<Renderable> mSponza;
	Renderable mSnakeRenderable;
};

} // namespace sfz
