// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include <phantasy_engine/screens/GameScreen.hpp>

using phe::GameLogic;
using phe::GameScreen;
using sfz::DynArray;
using sfz::sdl::ButtonState;
using sfz::sdl::Mouse;
using sfz::UpdateOp;
using sfz::UpdateOpType;
using sfz::UpdateState;

// TestbedLogic
// ------------------------------------------------------------------------------------------------

class TestbedLogic final : public GameLogic {
public:
	
	// Overriden methods from GameLogic
	// --------------------------------------------------------------------------------------------

	UpdateOp update(GameScreen& screen, UpdateState& state) noexcept override final;

private:
	// Private structs
	// --------------------------------------------------------------------------------------------

	struct EmulatedGameController {
		sfz::sdl::GameControllerState state;
		ButtonState leftStickUp = ButtonState::NOT_PRESSED;
		ButtonState leftStickDown = ButtonState::NOT_PRESSED;
		ButtonState leftStickLeft = ButtonState::NOT_PRESSED;
		ButtonState leftStickRight = ButtonState::NOT_PRESSED;
		ButtonState shiftPressed = ButtonState::NOT_PRESSED;
	};

	// Private methods
	// --------------------------------------------------------------------------------------------

	void updateEmulatedController(const DynArray<SDL_Event>& events, const Mouse& rawMouse) noexcept;

	// Private members
	// --------------------------------------------------------------------------------------------

	EmulatedGameController mEmulatedController;
};
