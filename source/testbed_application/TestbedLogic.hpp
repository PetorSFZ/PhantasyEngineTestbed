#pragma once

#include <phantasy_engine/screens/GameScreen.hpp>

namespace sfz {

using sdl::ButtonState;

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
		sdl::GameControllerState state;
		ButtonState leftStickUp = ButtonState::NOT_PRESSED;
		ButtonState leftStickDown = ButtonState::NOT_PRESSED;
		ButtonState leftStickLeft = ButtonState::NOT_PRESSED;
		ButtonState leftStickRight = ButtonState::NOT_PRESSED;
		ButtonState shiftPressed = ButtonState::NOT_PRESSED;
	};

	// Private methods
	// --------------------------------------------------------------------------------------------

	void updateEmulatedController(const DynArray<SDL_Event>& events, const sdl::Mouse& rawMouse) noexcept;

	// Private members
	// --------------------------------------------------------------------------------------------

	EmulatedGameController mEmulatedController;
};

} // namespace sfz
