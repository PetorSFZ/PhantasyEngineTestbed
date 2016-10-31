// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include <phantasy_engine/screens/GameScreen.hpp>
#include <phantasy_engine/level/Level.hpp>
#include <sfz/containers/HashMap.hpp>
#include <sfz/math/Vector.hpp>

#include "Helpers.hpp"

using phe::GameLogic;
using phe::GameScreen;
using sfz::DynArray;
using sfz::HashMap;
using sfz::sdl::ButtonState;
using sfz::sdl::Mouse;
using sfz::UpdateOp;
using sfz::UpdateOpType;
using sfz::UpdateState;
using sfz::vec3;

// TestbedLogic
// ------------------------------------------------------------------------------------------------

class TestbedLogic final : public GameLogic {
public:
	
	// Constructors & destructors
	// --------------------------------------------------------------------------------------------

	TestbedLogic(DynArray<RendererAndStatus>&& renderers, uint32_t rendererIndex, phe::Level& level) noexcept;

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

	DynArray<RendererAndStatus> mRenderers;
	uint32_t mCurrentRenderer;

	DynArray<uint32_t> instanceHandles;
	DynArray<uint32_t> movingInstanceHandles;
	DynArray<uint32_t> nonmovingInstanceHandles;
	HashMap<uint32_t, vec3> objectPositions;
};
