// Copyright (c) Peter Hillerström (skipifzero.com, peter@hstroem.se)

#include "screens/GameScreen.hpp"

namespace sfz {

// GameScreen: Constructors & destructors
// ------------------------------------------------------------------------------------------------

GameScreen::GameScreen() noexcept
{
	// TODO: Well. Should probably do something here.
}

// LevelSelectScreen: Overriden methods from sfz::BaseScreen
// ------------------------------------------------------------------------------------------------

UpdateOp GameScreen::update(UpdateState& state)
{
	// Handle input
	for (const SDL_Event& event : state.events) {
		switch (event.type) {
		case SDL_QUIT: return SCREEN_QUIT;
		case SDL_KEYUP:
			switch (event.key.keysym.sym) {
			case SDLK_ESCAPE: return SCREEN_QUIT;
			}
			break;
		}
	}

	return SCREEN_NO_OP;
}

void GameScreen::render(UpdateState& state)
{

}

void GameScreen::onQuit()
{

}

void GameScreen::onResize(vec2 dimensions, vec2 drawableDimensions)
{

}

} // namespace sfz
