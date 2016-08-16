// Copyright (c) Peter Hillerstr�m (skipifzero.com, peter@hstroem.se)

#include "screens/LevelSelectScreen.hpp"
#include "screens/GameScreen.hpp"

namespace sfz {

// LevelSelectScreen: Constructors & destructors
// ------------------------------------------------------------------------------------------------

LevelSelectScreen::LevelSelectScreen() noexcept
{
	// TODO: Well. Should probably do something here.
}

// LevelSelectScreen: Overriden methods from sfz::BaseScreen
// ------------------------------------------------------------------------------------------------

UpdateOp LevelSelectScreen::update(UpdateState& state)
{
	return UpdateOp(UpdateOpType::SWITCH_SCREEN, SharedPtr<BaseScreen>(sfz_new<GameScreen>()));
}

void LevelSelectScreen::render(UpdateState& state)
{

}

void LevelSelectScreen::onQuit()
{

}

void LevelSelectScreen::onResize(vec2 dimensions, vec2 drawableDimensions)
{

}

} // namespace sfz