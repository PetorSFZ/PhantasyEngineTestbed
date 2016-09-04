// Copyright (c) Peter Hillerström (skipifzero.com, peter@hstroem.se)

#pragma once

#include <sfz/screens/BaseScreen.hpp>

namespace sfz {

class LevelSelectScreen : public sfz::BaseScreen {
public:
	
	// Constructors & destructors
	// --------------------------------------------------------------------------------------------

	LevelSelectScreen() noexcept;

	// Overriden methods from sfz::BaseScreen
	// --------------------------------------------------------------------------------------------

	virtual UpdateOp update(UpdateState& state) override final;
	virtual void render(UpdateState& state) override final;
	virtual void onQuit() override final;
	virtual void onResize(vec2 dimensions, vec2 drawableDimensions) override final;
};

} // namespace sfz
