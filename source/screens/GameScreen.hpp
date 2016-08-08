// Copyright (c) Peter Hillerström (skipifzero.com, peter@hstroem.se)

#pragma once

#include <sfz/geometry/ViewFrustum.hpp>
#include <sfz/screens/BaseScreen.hpp>

#include "renderers/BaseRenderer.hpp"
#include "resources/Renderable.hpp"

namespace sfz {

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
	// Private members
	// --------------------------------------------------------------------------------------------

	ViewFrustum mCam;
	Renderable mSnakeRenderable;
	UniquePtr<BaseRenderer> mRendererPtr;
	CameraMatrices mMatrices;
	DynArray<DrawOp> mDrawOps;
};

} // namespace sfz
