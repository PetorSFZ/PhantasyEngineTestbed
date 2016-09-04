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

// GameScreen updatable
// ------------------------------------------------------------------------------------------------

class GameScreen; // Forward declare

class GameLogic {
public:
	virtual UpdateOp update(GameScreen& screen, UpdateState& state) noexcept = 0;
};

// GameScreen
// ------------------------------------------------------------------------------------------------

class GameScreen final : public sfz::BaseScreen {
public:
	// Public members
	// --------------------------------------------------------------------------------------------

	UniquePtr<GameLogic> gameLogic;
	UniquePtr<BaseRenderer> renderer;

	ViewFrustum cam;
	CameraMatrices matrices;

	// Constructors & destructors
	// --------------------------------------------------------------------------------------------

	GameScreen() = delete;
	GameScreen(const GameScreen&) = delete;
	GameScreen& operator= (const GameScreen&) = delete;
	GameScreen(GameScreen&&) = delete;
	GameScreen& operator= (GameScreen&&) = delete;

	GameScreen(UniquePtr<GameLogic>&& gameLogic, UniquePtr<BaseRenderer>&& renderer) noexcept;

	// Overriden methods from sfz::BaseScreen
	// --------------------------------------------------------------------------------------------

	virtual UpdateOp update(UpdateState& state) override final;
	virtual void render(UpdateState& state) override final;

private:
	// Private methods
	// --------------------------------------------------------------------------------------------

	void reloadFramebuffers(vec2i maxResolution) noexcept;
	void reloadShaders() noexcept;

	// Private members
	// --------------------------------------------------------------------------------------------

	DynArray<DrawOp> mDrawOps;

	Framebuffer mResultFB, mGammaCorrectedFB;
	Program mScalingShader, mGammaCorrectionShader;
	FullscreenTriangle mFullscreenTriangle;

	// Temp
	Renderable mSponza;
	Renderable mSnakeRenderable;

	Scene scene;
};

} // namespace sfz
