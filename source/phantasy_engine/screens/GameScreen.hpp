// Copyright (c) Peter Hillerström (skipifzero.com, peter@hstroem.se)

#pragma once

#include <sfz/gl/Program.hpp>
#include <sfz/screens/BaseScreen.hpp>

#include "level/Level.hpp"
#include "renderers/BaseRenderer.hpp"
#include "renderers/ViewFrustum.hpp"
#include "renderers/FullscreenTriangle.hpp"

namespace sfz {

using gl::Program;

// GameScreen updatable
// ------------------------------------------------------------------------------------------------

class GameScreen; // Forward declare

/// Class responsible for handling the update() step of a GameScreen. It is free to do anything
/// that a normal sfz::Screen is allowed to do, such as changing screens, exiting the game loop,
/// modify members of GameScreen, etc.
class GameLogic {
public:
	virtual UpdateOp update(GameScreen& screen, UpdateState& state) noexcept = 0;
};

// GameScreen
// ------------------------------------------------------------------------------------------------

/// The component that ties PhantasyEngine together. A GameScreen has three major components,
/// a game logic component, a level and a renderer. It is responsible for applying the game logic
/// every frame and then rendering the level.
class GameScreen final : public sfz::BaseScreen {
public:
	// Public members
	// --------------------------------------------------------------------------------------------

	UniquePtr<GameLogic> gameLogic;
	UniquePtr<Level> level;
	UniquePtr<BaseRenderer> renderer;

	// TODO: Come up with something better than these?
	ViewFrustum cam;
	CameraMatrices matrices;

	// Constructors & destructors
	// --------------------------------------------------------------------------------------------

	GameScreen() = delete;
	GameScreen(const GameScreen&) = delete;
	GameScreen& operator= (const GameScreen&) = delete;
	GameScreen(GameScreen&&) = delete;
	GameScreen& operator= (GameScreen&&) = delete;

	GameScreen(UniquePtr<GameLogic>&& gameLogic, UniquePtr<Level>&& level,
	           UniquePtr<BaseRenderer>&& renderer) noexcept;

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
};

} // namespace sfz
