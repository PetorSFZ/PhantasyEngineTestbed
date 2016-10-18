// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include <sfz/gl/Program.hpp>
#include <sfz/screens/BaseScreen.hpp>

#include "phantasy_engine/config/GlobalConfig.hpp"
#include "phantasy_engine/level/Level.hpp"
#include "phantasy_engine/rendering/BaseRenderer.hpp"
#include "phantasy_engine/rendering/ViewFrustum.hpp"
#include "phantasy_engine/rendering/FullscreenTriangle.hpp"

namespace phe {

using sfz::gl::Framebuffer;
using sfz::gl::Program;
using sfz::UniquePtr;
using sfz::UpdateOp;
using sfz::UpdateState;
using sfz::UpdateOpType;
using sfz::vec2i;

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

	SharedPtr<GameLogic> gameLogic;
	SharedPtr<Level> level;
	SharedPtr<BaseRenderer> renderer;

	// TODO: Come up with something better than these?
	ViewFrustum cam;

	bool debugUIActive = false;

	// Constructors & destructors
	// --------------------------------------------------------------------------------------------

	GameScreen() = delete;
	GameScreen(const GameScreen&) = delete;
	GameScreen& operator= (const GameScreen&) = delete;
	GameScreen(GameScreen&&) = delete;
	GameScreen& operator= (GameScreen&&) = delete;

	GameScreen(SharedPtr<GameLogic> gameLogic, SharedPtr<Level> level,
	           SharedPtr<BaseRenderer> renderer) noexcept;

	~GameScreen() noexcept;

	// Overriden methods from sfz::BaseScreen
	// --------------------------------------------------------------------------------------------

	virtual UpdateOp update(UpdateState& state) override final;
	virtual void render(UpdateState& state) override final;

	// Public methods
	// --------------------------------------------------------------------------------------------

	void setRenderer(const SharedPtr<BaseRenderer>& shared) noexcept;

	private:
	// Private methods
	// --------------------------------------------------------------------------------------------

	void targetResolutionUpdated(vec2i targetRes) noexcept;
	void reloadShaders() noexcept;
	void resetTAA() noexcept;
	void renderDebugUI() const noexcept;

	// Private members
	// --------------------------------------------------------------------------------------------

	uint32_t mPreviousDepthTexture = 0;
	Framebuffer mGammaCorrectedFB;
	Framebuffer mVelocityFB[2];
	Framebuffer mTaaFB[2];

	Program mScalingShader, mGammaCorrectionShader, mVelocityShader, mTaaShader;

	uint32_t mFBIndex = 0;
	uint32_t mHaltonIndex = 0;

	ViewFrustum mPreviousCamera;
	GraphicsConfigValues mPreviousGraphicsConfig;

	FullscreenTriangle mFullscreenTriangle;
};

} // namespace phe
