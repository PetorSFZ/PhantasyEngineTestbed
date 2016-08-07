// Copyright (c) Peter Hillerström (skipifzero.com, peter@hstroem.se)

#include "screens/GameScreen.hpp"

#include <sfz/gl/IncludeOpenGL.hpp>

#include "renderers/DeferredRenderer.hpp"

namespace sfz {

// GameScreen: Constructors & destructors
// ------------------------------------------------------------------------------------------------

GameScreen::GameScreen() noexcept
{
	mRendererPtr = SharedPtr<BaseRenderer>(sfz_new<DeferredRenderer>());
}

// GameScreen: Overriden methods from sfz::BaseScreen
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
	mat4 viewMatrix = lookAt(vec3(0.0f), vec3(1.0f, 0.0f, 0.0f), vec3(0.0f, 1.0f, 0.0f));
	mat4 projMatrix = perspectiveProjectionMatrix(60.0f, 1.0f, 0.01f, 100.0f);
	mRendererPtr->render(viewMatrix, projMatrix);

	/*glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClearDepth(1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// Run scaler here*/

	SDL_GL_SwapWindow(state.window.ptr());
}

void GameScreen::onQuit()
{

}

void GameScreen::onResize(vec2 dimensions, vec2 drawableDimensions)
{

}

} // namespace sfz
