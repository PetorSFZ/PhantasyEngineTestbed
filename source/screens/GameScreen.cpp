// Copyright (c) Peter Hillerström (skipifzero.com, peter@hstroem.se)

#include "screens/GameScreen.hpp"

#include <sfz/gl/IncludeOpenGL.hpp>
#include <sfz/util/IO.hpp>

#include "renderers/DeferredRenderer.hpp"

namespace sfz {

// GameScreen: Constructors & destructors
// ------------------------------------------------------------------------------------------------

GameScreen::GameScreen() noexcept
{
	StackString128 modelsPath;
	modelsPath.printf("%sresources/models/", basePath());

	mCam = sfz::ViewFrustum(vec3(0.0f, 3.0f, -6.0f), normalize(vec3(0.0f, -0.25f, 1.0f)),
	                        normalize(vec3(0.0f, 1.0f, 0.0)), 60.0f, 1.0f, 0.01f, 100.0f);

	mRendererPtr = UniquePtr<BaseRenderer>(sfz_new<DeferredRenderer>());
	mDrawOps.ensureCapacity(8192);

	mSnakeRenderable = tinyObjLoadRenderable(modelsPath.str, "head_d2u_f2.obj");
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

	// Update renderer matrices
	mMatrices.headMatrix = mCam.viewMatrix();
	mMatrices.projMatrix = mCam.projMatrix();
	mRendererPtr->updateMatrices(mMatrices);

	return SCREEN_NO_OP;
}

void GameScreen::render(UpdateState& state)
{
	mDrawOps.clear();
	mDrawOps.add(DrawOp(identityMatrix4<float>(), &mSnakeRenderable));
	mDrawOps.add(DrawOp(yRotationMatrix4(1.0f), &mSnakeRenderable));
	mRendererPtr->render(mDrawOps);

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
