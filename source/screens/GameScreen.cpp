// Copyright (c) Peter Hillerström (skipifzero.com, peter@hstroem.se)

#include "screens/GameScreen.hpp"

#include <chrono>

#include <sfz/gl/IncludeOpenGL.hpp>
#include <sfz/util/IO.hpp>
#include <sfz/math/Vector.hpp>

#include "config/GlobalConfig.hpp"
#include "renderers/cuda_ray_tracer/CudaRayTracerRenderer.hpp"
#include "renderers/DeferredRenderer.hpp"

namespace sfz {

using sdl::ButtonState;
using sdl::GameControllerState;

// GameScreen: Constructors & destructors
// ------------------------------------------------------------------------------------------------

GameScreen::GameScreen() noexcept
{
	auto& cfg = GlobalConfig::instance();

	mCam = ViewFrustum(vec3(0.0f, 3.0f, -6.0f), normalize(vec3(0.0f, -0.25f, 1.0f)),
	                   normalize(vec3(0.0f, 1.0f, 0.0)), 60.0f, 1.0f, 0.01f, 10000.0f);

	if (cfg.graphcisCfg().renderingBackend->intValue() == 0) {
		mRendererPtr = UniquePtr<BaseRenderer>(sfz_new<DeferredRenderer>());
	} else {
		mRendererPtr = UniquePtr<BaseRenderer>(sfz_new<CUDARayTracerRenderer>());
	}

	mDrawOps.ensureCapacity(8192);
	
	// Load shaders
	StackString192 shadersPath;
	shadersPath.printf("%sresources/shaders/", basePath());

	mScalingShader = Program::postProcessFromFile(shadersPath.str, "scaling.frag");
	glUseProgram(mScalingShader.handle());
	gl::setUniform(mScalingShader, "uSrcTexture", 0);

	// Load models
	StackString192 modelsPath;
	modelsPath.printf("%sresources/models/", basePath());

	using time_point = std::chrono::high_resolution_clock::time_point;
	time_point before = std::chrono::high_resolution_clock::now();
	
	mSponza = assimpLoadSponza(modelsPath.str, "sponzaPBR/sponzaPBR.obj");

	time_point after = std::chrono::high_resolution_clock::now();
	using FloatSecond = std::chrono::duration<float>;
	float delta = std::chrono::duration_cast<FloatSecond>(after - before).count();
	printf("Time spent loading models: %.3f seconds\n", delta);

	// Add the sponza model to the scene
	scene.staticRenderables.add(std::move(mSponza));

	// Add lights to the scene
	vec3 colours[]{
		vec3{ 1.0f, 0.0f, 0.0f },
		vec3{ 1.0f, 0.0f, 1.0f },
		vec3{ 0.0f, 1.0f, 1.0f },
		vec3{ 1.0f, 1.0f, 0.0f },
		vec3{ 0.0f, 1.0f, 0.0f }
	};
	for (int i = 0; i < 5; i++) {
		PointLight pointLight;
		pointLight.pos = vec3{ -50.0f + 25.0f * i , 5.0f, 0.0f };
		pointLight.radius = 50.0f;
		pointLight.strength = 100.0f * colours[i];
		scene.staticPointLights.add(pointLight);
	}	
}

// GameScreen: Overriden methods from sfz::BaseScreen
// ------------------------------------------------------------------------------------------------

UpdateOp GameScreen::update(UpdateState& state)
{
	using sdl::GameController;

	// Handle input
	/*for (const SDL_Event& event : state.events) {
		switch (event.type) {
		case SDL_QUIT: return SCREEN_QUIT;
		case SDL_KEYUP:
			switch (event.key.keysym.sym) {
			case SDLK_ESCAPE: return SCREEN_QUIT;
			}
			break;
		}
	}*/

	updateEmulatedController(state.events, state.rawMouse);
	uint32_t controllerIndex = 0;
	GameController* controller = state.controllers.get(controllerIndex);
	const GameControllerState& ctrl = (controller != nullptr) ? controller->state() : mEmulatedController.state;
	
	
	float currentSpeed = 10.0f;
	float turningSpeed = 1.25f * PI();

	// Triggers
	if (ctrl.leftTrigger > ctrl.triggerDeadzone) {
		currentSpeed += (ctrl.leftTrigger * 25.0f);
	}
	if (ctrl.rightTrigger > ctrl.triggerDeadzone) {
		
	}

	// Analogue Sticks
	if (length(ctrl.rightStick) > ctrl.stickDeadzone) {
		vec3 right = normalize(cross(mCam.dir(), mCam.up()));
		mat3 xTurn = rotationMatrix3(vec3{0.0f, -1.0f, 0.0f}, ctrl.rightStick[0] * turningSpeed * state.delta);
		mat3 yTurn = rotationMatrix3(right, ctrl.rightStick[1] * turningSpeed * state.delta);
		mCam.setDir(yTurn * xTurn * mCam.dir(), yTurn * xTurn * mCam.up());
	}
	if (length(ctrl.leftStick) > ctrl.stickDeadzone) {
		vec3 right = normalize(cross(mCam.dir(), mCam.up()));
		mCam.setPos(mCam.pos() + ((mCam.dir() * ctrl.leftStick[1] + right * ctrl.leftStick[0]) * currentSpeed * state.delta));
	}

	// Control Pad
	if (ctrl.padUp == ButtonState::DOWN) {

	} else if (ctrl.padDown == ButtonState::DOWN) {

	} else if (ctrl.padLeft == ButtonState::DOWN) {

	} else if (ctrl.padRight == ButtonState::DOWN) {

	}

	// Shoulder buttons
	if (ctrl.leftShoulder == ButtonState::DOWN || ctrl.leftShoulder == ButtonState::HELD) {
		mCam.setPos(mCam.pos() - vec3(0.0f, 1.0f, 0.0f) * currentSpeed * state.delta);
	} else if (ctrl.rightShoulder == ButtonState::DOWN || ctrl.rightShoulder == ButtonState::HELD) {
		mCam.setPos(mCam.pos() + vec3(0.0f, 1.0f, 0.0f) * currentSpeed * state.delta);
	}

	// Face buttons
	if (ctrl.y == ButtonState::UP) {
	}
	if (ctrl.x == ButtonState::UP) {
	}
	if (ctrl.b == ButtonState::UP) {
	}
	if (ctrl.a == ButtonState::UP) {
	}

	// Menu buttons
	if (ctrl.back == ButtonState::UP) {
		return SCREEN_QUIT;
	}

	mCam.setDir(mCam.dir(), vec3(0.0f, 1.0f, 0.0f));



	// Update renderer matrices
	mMatrices.headMatrix = mCam.viewMatrix();
	mMatrices.projMatrix = mCam.projMatrix();
	mRendererPtr->updateMatrices(mMatrices);

	return SCREEN_NO_OP;
}

void GameScreen::render(UpdateState& state)
{
	vec2i drawableDim = state.window.drawableDimensions();

	if (drawableDim.x > 0 && drawableDim.y > 0) {
		mCam.setAspectRatio(float(drawableDim.x) / float(drawableDim.y));
	}

	if (drawableDim != mRendererPtr->resolution()) {
		mRendererPtr->setMaxResolution(drawableDim);
		mRendererPtr->setResolution(drawableDim);
	}

	mDrawOps.clear();

	// Render all static objects in the scene, scaled down to 5% of their sizes
	for (Renderable& renderable : scene.staticRenderables) {
		mDrawOps.add(DrawOp(scalingMatrix4<float>(0.05f), &renderable));
	}
	mDrawOps.add(DrawOp(identityMatrix4<float>(), &mSnakeRenderable));
	mRendererPtr->render(mDrawOps, scene.staticPointLights);

	// Scale result to screen
	glDisable(GL_BLEND);
	glEnable(GL_CULL_FACE);
	glDisable(GL_DEPTH_TEST);

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glViewport(0, 0, drawableDim.x, drawableDim.y);
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClearDepth(0.0f); // Reverse depth buffer
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glUseProgram(mScalingShader.handle());
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, mRendererPtr->getResult().texture(0));
	gl::setUniform(mScalingShader, "uViewportRes", vec2(mRendererPtr->resolution()));
	gl::setUniform(mScalingShader, "uDstRes", vec2(drawableDim));

	mFullscreenQuad.render();

	SDL_GL_SwapWindow(state.window.ptr());
}

void GameScreen::onQuit()
{

}

void GameScreen::onResize(vec2 dimensions, vec2 drawableDimensions)
{
	//this->mRendererPtr->setMaxResolution(vec2i(drawableDimensions));
	//this->mRendererPtr->setResolution(vec2i(drawableDimensions));
}

// GameScreen: Private methods
// ------------------------------------------------------------------------------------------------

void GameScreen::reloadShaders() noexcept
{
	mScalingShader.reload();
	if (mScalingShader.isValid()) {
		glUseProgram(mScalingShader.handle());
		gl::setUniform(mScalingShader, "uSrcTexture", 0);
	}
}

void GameScreen::updateEmulatedController(const DynArray<SDL_Event>& events,
                                          const sdl::Mouse& rawMouse) noexcept
{
	GameControllerState& c = mEmulatedController.state;

	// Changes previous DOWN state to HELD state.

	if (c.a == ButtonState::DOWN) c.a = ButtonState::HELD;
	if (c.b == ButtonState::DOWN) c.b = ButtonState::HELD;
	if (c.x == ButtonState::DOWN) c.x = ButtonState::HELD;
	if (c.y == ButtonState::DOWN) c.y = ButtonState::HELD;

	if (c.leftShoulder == ButtonState::DOWN) c.leftShoulder = ButtonState::HELD;
	if (c.rightShoulder == ButtonState::DOWN) c.rightShoulder = ButtonState::HELD;
	if (c.leftStickButton == ButtonState::DOWN) c.leftStickButton = ButtonState::HELD;
	if (c.rightStickButton == ButtonState::DOWN) c.rightStickButton = ButtonState::HELD;

	if (c.padUp == ButtonState::DOWN) c.padUp = ButtonState::HELD;
	if (c.padDown == ButtonState::DOWN) c.padDown = ButtonState::HELD;
	if (c.padLeft == ButtonState::DOWN) c.padLeft = ButtonState::HELD;
	if (c.padRight == ButtonState::DOWN) c.padRight = ButtonState::HELD;

	if (c.start == ButtonState::DOWN) c.start = ButtonState::HELD;
	if (c.back == ButtonState::DOWN) c.back = ButtonState::HELD;
	if (c.guide == ButtonState::DOWN) c.guide = ButtonState::HELD;

	if (mEmulatedController.leftStickDown == ButtonState::DOWN) mEmulatedController.leftStickDown = ButtonState::HELD;
	if (mEmulatedController.leftStickUp == ButtonState::DOWN) mEmulatedController.leftStickUp = ButtonState::HELD;
	if (mEmulatedController.leftStickLeft == ButtonState::DOWN) mEmulatedController.leftStickLeft = ButtonState::HELD;
	if (mEmulatedController.leftStickRight == ButtonState::DOWN) mEmulatedController.leftStickRight = ButtonState::HELD;
	if (mEmulatedController.shiftPressed == ButtonState::DOWN) mEmulatedController.shiftPressed = ButtonState::HELD;

	// Changes previous UP state to NOT_PRESSED state.

	if (c.a == ButtonState::UP) c.a = ButtonState::NOT_PRESSED;
	if (c.b == ButtonState::UP) c.b = ButtonState::NOT_PRESSED;
	if (c.x == ButtonState::UP) c.x = ButtonState::NOT_PRESSED;
	if (c.y == ButtonState::UP) c.y = ButtonState::NOT_PRESSED;

	if (c.leftShoulder == ButtonState::UP) c.leftShoulder = ButtonState::NOT_PRESSED;
	if (c.rightShoulder == ButtonState::UP) c.rightShoulder = ButtonState::NOT_PRESSED;
	if (c.leftStickButton == ButtonState::UP) c.leftStickButton = ButtonState::NOT_PRESSED;
	if (c.rightStickButton == ButtonState::UP) c.rightStickButton = ButtonState::NOT_PRESSED;

	if (c.padUp == ButtonState::UP) c.padUp = ButtonState::NOT_PRESSED;
	if (c.padDown == ButtonState::UP) c.padDown = ButtonState::NOT_PRESSED;
	if (c.padLeft == ButtonState::UP) c.padLeft = ButtonState::NOT_PRESSED;
	if (c.padRight == ButtonState::UP) c.padRight = ButtonState::NOT_PRESSED;

	if (c.start == ButtonState::UP) c.start = ButtonState::NOT_PRESSED;
	if (c.back == ButtonState::UP) c.back = ButtonState::NOT_PRESSED;
	if (c.guide == ButtonState::UP) c.guide = ButtonState::NOT_PRESSED;

	if (mEmulatedController.leftStickDown == ButtonState::UP) mEmulatedController.leftStickDown = ButtonState::NOT_PRESSED;
	if (mEmulatedController.leftStickUp == ButtonState::UP) mEmulatedController.leftStickUp = ButtonState::NOT_PRESSED;
	if (mEmulatedController.leftStickLeft == ButtonState::UP) mEmulatedController.leftStickLeft = ButtonState::NOT_PRESSED;
	if (mEmulatedController.leftStickRight == ButtonState::UP) mEmulatedController.leftStickRight = ButtonState::NOT_PRESSED;
	if (mEmulatedController.shiftPressed == ButtonState::UP) mEmulatedController.shiftPressed = ButtonState::NOT_PRESSED;

	// Check events from SDL
	
	for (const SDL_Event& event : events) {
		switch (event.type) {
		case SDL_KEYDOWN:
			switch (event.key.keysym.sym) {
			case 'w':
			case 'W':
				mEmulatedController.leftStickUp = ButtonState::DOWN;
				break;
			case 'a':
			case 'A':
				mEmulatedController.leftStickLeft = ButtonState::DOWN;
				break;
			case 's':
			case 'S':
				mEmulatedController.leftStickDown = ButtonState::DOWN;
				break;
			case 'd':
			case 'D':
				mEmulatedController.leftStickRight = ButtonState::DOWN;
				break;
			case SDLK_LSHIFT:
			case SDLK_RSHIFT:
				mEmulatedController.shiftPressed = ButtonState::DOWN;
				break;
			case 'q':
			case 'Q':
				c.leftShoulder = ButtonState::DOWN;
				break;
			case 'e':
			case 'E':
				c.rightShoulder = ButtonState::DOWN;
				break;
			case SDLK_ESCAPE:
				c.back = ButtonState::DOWN;
				break;
			}
			break;
		case SDL_KEYUP:
			switch (event.key.keysym.sym) {
			case 'w':
			case 'W':
				mEmulatedController.leftStickUp = ButtonState::UP;
				break;
			case 'a':
			case 'A':
				mEmulatedController.leftStickLeft = ButtonState::UP;
				break;
			case 's':
			case 'S':
				mEmulatedController.leftStickDown = ButtonState::UP;
				break;
			case 'd':
			case 'D':
				mEmulatedController.leftStickRight = ButtonState::UP;
				break;
			case SDLK_LSHIFT:
			case SDLK_RSHIFT:
				mEmulatedController.shiftPressed = ButtonState::UP;
				break;
			case 'q':
			case 'Q':
				c.leftShoulder = ButtonState::UP;
				break;
			case 'e':
			case 'E':
				c.rightShoulder = ButtonState::UP;
				break;
			case SDLK_ESCAPE:
				c.back = ButtonState::UP;
				break;
			}
			break;
		}
	}

	// Set left stick
	vec2 leftStick = vec2(0.0f);
	if (mEmulatedController.leftStickUp != ButtonState::NOT_PRESSED) leftStick.y = 1.0f;
	else if (mEmulatedController.leftStickDown != ButtonState::NOT_PRESSED) leftStick.y = -1.0f;
	if (mEmulatedController.leftStickLeft != ButtonState::NOT_PRESSED) leftStick.x = -1.0f;
	else if (mEmulatedController.leftStickRight != ButtonState::NOT_PRESSED) leftStick.x = 1.0f;
	
	leftStick = safeNormalize(leftStick);
	if (mEmulatedController.shiftPressed != ButtonState::NOT_PRESSED) leftStick *= 0.5f;

	mEmulatedController.state.leftStick = leftStick;

	// Set right stick
	mEmulatedController.state.rightStick = rawMouse.motion * 200.0f;

	const uint8_t* keys = SDL_GetKeyboardState(nullptr);
	vec2 arrowKeyVector(keys[SDL_SCANCODE_RIGHT] - keys[SDL_SCANCODE_LEFT], keys[SDL_SCANCODE_UP] - keys[SDL_SCANCODE_DOWN]);
	mEmulatedController.state.rightStick += 0.7f * arrowKeyVector;

	// Set triggers
	if (rawMouse.leftButton == ButtonState::NOT_PRESSED) {
		mEmulatedController.state.rightTrigger = 0.0f;
	} else {
		mEmulatedController.state.rightTrigger = 1.0f;
	}
	if (rawMouse.rightButton == ButtonState::NOT_PRESSED) {
		mEmulatedController.state.leftTrigger = 0.0f;
	} else {
		mEmulatedController.state.leftTrigger = 1.0f;
	}
}

} // namespace sfz
