// Copyright (c) Peter Hillerström (skipifzero.com, peter@hstroem.se)

#include "screens/GameScreen.hpp"

#include <chrono>

#include <sfz/gl/IncludeOpenGL.hpp>
#include <sfz/util/IO.hpp>
#include <sfz/math/Vector.hpp>

#include "config/GlobalConfig.hpp"
#include "renderers/DeferredRenderer.hpp"
#include "renderers/cpu_ray_tracer/CPURayTracerRenderer.hpp"

#ifdef CUDA_TRACER_AVAILABLE
//#include "renderers/cuda_ray_tracer/CudaRayTracerRenderer.hpp"
#endif

//#define DEV_NOT_USING_SPONZA

namespace sfz {

using sdl::ButtonState;
using sdl::GameControllerState;

// GameScreen: Constructors & destructors
// ------------------------------------------------------------------------------------------------

GameScreen::GameScreen(UniquePtr<GameLogic>&& gameLogic, UniquePtr<BaseRenderer>&& renderer) noexcept
:
	gameLogic(std::move(gameLogic)),
	renderer(std::move(renderer))
{
	auto& cfg = GlobalConfig::instance();

	cam = ViewFrustum(vec3(0.0f, 3.0f, -6.0f), normalize(vec3(0.0f, -0.25f, 1.0f)),
	                  normalize(vec3(0.0f, 1.0f, 0.0)), 60.0f, 1.0f, 0.01f, 10000.0f);

	mDrawOps.ensureCapacity(8192);
	
	// Load shaders
	StackString192 shadersPath;
	shadersPath.printf("%sresources/shaders/", basePath());

	mScalingShader = Program::postProcessFromFile(shadersPath.str, "scaling.frag");
	glUseProgram(mScalingShader.handle());
	gl::setUniform(mScalingShader, "uSrcTexture", 0);

	mGammaCorrectionShader = Program::postProcessFromFile(shadersPath.str, "gamma_correction.frag");
	glUseProgram(mGammaCorrectionShader.handle());
	gl::setUniform(mGammaCorrectionShader, "uLinearTexture", 0);

	// Load models
	StackString192 modelsPath;
	modelsPath.printf("%sresources/models/", basePath());

	using time_point = std::chrono::high_resolution_clock::time_point;
	time_point before = std::chrono::high_resolution_clock::now();
	
#ifndef DEV_NOT_USING_SPONZA
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
		pointLight.range = 50.0f;
		pointLight.strength = 100.0f * colours[i];
		scene.staticPointLights.add(pointLight);
	}
#else
	Renderable testRenderable;
	RenderableComponent testComponent;
	Vertex v1;
	v1.pos = {-0.5f, 2.0f, -2.0f};
	v1.normal = {0.0f, 0.0f, 0.0f};
	v1.uv = {0.0f, 1.0f};

	Vertex v2;
	v2.pos = {0.0f, 2.0f, -2.0f};
	v2.uv = {1.0f, 1.0f};

	Vertex v3;
	v3.pos = {0.0f, 2.5f, -2.0f};
	v3.uv = {1.0f, 0.0f};

	Vertex testVertices[3] = {v1, v2, v3};
	uint32_t indices[3] = {0, 1, 2};
	testComponent.geometry.vertices.add(testVertices, 3);
	testComponent.geometry.indices.add(indices, 3);
	testComponent.glModel.load(testComponent.geometry);

	testRenderable.components.add(std::move(testComponent));
	scene.staticRenderables.add(std::move(testRenderable));
#endif

	this->renderer->prepareForScene(scene);
}

// GameScreen: Overriden methods from sfz::BaseScreen
// ------------------------------------------------------------------------------------------------

UpdateOp GameScreen::update(UpdateState& state)
{
	return gameLogic->update(*this, state);
}

void GameScreen::render(UpdateState& state)
{
	auto& cfg = GlobalConfig::instance();

	const vec2i drawableDim = state.window.drawableDimensions();
	const vec2i targetRes = cfg.graphcisCfg().getTargetResolution(drawableDim);

	// Check if framebuffers / renderer needs to be reloaded
	if (targetRes != renderer->targetResolution()) {
		printf("New target resolution: %s, reloading framebuffers\n", toString(targetRes).str);
		this->reloadFramebuffers(targetRes);
		renderer->setTargetResolution(targetRes);
		cam.setAspectRatio(float(targetRes.x) / float(targetRes.y));
	}

	mDrawOps.clear();

	// Render all static objects in the scene, scaled down to 5% of their sizes
	for (Renderable& renderable : scene.staticRenderables) {
		mDrawOps.add(DrawOp(scalingMatrix4<float>(0.05f), &renderable));
	}
	mDrawOps.add(DrawOp(identityMatrix4<float>(), &mSnakeRenderable));
	RenderResult res = renderer->render(mResultFB, mDrawOps, scene.staticPointLights);

	glDisable(GL_BLEND);
	glEnable(GL_CULL_FACE);
	glDisable(GL_DEPTH_TEST);
	glActiveTexture(GL_TEXTURE0);

	// Apply gamma correction
	mGammaCorrectedFB.bindViewportClearColor();
	glUseProgram(mGammaCorrectionShader.handle());
	gl::setUniform(mGammaCorrectionShader, "uGamma", cfg.windowCfg().screenGamma->floatValue());
	glBindTexture(GL_TEXTURE_2D, mResultFB.texture(0));
	mFullscreenTriangle.render();

	// Scale result to screen
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glViewport(0, 0, drawableDim.x, drawableDim.y);
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClearDepth(0.0f); // Reverse depth buffer
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glUseProgram(mScalingShader.handle());
	glBindTexture(GL_TEXTURE_2D, mGammaCorrectedFB.texture(0));
	gl::setUniform(mScalingShader, "uViewportRes", vec2(res.renderedRes));
	gl::setUniform(mScalingShader, "uDstRes", vec2(drawableDim));

	mFullscreenTriangle.render();

	SDL_GL_SwapWindow(state.window.ptr());
}

// GameScreen: Private methods
// ------------------------------------------------------------------------------------------------

void GameScreen::reloadFramebuffers(vec2i internalRes) noexcept
{
	using gl::FBDepthFormat;
	using gl::FBTextureFiltering;
	using gl::FBTextureFormat;
	using gl::FramebufferBuilder;

	if (mResultFB.dimensions() != internalRes) {
		mResultFB = FramebufferBuilder(internalRes)
		            .addTexture(0, FBTextureFormat::RGBA_F16, FBTextureFiltering::LINEAR)
		            .addDepthTexture(FBDepthFormat::F32, FBTextureFiltering::NEAREST)
		            .build();
	}

	if (mGammaCorrectedFB.dimensions() != internalRes) {
		mGammaCorrectedFB = FramebufferBuilder(internalRes)
		                    .addTexture(0, FBTextureFormat::RGBA_F16, FBTextureFiltering::LINEAR)
		                    .build();
	}
}

void GameScreen::reloadShaders() noexcept
{
	mScalingShader.reload();
	if (mScalingShader.isValid()) {
		glUseProgram(mScalingShader.handle());
		gl::setUniform(mScalingShader, "uSrcTexture", 0);
	}

	mGammaCorrectionShader.reload();
	if (mGammaCorrectionShader.isValid()) {
		glUseProgram(mGammaCorrectionShader.handle());
		gl::setUniform(mGammaCorrectionShader, "uLinearTexture", 0);
	}
}

} // namespace sfz
