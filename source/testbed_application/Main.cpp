// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#include <chrono>

#include <sfz/memory/New.hpp>
#include <sfz/Screens.hpp>
#include <sfz/util/IO.hpp>

#include <phantasy_engine/PhantasyEngine.hpp>
#include <phantasy_engine/Config.hpp>
#include <phantasy_engine/Renderers.hpp>

#include <CudaRayTracerRenderer.hpp>

#include "TestbedLogic.hpp"

using namespace sfz;
using namespace sfz::gl;
using namespace sfz::sdl;

// Statics
// ------------------------------------------------------------------------------------------------

static void ensureIniDirectoryExists()
{
	StackString256 tmp;
	tmp.printf("%sPhantasyEngineTestbed", sfz::gameBaseFolderPath());
	sfz::createDirectory(tmp.str);
}

// Main
// ------------------------------------------------------------------------------------------------

int main(int, char**)
{
	// Initialize phantasy engine
	PhantasyEngine& engine = PhantasyEngine::instance();
	ensureIniDirectoryExists();
	engine.init(sfz::gameBaseFolderPath(), "PhantasyEngineTestbed/Config.ini");

	// Retrieve global config and add testbed specific settings
	sfz::GlobalConfig& cfg = sfz::GlobalConfig::instance();
	Setting* renderingBackendSetting = cfg.sanitizeInt("PhantasyEngineTestbed", "renderingBackend", 0, 0, 2);
	Setting* useSponzaSetting = cfg.sanitizeBool("PhantasyEngineTestbed", "useSponza", true);

	// Print all settings
	DynArray<Setting*> settings;
	cfg.getSettings(settings);
	printf("Available settings:\n");
	for (Setting* setting : settings) {
		if (setting->section() != "") printf("%s.", setting->section().str);
		printf("%s = ", setting->key().str);
		switch (setting->type()) {
		case SettingType::INT:
			printf("%i\n", setting->intValue());
			break;
		case SettingType::FLOAT:
			printf("%f\n", setting->floatValue());
			break;
		case SettingType::BOOL:
			printf("%s\n", setting->boolValue() ? "true" : "false");
			break;
		}
	}
	printf("\n");

	// Trap mouse
	SDL_SetRelativeMouseMode(SDL_TRUE);

	// Select rendering backend based on config
	UniquePtr<BaseRenderer> renderer;
	switch (renderingBackendSetting->intValue()) {
	default:
		printf("%s\n", "Something is wrong with the config. Falling back to deferred rendering.");
	case 0:
		renderer = UniquePtr<BaseRenderer>(sfz_new<DeferredRenderer>());
		break;
	case 1:
#ifdef CUDA_TRACER_AVAILABLE
		renderer = UniquePtr<BaseRenderer>(sfz_new<CUDARayTracerRenderer>());
#else
		printf("%s\n", "CUDA not available in this build, using deferred renderer instead.");
		renderer = UniquePtr<BaseRenderer>(sfz_new<DeferredRenderer>());
#endif
		break;
	case 2:
		renderer = UniquePtr<BaseRenderer>(sfz_new<CPURayTracerRenderer>());
		break;
	}

	// Load level
	StackString192 modelsPath;
	modelsPath.printf("%sresources/models/", basePath());
	
	UniquePtr<Level> level = makeUnique<Level>();
	if (useSponzaSetting->boolValue()) {
		
		using time_point = std::chrono::high_resolution_clock::time_point;
		time_point before = std::chrono::high_resolution_clock::now();

		level->scene.staticRenderables.add(assimpLoadSponza(modelsPath.str, "sponzaPBR/sponzaPBR.obj"));
	
		time_point after = std::chrono::high_resolution_clock::now();
		using FloatSecond = std::chrono::duration<float>;
		float delta = std::chrono::duration_cast<FloatSecond>(after - before).count();
		printf("Time spent loading sponza: %.3f seconds\n", delta);

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
			level->scene.staticPointLights.add(pointLight);
		}
	}
	else {
		Renderable testRenderable;
		RenderableComponent testComponent;
		Vertex v1;
		v1.pos ={-0.5f, 2.0f, -2.0f};
		v1.normal ={0.0f, 0.0f, 0.0f};
		v1.uv ={0.0f, 1.0f};

		Vertex v2;
		v2.pos ={0.0f, 2.0f, -2.0f};
		v2.uv ={1.0f, 1.0f};

		Vertex v3;
		v3.pos ={0.0f, 2.5f, -2.0f};
		v3.uv ={1.0f, 0.0f};

		Vertex testVertices[3] ={v1, v2, v3};
		uint32_t indices[3] ={0, 1, 2};
		testComponent.geometry.vertices.add(testVertices, 3);
		testComponent.geometry.indices.add(indices, 3);
		testComponent.glModel.load(testComponent.geometry);

		testRenderable.components.add(std::move(testComponent));
		level->scene.staticRenderables.add(std::move(testRenderable));
	}

	// Run gameloop
	sfz::runGameLoop(engine.window(), SharedPtr<BaseScreen>(sfz_new<GameScreen>(
		UniquePtr<GameLogic>(sfz_new<TestbedLogic>()),
		std::move(level),
		std::move(renderer)
	)));

	// Deinitializes Phantasy Engine
	engine.destroy();
	return 0;
}
