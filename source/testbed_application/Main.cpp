// Copyright (c) Peter Hillerström (skipifzero.com, peter@hstroem.se)

#include <sfz/memory/New.hpp>
#include <sfz/Screens.hpp>
#include <sfz/util/IO.hpp>

#include <PhantasyEngine.hpp>
#include <Config.hpp>
#include <Renderers.hpp>

#include <CudaRayTracerRenderer.hpp>

#include "TestbedLogic.hpp"

using namespace sfz;
using namespace sfz::gl;
using namespace sfz::sdl;

// Main
// ------------------------------------------------------------------------------------------------

int main(int, char**)
{
	// Initialize phantasy engine
	PhantasyEngine& engine = PhantasyEngine::instance();
	engine.init(sfz::basePath(), "Config.ini");

	// Retrieve and print all settings
	sfz::GlobalConfig& cfg = sfz::GlobalConfig::instance();
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
	switch (cfg.graphcisCfg().renderingBackend->intValue()) {
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

	// Run gameloop
	sfz::runGameLoop(engine.window(), SharedPtr<BaseScreen>(sfz_new<GameScreen>(
		UniquePtr<GameLogic>(sfz_new<TestbedLogic>()),
		std::move(renderer)
	)));

	// Deinitializes Phantasy Engine
	engine.destroy();
	return 0;
}
