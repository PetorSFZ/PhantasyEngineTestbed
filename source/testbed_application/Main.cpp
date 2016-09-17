// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#include <chrono>

#include <sfz/memory/New.hpp>
#include <sfz/Screens.hpp>
#include <sfz/util/IO.hpp>

#include <phantasy_engine/PhantasyEngine.hpp>
#include <phantasy_engine/Config.hpp>
#include <phantasy_engine/level/SponzaLoader.hpp>

#include "Helpers.hpp"
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
	sfz::createDirectory(sfz::gameBaseFolderPath());
	sfz::createDirectory(tmp.str);
}

// Main
// ------------------------------------------------------------------------------------------------

int main(int, char**)
{
	using namespace phe;
	using namespace sfz;

	// Initialize phantasy engine
	PhantasyEngine& engine = PhantasyEngine::instance();
	ensureIniDirectoryExists();
	engine.init("Phantasy Engine - Testbed", sfz::gameBaseFolderPath(), "PhantasyEngineTestbed/Config.ini");

	// Retrieve global config and add testbed specific settings
	GlobalConfig& cfg = GlobalConfig::instance();
	Setting* renderingBackendSetting = cfg.sanitizeInt("PhantasyEngineTestbed", "renderingBackend", 0, 0, 2);

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
	uint32_t rendererIndex = ~0u;
	auto renderers = createRenderers(rendererIndex);
	SharedPtr<BaseRenderer> initialRenderer = renderers[rendererIndex].renderer;

	// Load level
	StackString192 modelsPath;
	modelsPath.printf("%sresources/models/", basePath());
	
	SharedPtr<Level> level = makeShared<Level>();

	using time_point = std::chrono::high_resolution_clock::time_point;
	time_point before = std::chrono::high_resolution_clock::now();

	loadStaticSceneSponza(modelsPath.str, "sponzaPBR/sponzaPBR.obj", *level, scalingMatrix4(0.05f));

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
		level->staticScene.pointLights.add(pointLight);
	}

	// Run gameloop
	sfz::runGameLoop(engine.window(), SharedPtr<BaseScreen>(sfz_new<GameScreen>(
		SharedPtr<GameLogic>(sfz_new<TestbedLogic>(std::move(renderers), rendererIndex)),
		level,
		initialRenderer
	)));

	// Deinitializes Phantasy Engine
	engine.destroy();
	return 0;
}
