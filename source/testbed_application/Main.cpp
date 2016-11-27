// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#include <chrono>

#include <sfz/memory/New.hpp>
#include <sfz/Screens.hpp>
#include <sfz/util/IO.hpp>

#include <phantasy_engine/PhantasyEngine.hpp>
#include <phantasy_engine/Config.hpp>
#include <phantasy_engine/level/SponzaLoader.hpp>
#include <phantasy_engine/level/ObjectLoader.hpp>
#include <phantasy_engine/ray_tracer_common/StaticBVHBuilder.hpp>

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
	Setting* renderingBackendSetting = cfg.sanitizeInt("PhantasyEngineTestbed", "renderingBackend", 0, 0, 1);

	cfg.sanitizeFloat("PhantasyEngineTestbed", "sphereColourR", 1.0f, 0.0f, 1.0f);
	cfg.sanitizeFloat("PhantasyEngineTestbed", "sphereColourG", 1.0f, 0.0f, 1.0f);
	cfg.sanitizeFloat("PhantasyEngineTestbed", "sphereColourB", 1.0f, 0.0f, 1.0f);
	cfg.sanitizeFloat("PhantasyEngineTestbed", "sphereRoughness", 0.5f, 0.0f, 1.0f);
	cfg.sanitizeFloat("PhantasyEngineTestbed", "sphereMetallic", 0.0f, 0.0f, 1.0f);

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

	StackString192 modelsPath;

	// Load level
	modelsPath.printf("%sresources/models/", basePath());
	
	// Create level struct and initialize ECS system
	SharedPtr<Level> level = makeShared<Level>();
	level->ecs = EntityComponentSystem(8192);
	level->ecsRenderComponents = EcsComponentAccessor<RenderComponent>(level->ecs);

	using time_point = std::chrono::high_resolution_clock::time_point;
	using FloatSecond = std::chrono::duration<float>;
	time_point before, after;
	float delta;

	before = std::chrono::high_resolution_clock::now();

	loadStaticSceneSponza(modelsPath.str, "sponzaPBR/sponzaPBR.obj", *level, scalingMatrix4(0.05f));
	
	after = std::chrono::high_resolution_clock::now();
	delta = std::chrono::duration_cast<FloatSecond>(after - before).count();
	printf("Time spent loading sponza: %.3f seconds\n", delta);

	{
		StackString256 bvhNodesCachePath;
		bvhNodesCachePath.printf("%ssponza_bvhnode.bvhcache", basePath());
		StackString256 bvhTriVertsCachePath;
		bvhTriVertsCachePath.printf("%ssponza_triverts.bvhcache", basePath());
		StackString256 bvhTriDatasCachePath;
		bvhTriDatasCachePath.printf("%ssponza_tridatas.bvhcache", basePath());

		BVH& bvh = level->staticScene.bvh;

		// Check if BVH is cached
		if (fileExists(bvhNodesCachePath.str) &&
			fileExists(bvhTriVertsCachePath.str) &&
			fileExists(bvhTriDatasCachePath.str)) {

			printf("Started reading static BVH from file cache\n");
			DynArray<uint8_t> nodes = readBinaryFile(bvhNodesCachePath.str);
			DynArray<uint8_t> triVerts = readBinaryFile(bvhTriVertsCachePath.str);
			DynArray<uint8_t> triDatas = readBinaryFile(bvhTriDatasCachePath.str);

			bvh.nodes.add((BVHNode*)nodes.data(), nodes.size() / sizeof(BVHNode));
			bvh.triangleVerts.add((TriangleVertices*)triVerts.data(), triVerts.size() / sizeof(TriangleVertices));
			bvh.triangleDatas.add((TriangleData*)triDatas.data(), triDatas.size() / sizeof(TriangleData));

			printf("Finished reading static BVH from file cache\n");

			if (bvh.triangleVerts.size() != bvh.triangleDatas.size()) {
				sfz::error("Invalid BVH cache");
			}
			// TODO: Insert more error checks
		}

		// Create BVH and write it to file
		else {
			printf("Started building static BVH\n");
			before = std::chrono::high_resolution_clock::now();

			phe::buildStaticBVH(level->staticScene);
			phe::sanitizeBVH(bvh);

			after = std::chrono::high_resolution_clock::now();
			delta = std::chrono::duration_cast<FloatSecond>(after - before).count();
			printf("Finished building static BVH: %.3f seconds\n", delta);

			writeBinaryFile(bvhNodesCachePath.str, (const uint8_t*)bvh.nodes.data(), bvh.nodes.size() * sizeof(BVHNode));
			writeBinaryFile(bvhTriVertsCachePath.str, (const uint8_t*)bvh.triangleVerts.data(), bvh.triangleVerts.size() * sizeof(TriangleVertices));
			writeBinaryFile(bvhTriDatasCachePath.str, (const uint8_t*)bvh.triangleDatas.data(), bvh.triangleDatas.size() * sizeof(TriangleData));
			printf("Wrote static BVH to file for future runs\n");
		}
	}

	// Add lights to the scene
	vec3 lightColors[]{
		vec3{ 1.0f, 0.0f, 1.0f },
		vec3{ 1.0f, 1.0f, 1.0f }
		//vec3{ 0.0f, 1.0f, 1.0f },
		//vec3{ 1.0f, 1.0f, 0.0f },
		//vec3{ 1.0f, 1.0f, 1.0f }
	};
	size_t numLights = sizeof(lightColors) / sizeof(vec3);

	for (int i = 0; i < numLights; i++) {
		SphereLight sphereLight;
		sphereLight.pos = vec3(-50.0f + 100.0f * i / (numLights - 1), 5.0f, 0.0f);
		sphereLight.range = 70.0f;
		sphereLight.strength = 300.0f * lightColors[i];
		sphereLight.radius = 0.5f;
		sphereLight.staticShadows = true;
		sphereLight.dynamicShadows = true;
		level->staticScene.sphereLights.add(std::move(sphereLight));
	}

	SphereLight sunlight;
	sunlight.pos = vec3(28.0f, 81.0f, 0.0f);
	sunlight.range = 100.0f;
	sunlight.strength = 5000.0f * vec3(1.0f);
	sunlight.radius = 2.0f;
	sunlight.staticShadows = true;
	sunlight.dynamicShadows = true;

	level->staticScene.sphereLights.add(std::move(sunlight));

	// Add triangle mesh to scene
	loadDynObject(modelsPath.str, "skysphere/skysphere.obj", *level, scalingMatrix4(10.0f));
	loadDynObjectCustomMaterial(modelsPath.str, "sphere.obj", *level, vec3(1.0f, 0.0f, 0.0f), 0.1f, 1.0f);
	loadDynObjectCustomMaterial(modelsPath.str, "sphere.obj", *level, vec3(0.0f, 1.0f, 0.0f), 0.1f, 1.0f);

	// Run gameloop
	sfz::runGameLoop(engine.window(), SharedPtr<BaseScreen>(sfz_new<GameScreen>(
		SharedPtr<GameLogic>(sfz_new<TestbedLogic>(std::move(renderers), rendererIndex, *level)),
		level,
		initialRenderer
	)));

	// Deinitializes Phantasy Engine
	engine.destroy();
	return 0;
}
