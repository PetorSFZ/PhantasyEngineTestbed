// Copyright (c) Peter Hillerström (skipifzero.com, peter@hstroem.se)

#include <sfz/GL.hpp>
#include <sfz/gl/IncludeOpenGL.hpp>
#include <sfz/Screens.hpp>
#include <sfz/SDL.hpp>

#include "Config.hpp"
#include "Screens.hpp"

#ifdef _WIN32
#include <sfz/util/IO.hpp>
#include <direct.h>
#endif

using namespace sfz;
using namespace sfz::gl;
using namespace sfz::sdl;

// Helper functions
// ------------------------------------------------------------------------------------------------

static Context createGLContext(const Window& window, int major, int minor) noexcept
{
#if !defined(SFZ_NO_DEBUG)
#ifdef _WIN32
	return Context(window.ptr(), major, minor, GLContextProfile::COMPATIBILITY, true);
#else
	return Context(window.ptr(), major, minor, GLContextProfile::CORE, true);
#endif
#else
#ifdef _WIN32
	return Context(window.ptr(), major, minor, GLContextProfile::COMPATIBILITY, false);
#else
	return Context(window.ptr(), major, minor, GLContextProfile::CORE, false);
#endif
#endif
}

// Main
// ------------------------------------------------------------------------------------------------

int main(int, char**)
{
	// Windwows specific hacks
#ifdef _WIN32
	// Enable hi-dpi awareness
	SetProcessDPIAware();

	// Set current working directory to SDL_BasePath()
	_chdir(sfz::basePath());
#endif

	// Load global settings
	GlobalConfig& cfg = GlobalConfig::instance();
	cfg.init(sfz::basePath(), "Config.ini");
	cfg.load();
	const WindowConfig& wCfg = cfg.windowCfg();

	// Print all available settings and their values
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

	// Start SDL session and create window
	Session sdlSession({SDLInitFlags::EVENTS, SDLInitFlags::VIDEO, SDLInitFlags::AUDIO,
	                    SDLInitFlags::GAMECONTROLLER}, {});
	Window window("Phantasy Engine - Testbed", wCfg.width->intValue(), wCfg.height->intValue(),
	              {WindowFlags::OPENGL, WindowFlags::RESIZABLE, WindowFlags::ALLOW_HIGHDPI});

	// OpenGL context
	Context glContext = createGLContext(window, 4, 5);

	// Initializes GLEW, must happen after GL context is created.
	glewExperimental = GL_TRUE;
	GLenum glewError = glewInit();
	if (glewError != GLEW_OK) {
		sfz::error("GLEW init failure: %s", glewGetErrorString(glewError));
	}

	gl::printSystemGLInfo();

	// Make sure selected display index is valid
	const int numDisplays = SDL_GetNumVideoDisplays();
	if (numDisplays < 0) sfz::printErrorMessage("SDL_GetNumVideoDisplays() failed: %s", SDL_GetError());
	if (wCfg.displayIndex->intValue() >= numDisplays) {
		sfz::printErrorMessage("Display index %i is invalid, number of displays is %i. Resetting to 0.",
		                       wCfg.displayIndex->intValue(), numDisplays);
		wCfg.displayIndex->setInt(0);
	}

	// Fullscreen & VSync
	window.setVSync(static_cast<VSync>(wCfg.vsync->intValue()));
	window.setFullscreen(static_cast<Fullscreen>(wCfg.fullscreenMode->intValue()), wCfg.displayIndex->intValue());

	// Enable OpenGL debug message if in debug mode
#if !defined(SFZ_NO_DEBUG)
	gl::setupDebugMessages(gl::Severity::MEDIUM, gl::Severity::MEDIUM);
#endif

	// Trap mouse
	SDL_SetRelativeMouseMode(SDL_TRUE);

	// Run gameloop
	sfz::runGameLoop(window, SharedPtr<BaseScreen>(sfz_new<GameScreen>()));

	// Store global settings
	cfg.save();
	cfg.destroy();

	return 0;
}