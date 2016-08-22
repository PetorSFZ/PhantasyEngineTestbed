// Copyright (c) Peter Hillerström (skipifzero.com, peter@hstroem.se)

#include <sfz/GL.hpp>
#include <sfz/gl/IncludeOpenGL.hpp>
#include <sfz/Screens.hpp>
#include <sfz/SDL.hpp>

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

	// Start SDL session and create window
	Session sdlSession({SDLInitFlags::EVENTS, SDLInitFlags::VIDEO, SDLInitFlags::AUDIO,
	                    SDLInitFlags::GAMECONTROLLER}, {});
	Window window("Phantasy Engine - Testbed", 1600, 900, {WindowFlags::OPENGL,
	              WindowFlags::RESIZABLE, WindowFlags::ALLOW_HIGHDPI});

	// OpenGL context
	Context glContext = createGLContext(window, 4, 5);

	// Initializes GLEW, must happen after GL context is created.
	glewExperimental = GL_TRUE;
	GLenum glewError = glewInit();
	if (glewError != GLEW_OK) {
		sfz::error("GLEW init failure: %s", glewGetErrorString(glewError));
	}

	gl::printSystemGLInfo();

	// Fullscreen & VSync
	window.setVSync(VSync::OFF);
	//window.setFullscreen(Fullscreen::WINDOWED, 0);

	// Enable OpenGL debug message if in debug mode
#if !defined(SFZ_NO_DEBUG)
	gl::setupDebugMessages(gl::Severity::MEDIUM, gl::Severity::MEDIUM);
#endif

	// Trap mouse
	SDL_SetRelativeMouseMode(SDL_TRUE);

	// Run gameloop
	sfz::runGameLoop(window, SharedPtr<BaseScreen>(sfz_new<GameScreen>()));

	return 0;
}