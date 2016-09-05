// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#include "PhantasyEngine.hpp"

#ifdef _WIN32
#include <direct.h>
#endif

#include <sfz/gl/IncludeOpenGL.hpp>
#include <sfz/GL.hpp>
#include <sfz/memory/New.hpp>
#include <sfz/SDL.hpp>
#include <sfz/util/IO.hpp>

#include "phantasy_engine/Config.hpp"

namespace sfz {

using namespace sdl;
using namespace gl;

// Statics
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

// PhantasyEngineImpl
// ------------------------------------------------------------------------------------------------

class PhantasyEngineImpl final {
public:
	// Members
	// --------------------------------------------------------------------------------------------

	sdl::Session sdlSession;
	sdl::Window window;
	gl::Context glContext;

	// Constructors & destructors
	// --------------------------------------------------------------------------------------------

	PhantasyEngineImpl() noexcept = default;
	PhantasyEngineImpl(const PhantasyEngineImpl&) = delete;
	PhantasyEngineImpl& operator= (const PhantasyEngineImpl&) = delete;
	PhantasyEngineImpl(PhantasyEngineImpl&&) = delete;
	PhantasyEngineImpl& operator= (PhantasyEngineImpl&&) = delete;
	~PhantasyEngineImpl() noexcept = default;
};

// PhantasyEngine: Singleton instance
// ------------------------------------------------------------------------------------------------

PhantasyEngine& PhantasyEngine::instance() noexcept
{
	static PhantasyEngine engine;
	return engine;
}

// Phantasy Engine: Methods
// ------------------------------------------------------------------------------------------------

void PhantasyEngine::init(const char* projectName, const char* iniBasePath, const char* iniFileName) noexcept
{
	if (mImpl != nullptr) {
		sfz::printErrorMessage("Attempting to initialize PhantasyEngine while already initialized.");
		this->destroy();
	}
	mImpl = sfz_new<PhantasyEngineImpl>();

	// Windwows specific hacks
#ifdef _WIN32
	// Enable hi-dpi awareness
	SetProcessDPIAware();

	// Set current working directory to SDL_BasePath()
	_chdir(sfz::basePath());
#endif

	// Load global settings
	GlobalConfig& cfg = GlobalConfig::instance();
	cfg.init(iniBasePath, iniFileName);
	cfg.load();
	const WindowConfig& wCfg = cfg.windowCfg();

	// Start SDL session and create window
	mImpl->sdlSession = Session({SDLInitFlags::EVENTS, SDLInitFlags::VIDEO, SDLInitFlags::AUDIO,
	                             SDLInitFlags::GAMECONTROLLER}, {});
	mImpl->window = Window(projectName, wCfg.width->intValue(), wCfg.height->intValue(),
	                       {WindowFlags::OPENGL, WindowFlags::RESIZABLE, WindowFlags::ALLOW_HIGHDPI});

	// OpenGL context
	SDL_GL_SetAttribute(SDL_GL_FRAMEBUFFER_SRGB_CAPABLE, 0); // Request a non-sRGB framebuffer
	mImpl->glContext = createGLContext(mImpl->window, 4, 5);

	// Initializes GLEW, must happen after GL context is created.
	glewExperimental = GL_TRUE;
	GLenum glewError = glewInit();
	if (glewError != GLEW_OK) {
		sfz::error("GLEW init failure: %s", glewGetErrorString(glewError));
	}

	gl::printSystemGLInfo();

	// Change OpenGL clip space to match Direct3D/Vulkan
	if (!GLEW_ARB_clip_control) {
		sfz::error("OpenGL error: ARB_clip_control not available");
	}
	glClipControl(GL_UPPER_LEFT, GL_ZERO_TO_ONE);

	// Make sure selected display index is valid
	const int numDisplays = SDL_GetNumVideoDisplays();
	if (numDisplays < 0) sfz::printErrorMessage("SDL_GetNumVideoDisplays() failed: %s", SDL_GetError());
	if (wCfg.displayIndex->intValue() >= numDisplays) {
		sfz::printErrorMessage("Display index %i is invalid, number of displays is %i. Resetting to 0.",
		                       wCfg.displayIndex->intValue(), numDisplays);
		wCfg.displayIndex->setInt(0);
	}

	// Fullscreen & VSync
	// TODO: display index broken?
	mImpl->window.setVSync(static_cast<VSync>(wCfg.vsync->intValue()));
	mImpl->window.setFullscreen(static_cast<Fullscreen>(wCfg.fullscreenMode->intValue()), wCfg.displayIndex->intValue());
	if (wCfg.maximized->boolValue()) {
		SDL_MaximizeWindow(mImpl->window.ptr());
	}

	// Attempt to disable sRGB framebuffer for the default framebuffer
	if (GLEW_ARB_framebuffer_sRGB) {
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
		glDisable(GL_FRAMEBUFFER_SRGB_EXT);
	}

	// Enable OpenGL debug message if in debug mode
#if !defined(SFZ_NO_DEBUG)
	gl::setupDebugMessages(gl::Severity::MEDIUM, gl::Severity::MEDIUM);
#endif
}

void PhantasyEngine::destroy() noexcept
{
	if (mImpl == nullptr) return;

	// Store global settings
	GlobalConfig& cfg = GlobalConfig::instance();
	cfg.save();
	cfg.destroy();

	sfz_delete(mImpl);
	mImpl = nullptr;
}

// Phantasy Engine: Getters
// ------------------------------------------------------------------------------------------------

sdl::Window& PhantasyEngine::window() noexcept
{
	sfz_assert_debug(mImpl != nullptr);
	return mImpl->window;
}

// Phantasy Engine: Private constructors & destructors
// ------------------------------------------------------------------------------------------------

PhantasyEngine::~PhantasyEngine() noexcept
{
	this->destroy();
}

} // namespace sfz
