// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#include "PhantasyEngine.hpp"

#ifdef _WIN32
#include <direct.h>
#endif

#include <imgui.h>
#include <imgui_impl_sdl_gl3.h>

#include <sfz/gl/IncludeOpenGL.hpp>
#include <sfz/GL.hpp>
#include <sfz/memory/New.hpp>
#include <sfz/SDL.hpp>
#include <sfz/util/IO.hpp>

#include "phantasy_engine/Config.hpp"

namespace phe {

using namespace sfz;
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
	glFrontFace(GL_CW); // Changes to D3D standard where we use clock wise winding order for front face

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

	if (!GLEW_ARB_bindless_texture) {
		sfz::error("OpenGL error: ARB_bindless_texture not available");
	}

	if (!GLEW_ARB_texture_storage) {
		sfz::error("OpenGL error: ARB_texture_storage not available");
	}

	// Enable OpenGL debug message if in debug mode
#if !defined(SFZ_NO_DEBUG)
	gl::setupDebugMessages(gl::Severity::MEDIUM, gl::Severity::MEDIUM);
#endif

	initImGui();
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

// Private methods
// ------------------------------------------------------------------------------------------------

void PhantasyEngine::initImGui() const noexcept
{
	using namespace ImGui;

	ImGuiIO& io = GetIO();

	// Disable logging and saving settings
	io.IniFilename = nullptr;
	io.LogFilename = nullptr;

	ImGui_ImplSdlGL3_Init(mImpl->window.ptr());

	ImGuiStyle& style = ImGui::GetStyle();
	style.Colors[ImGuiCol_Text] = ImVec4(0.00f, 0.00f, 0.00f, 0.92f);
	style.Colors[ImGuiCol_TextDisabled] = ImVec4(0.60f, 0.60f, 0.60f, 1.00f);
	style.Colors[ImGuiCol_WindowBg] = ImVec4(0.94f, 0.94f, 0.94f, 1.00f);
	style.Colors[ImGuiCol_ChildWindowBg] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
	style.Colors[ImGuiCol_PopupBg] = ImVec4(0.94f, 0.94f, 0.94f, 1.00f);
	style.Colors[ImGuiCol_Border] = ImVec4(0.00f, 0.00f, 0.00f, 0.39f);
	style.Colors[ImGuiCol_BorderShadow] = ImVec4(1.00f, 1.00f, 1.00f, 0.10f);
	style.Colors[ImGuiCol_FrameBg] = ImVec4(1.00f, 1.00f, 0.99f, 1.00f);
	style.Colors[ImGuiCol_FrameBgHovered] = ImVec4(0.26f, 0.59f, 0.98f, 0.40f);
	style.Colors[ImGuiCol_FrameBgActive] = ImVec4(0.26f, 0.59f, 0.98f, 0.67f);
	style.Colors[ImGuiCol_TitleBg] = ImVec4(0.86f, 0.86f, 0.86f, 0.99f);
	style.Colors[ImGuiCol_TitleBgCollapsed] = ImVec4(1.00f, 1.00f, 1.00f, 0.51f);
	style.Colors[ImGuiCol_TitleBgActive] = ImVec4(0.82f, 0.82f, 0.82f, 1.00f);
	style.Colors[ImGuiCol_MenuBarBg] = ImVec4(0.86f, 0.86f, 0.86f, 1.00f);
	style.Colors[ImGuiCol_ScrollbarBg] = ImVec4(0.98f, 0.98f, 0.98f, 0.53f);
	style.Colors[ImGuiCol_ScrollbarGrab] = ImVec4(0.69f, 0.69f, 0.69f, 0.80f);
	style.Colors[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.49f, 0.49f, 0.49f, 0.80f);
	style.Colors[ImGuiCol_ScrollbarGrabActive] = ImVec4(0.49f, 0.49f, 0.49f, 1.00f);
	style.Colors[ImGuiCol_ComboBg] = ImVec4(0.86f, 0.86f, 0.86f, 0.99f);
	style.Colors[ImGuiCol_CheckMark] = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
	style.Colors[ImGuiCol_SliderGrab] = ImVec4(0.26f, 0.59f, 0.98f, 0.78f);
	style.Colors[ImGuiCol_SliderGrabActive] = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
	style.Colors[ImGuiCol_Button] = ImVec4(0.26f, 0.59f, 0.98f, 0.40f);
	style.Colors[ImGuiCol_ButtonHovered] = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
	style.Colors[ImGuiCol_ButtonActive] = ImVec4(0.06f, 0.53f, 0.98f, 1.00f);
	style.Colors[ImGuiCol_Header] = ImVec4(0.26f, 0.59f, 0.98f, 0.31f);
	style.Colors[ImGuiCol_HeaderHovered] = ImVec4(0.26f, 0.59f, 0.98f, 0.80f);
	style.Colors[ImGuiCol_HeaderActive] = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
	style.Colors[ImGuiCol_Column] = ImVec4(0.39f, 0.39f, 0.39f, 1.00f);
	style.Colors[ImGuiCol_ColumnHovered] = ImVec4(0.26f, 0.59f, 0.98f, 0.78f);
	style.Colors[ImGuiCol_ColumnActive] = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
	style.Colors[ImGuiCol_ResizeGrip] = ImVec4(1.00f, 1.00f, 1.00f, 0.00f);
	style.Colors[ImGuiCol_ResizeGripHovered] = ImVec4(0.26f, 0.59f, 0.98f, 0.67f);
	style.Colors[ImGuiCol_ResizeGripActive] = ImVec4(0.26f, 0.59f, 0.98f, 0.95f);
	style.Colors[ImGuiCol_CloseButton] = ImVec4(0.59f, 0.59f, 0.59f, 0.50f);
	style.Colors[ImGuiCol_CloseButtonHovered] = ImVec4(0.98f, 0.39f, 0.36f, 1.00f);
	style.Colors[ImGuiCol_CloseButtonActive] = ImVec4(0.98f, 0.39f, 0.36f, 1.00f);
	style.Colors[ImGuiCol_PlotLines] = ImVec4(0.39f, 0.39f, 0.39f, 1.00f);
	style.Colors[ImGuiCol_PlotLinesHovered] = ImVec4(1.00f, 0.43f, 0.35f, 1.00f);
	style.Colors[ImGuiCol_PlotHistogram] = ImVec4(0.90f, 0.70f, 0.00f, 1.00f);
	style.Colors[ImGuiCol_PlotHistogramHovered] = ImVec4(1.00f, 0.60f, 0.00f, 1.00f);
	style.Colors[ImGuiCol_TextSelectedBg] = ImVec4(0.26f, 0.59f, 0.98f, 0.35f);
	style.Colors[ImGuiCol_ModalWindowDarkening] = ImVec4(0.20f, 0.20f, 0.20f, 0.35f);

	style.Alpha = 1.0f;
	style.AntiAliasedShapes = true;
	style.AntiAliasedLines = true;
	style.FrameRounding = 4;
}

} // namespace phe
