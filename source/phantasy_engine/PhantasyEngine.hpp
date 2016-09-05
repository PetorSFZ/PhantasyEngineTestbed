// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include <sfz/sdl/Window.hpp>

namespace sfz {

// Phantasy Engine singleton class
// ------------------------------------------------------------------------------------------------

class PhantasyEngineImpl; // Pimpl pattern

class PhantasyEngine final {
public:
	// Singleton instance
	// --------------------------------------------------------------------------------------------

	static PhantasyEngine& instance() noexcept;

	// Methods
	// --------------------------------------------------------------------------------------------

	/// Initializes phantasy engine. Should more or less be the first thing you call in your main
	/// function. It sets up the config system, creates a window, initializes OpenGL, etc.
	void init(const char* projectName, const char* iniBasePath, const char* iniFileName) noexcept;

	/// Deinitializes phantasy engine. Should be called on shutdown when the engine will no longer
	/// be used, for example at the end of your main function.
	void destroy() noexcept;

	// Getters
	// --------------------------------------------------------------------------------------------

	sdl::Window& window() noexcept;

private:
	// Private constructors & destructors
	// --------------------------------------------------------------------------------------------

	PhantasyEngine(const PhantasyEngine&) = delete;
	PhantasyEngine& operator= (const PhantasyEngine&) = delete;
	PhantasyEngine(PhantasyEngine&&) = delete;
	PhantasyEngine& operator= (PhantasyEngine&&) = delete;

	PhantasyEngine() noexcept = default;
	~PhantasyEngine() noexcept;

	// Private members
	// --------------------------------------------------------------------------------------------

	PhantasyEngineImpl* mImpl = nullptr;
};

} // namespace sfz
