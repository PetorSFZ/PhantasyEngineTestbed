// Copyright (c) Peter Hillerström (skipifzero.com, peter@hstroem.se)

#pragma once

#include <sfz/containers/DynArray.hpp>
#include <sfz/math/Vector.hpp>

#include "config/Setting.hpp"

namespace sfz {

// Config structs
// ------------------------------------------------------------------------------------------------

struct WindowConfig final {
	Setting* displayIndex = nullptr;
	Setting* fullscreenMode = nullptr; // 0 = off, 1 = windowed, 2 = exclusive
	Setting* vsync = nullptr; // 0 = off, 1 = on, 2 = swap control tear
	Setting* width = nullptr;
	Setting* height = nullptr;
};

struct GraphicsConfig final {
	Setting* renderingBackend = nullptr; // 0 = Deferred, 1 = CUDARayTracing, 2 = CPURayTracing
	Setting* useNativeTargetResolution = nullptr;
	Setting* targetResolutionHeight = nullptr;

	/// Helper function that retrieves the target resolution
	/// Makes use of 'useNativeTargetResolution' and 'targetResolutionHeight' settings in
	/// combination with the current native resolution
	/// \param drawableDim the current native resolution, sdl::window.drawableDimensions()
	vec2i getTargetResolution(vec2i drawableDim) const noexcept;
};

// GlobalConfig
// ------------------------------------------------------------------------------------------------

class GlobalConfigImpl; // Pimpl pattern

/// The global config singleton
///
/// Setting invariants:
/// 1, All settings are owned by the singleton instance, no one else may delete the memory.
/// 2, A setting, once created, can never be destroyed or removed during runtime.
/// 3, A setting will occupy the same place in memory for the duration of the program's runtime.
/// 4, A setting can not change section or key identifiers once created.
///
/// These invariants mean that it is safe (and expected) to store direct pointers to settings and
/// read/write to them when needed. However, settings may change type during runtime. So it is
/// recommended to store a pointer to the setting itself and not its internal int value for
/// example.
///
/// Settings are expected to stay relatively static during the runtime of a program. They are not
/// meant for communication and should not be changed unless the user specifically requests for
/// them to be changed.
class GlobalConfig final {
public:
	// Singleton instance
	// --------------------------------------------------------------------------------------------

	static GlobalConfig& instance() noexcept;

	// Methods
	// --------------------------------------------------------------------------------------------

	void init(const char* basePath, const char* fileName) noexcept;
	void destroy() noexcept;

	bool load() noexcept;
	bool save() noexcept;

	/// Gets the specified Setting. If it does not exist it will be created (type int with value 0).
	/// The optional parameter "created" returns whether the Setting was created or already existed.
	Setting* getCreateSetting(const char* section, const char* key, bool* created = nullptr) noexcept;

	// Getters
	// --------------------------------------------------------------------------------------------

	/// Gets the specified Setting. Returns nullptr if it does not exist.
	Setting* getSetting(const char* section, const char* key) noexcept;
	Setting* getSetting(const char* key) noexcept;

	/// Returns pointers to all available settings
	void getSettings(DynArray<Setting*>& settings) noexcept;

	const WindowConfig& windowCfg() const noexcept;
	const GraphicsConfig& graphcisCfg() const noexcept;

private:
	// Private constructors & destructors
	// --------------------------------------------------------------------------------------------

	GlobalConfig() noexcept = default;
	GlobalConfig(const GlobalConfig&) = delete;
	GlobalConfig& operator= (const GlobalConfig&) = delete;
	GlobalConfig(GlobalConfig&&) = delete;
	GlobalConfig& operator= (GlobalConfig&&) = delete;

	~GlobalConfig() noexcept;

	// Private members
	// --------------------------------------------------------------------------------------------

	GlobalConfigImpl* mImpl = nullptr;
};

} // namespace sfz
