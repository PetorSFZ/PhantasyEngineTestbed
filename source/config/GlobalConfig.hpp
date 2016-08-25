// Copyright (c) Peter Hillerström (skipifzero.com, peter@hstroem.se)

#pragma once

#include <sfz/containers/DynArray.hpp>
#include <sfz/memory/SmartPointers.hpp>

#include "config/Setting.hpp"

namespace sfz {

// GlobalConfig
// ------------------------------------------------------------------------------------------------

class GlobalConfig final {
public:
	// Singleton instance
	// --------------------------------------------------------------------------------------------

	static GlobalConfig& instance() noexcept;

	// Methods
	// --------------------------------------------------------------------------------------------

	bool load(const char* path) noexcept;
	bool save(const char* path) noexcept;

	SharedPtr<Setting> setting(const char* identifier) const noexcept;

private:
	// Private constructors & destructors
	// --------------------------------------------------------------------------------------------

	GlobalConfig() noexcept = default;
	GlobalConfig(const GlobalConfig&) = delete;
	GlobalConfig& operator= (const GlobalConfig&) = delete;
	GlobalConfig(GlobalConfig&&) = delete;
	GlobalConfig& operator= (GlobalConfig&&) = delete;

	// Private members
	// --------------------------------------------------------------------------------------------

	DynArray<SharedPtr<Setting>> mSettings;
};

} // namespace sfz
