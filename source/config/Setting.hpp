// Copyright (c) Peter Hillerström (skipifzero.com, peter@hstroem.se)

#pragma once

#include <cstdint>

#include <sfz/containers/StackString.hpp>

namespace sfz {

using std::int32_t;

// Setting
// ------------------------------------------------------------------------------------------------

enum class SettingType : int32_t {
	INT,
	FLOAT,
	BOOL
};

class Setting final {
public:
	// Constructors & destructors
	// --------------------------------------------------------------------------------------------

	Setting(const Setting&) noexcept = default;
	Setting& operator= (const Setting&) noexcept = default;
	~Setting() noexcept = default;

	Setting() noexcept;
	Setting(const char* identifier) noexcept;
	Setting(const char* identifier, int32_t value) noexcept;
	Setting(const char* identifier, float value) noexcept;
	Setting(const char* identifier, bool value) noexcept;

	// Methods
	// --------------------------------------------------------------------------------------------

	inline const char* identifier() const noexcept { return mIdent.str; }
	inline SettingType type() const noexcept { return mType; }

	int32_t intValue() const noexcept;
	float floatValue() const noexcept;
	bool boolValue() const noexcept;

	void setInt(int32_t value) noexcept;
	void setFloat(float value) noexcept;
	void setBool(bool value) noexcept;

private:
	// Private members
	// --------------------------------------------------------------------------------------------

	StackString256 mIdent;
	SettingType mType;
	union {
		int32_t i;
		float f;
		bool b;
	};
};

} // namespace sfz
