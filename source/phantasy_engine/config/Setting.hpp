// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

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

	Setting() = delete;
	Setting(const Setting&) = delete;
	Setting& operator= (const Setting&) = delete;
	Setting(Setting&&) = delete;
	Setting& operator= (Setting&&) = delete;
	~Setting() noexcept = default;

	Setting(const char* section, const char* key) noexcept;
	Setting(const char* section, const char* key, int32_t value) noexcept;
	Setting(const char* section, const char* key, float value) noexcept;
	Setting(const char* section, const char* key, bool value) noexcept;

	// Methods
	// --------------------------------------------------------------------------------------------

	inline const StackString64& section() const noexcept { return mSection; }
	inline const StackString192& key() const noexcept { return mKey; }
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

	StackString64 mSection;
	StackString192 mKey;
	SettingType mType;
	union {
		int32_t i;
		float f;
		bool b;
	};
};

} // namespace sfz
