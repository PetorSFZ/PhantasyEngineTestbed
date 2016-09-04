// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#include "phantasy_engine/config/Setting.hpp"

#include <cmath>

#include <sfz/Assert.hpp>

namespace sfz {

// Setting: Constructors & destructors
// ------------------------------------------------------------------------------------------------

Setting::Setting(const char* section, const char* key) noexcept
:
	mSection(section),
	mKey(key)
{
	setInt(0);
}

Setting::Setting(const char* section, const char* key, int32_t value) noexcept
:
	mSection(section),
	mKey(key)
{
	setInt(value);
}

Setting::Setting(const char* section, const char* key, float value) noexcept
:
	mSection(section),
	mKey(key)
{
	setFloat(value);
}

Setting::Setting(const char* section, const char* key, bool value) noexcept
:
	mSection(section),
	mKey(key)
{
	setBool(value);
}

// Setting: Methods
// ------------------------------------------------------------------------------------------------

int32_t Setting::intValue() const noexcept
{
	sfz_assert_debug(mType == SettingType::INT || mType == SettingType::FLOAT);
	if (mType == SettingType::INT) {
		return i;
	}
	return int32_t(std::round(f));
}

float Setting::floatValue() const noexcept
{
	sfz_assert_debug(mType == SettingType::INT || mType == SettingType::FLOAT);
	if (mType == SettingType::FLOAT) {
		return f;
	}
	return float(i);
}

bool Setting::boolValue() const noexcept
{
	sfz_assert_debug(mType == SettingType::BOOL);
	if (mType == SettingType::INT) {
		return i != 0;
	}
	return b;
}

void Setting::setInt(int32_t value) noexcept
{
	this->mType = SettingType::INT;
	this->i = value;
}

void Setting::setFloat(float value) noexcept
{
	this->mType = SettingType::FLOAT;
	this->f = value;
}

void Setting::setBool(bool value) noexcept
{
	this->mType = SettingType::BOOL;
	this->b = value;
}

} // namespace sfz
