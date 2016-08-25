// Copyright (c) Peter Hillerström (skipifzero.com, peter@hstroem.se)

#include "config/Setting.hpp"

#include <sfz/Assert.hpp>

namespace sfz {

// Setting: Constructors & destructors
// ------------------------------------------------------------------------------------------------

Setting::Setting() noexcept
:
	mIdent("INVALID")
{
	setInt(0);
}

Setting::Setting(const char* identifier) noexcept
:
	mIdent(identifier)
{
	setBool(false);
}

Setting::Setting(const char* identifier, int32_t value) noexcept
:
	mIdent(identifier)
{
	setInt(value);
}

Setting::Setting(const char* identifier, float value) noexcept
:
	mIdent(identifier)
{
	setFloat(value);
}

Setting::Setting(const char* identifier, bool value) noexcept
:
	mIdent(identifier)
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
	return int32_t(f);
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
