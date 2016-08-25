// Copyright (c) Peter Hillerström (skipifzero.com, peter@hstroem.se)

#include "config/GlobalConfig.hpp"

#include <cstring>

#include <sfz/util/IniParser.hpp>
#include <sfz/math/MathHelpers.hpp>

namespace sfz {

// GlobalConfig: Singleton instance
// ------------------------------------------------------------------------------------------------

GlobalConfig& GlobalConfig::instance() noexcept
{
	static GlobalConfig config;
	return config;
}

// GlobalConfig: Methods
// ------------------------------------------------------------------------------------------------

bool GlobalConfig::load(const char* path) noexcept
{
	// Clear previous settings
	mSettings.clear();

	// Load ini file
	IniParser ini(path);
	if (!ini.load()) {
		return false;
	}

	// Create setting items of all ini items
	StackString256 tmpStr;
	for (auto item : ini) {
		size_t sectionLen = std::strlen(item.getSection());
		if (sectionLen == 0) {
			tmpStr.printf("%s", item.getKey());
		}
		else {
			tmpStr.printf("%s.%s", item.getSection(), item.getKey());
		}
		Setting setting(tmpStr.str);
		
		// Get value of setting
		if (item.getFloat() != nullptr) {
			float floatVal = *item.getFloat();
			int32_t intVal = *item.getInt();
			if (approxEqual(floatVal, float(intVal))) {
				setting.setInt(intVal);
			}
			else {
				setting.setFloat(floatVal);
			}
		}
		else if (item.getBool() != nullptr) {
			bool b = *item.getBool();
			setting.setBool(b);
		}

		// Add setting to global list of settings
		mSettings.add(makeShared<Setting>(setting));
	}

	// TODO: Shortcuts to common settings

	return true;
}

bool GlobalConfig::save(const char* path) noexcept
{
	IniParser ini(path);
	StackString64 sectionStr;
	StackString192 keyStr;
	
	// Add settings to temporary IniParser
	for (auto& setting : mSettings) {

		// Find separator between section and key (if it exists)
		const char* ident = setting->identifier();
		size_t identLen = std::strlen(ident);
		size_t separatorIndex = size_t(~0);
		for (size_t i = 0; i < 64 && i < identLen; i++) {
			if (ident[i] == '.') {
				separatorIndex = i;
				break;
			}
		}

		// Separate section and key
		if (separatorIndex == size_t(~0)) {
			sectionStr.printf("");
			sfz_assert_debug(identLen < size_t(192));
			keyStr.printf("%s", ident);
		}
		else {
			sfz_assert_debug(separatorIndex < size_t(64));
			sfz_assert_debug((identLen - separatorIndex) < size_t(192));
			std::memcpy(sectionStr.str, ident, separatorIndex);
			sectionStr.str[separatorIndex] = '\0';
			size_t keyLen = (identLen - separatorIndex - 1);
			std::memcpy(keyStr.str, ident + separatorIndex + 1, keyLen);
			keyStr.str[keyLen] = '\0';
		}

		// Set ini item
		switch (setting->type()) {
		case SettingType::INT:
			ini.setInt(sectionStr.str, keyStr.str, setting->intValue());
			break;
		case SettingType::FLOAT:
			ini.setFloat(sectionStr.str, keyStr.str, setting->floatValue());
			break;
		case SettingType::BOOL:
			ini.setBool(sectionStr.str, keyStr.str, setting->boolValue());
			break;
		}
	}

	// Write to ini
	if (ini.save()) {
		return true;
	}

	return false;
}

SharedPtr<Setting> GlobalConfig::setting(const char* identifier) const noexcept
{
	for (auto& setting : mSettings) {
		if (std::strcmp(setting->identifier(), identifier) == 0) {
			return setting;
		}
	}
	return SharedPtr<Setting>();
}

} // namespace sfz
