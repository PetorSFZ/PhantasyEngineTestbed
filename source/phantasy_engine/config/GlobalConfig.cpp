// Copyright (c) Peter Hillerström (skipifzero.com, peter@hstroem.se)

#include "config/GlobalConfig.hpp"

#include <cstring>

#include <sfz/containers/DynArray.hpp>
#include <sfz/util/IniParser.hpp>
#include <sfz/math/MathHelpers.hpp>
#include <sfz/memory/SmartPointers.hpp>

namespace sfz {

// GlobalConfig: Statics
// ------------------------------------------------------------------------------------------------

static void setWindowCfg(GlobalConfig& g, WindowConfig& cfg) noexcept
{
	cfg.displayIndex = g.sanitizeInt("Window", "displayIndex", 0, 0, 32);
	cfg.fullscreenMode = g.sanitizeInt("Window", "fullscreenMode", 0, 0, 2); // 0 = off, 1 = windowed, 2 = exclusive
	cfg.vsync = g.sanitizeInt("Window", "vsync", 1, 0, 2); // 0 = off, 1 = on, 2 = swap control tear
	cfg.width = g.sanitizeInt("Window", "width", 1280, 320, 3840);
	cfg.height = g.sanitizeInt("Window", "height", 720, 240, 2160);
	cfg.maximized = g.sanitizeBool("Window", "maximized", false);
	cfg.screenGamma = g.sanitizeFloat("Window", "screenGamma", 2.2f, 1.0f, 3.0f);
}

static void setGraphicsCfg(GlobalConfig& g, GraphicsConfig& cfg) noexcept
{
	cfg.renderingBackend = g.sanitizeInt("Graphics", "renderingBackend", 0, 0, 2);
	cfg.useNativeTargetResolution = g.sanitizeBool("Graphics", "useNativeTargetResolution", true);
	cfg.targetResolutionHeight = g.sanitizeInt("Graphics", "targetResolutionHeight", 720, 120, 4320);
}

// GraphicConfig methods
// ------------------------------------------------------------------------------------------------

vec2i GraphicsConfig::getTargetResolution(vec2i drawableDim) const noexcept
{
	vec2i result = drawableDim;
	if (!this->useNativeTargetResolution->boolValue()) {
		int32_t h = this->targetResolutionHeight->intValue();
		int32_t w = int32_t(std::round(float(h) * float(drawableDim.x) / float(drawableDim.y)));
		result = vec2i(w, h);
	}
	return max(result, vec2i(1,1));
}

// GlobalConfigImpl
// ------------------------------------------------------------------------------------------------

class GlobalConfigImpl final {
public:
	// Members
	// --------------------------------------------------------------------------------------------

	IniParser mIni;
	DynArray<UniquePtr<Setting>> mSettings;
	WindowConfig mWindowCfg;
	GraphicsConfig mGraphicsCfg;
	bool mLoaded = false; // Can only be loaded once... for now

	// Constructors & destructors
	// --------------------------------------------------------------------------------------------

	GlobalConfigImpl() noexcept = default;
	GlobalConfigImpl(const GlobalConfigImpl&) = delete;
	GlobalConfigImpl& operator= (const GlobalConfigImpl&) = delete;
	GlobalConfigImpl(GlobalConfigImpl&&) = delete;
	GlobalConfigImpl& operator= (GlobalConfigImpl&&) = delete;
	~GlobalConfigImpl() noexcept = default;
};

// GlobalConfig: Singleton instance
// ------------------------------------------------------------------------------------------------

GlobalConfig& GlobalConfig::instance() noexcept
{
	static GlobalConfig config;
	return config;
}

// GlobalConfig: Methods
// ------------------------------------------------------------------------------------------------

void GlobalConfig::init(const char* basePath, const char* fileName) noexcept
{
	if (mImpl != nullptr) this->destroy();
	mImpl = sfz_new<GlobalConfigImpl>();

	// Initialize IniParser with path
	StackString256 tmpPath;
	tmpPath.printf("%s%s", basePath, fileName);
	mImpl->mIni = IniParser(tmpPath.str);
}

void GlobalConfig::destroy() noexcept
{
	if (mImpl == nullptr) return;
	sfz_delete<GlobalConfigImpl>(mImpl);
	mImpl = nullptr;
}

bool GlobalConfig::load() noexcept
{
	sfz_assert_debug(mImpl != nullptr);
	sfz_assert_debug(!mImpl->mLoaded); // TODO: Make it possible to reload settings from file

	// Load ini file
	IniParser& ini = mImpl->mIni;
	ini.load();

	// Create setting items of all ini items
	for (auto item : ini) {
		
		// Create new setting
		mImpl->mSettings.add(makeUnique<Setting>(item.getSection(), item.getKey()));
		Setting& setting = *mImpl->mSettings.last();

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
	}

	// Fill specific setting structs
	setWindowCfg(*this, mImpl->mWindowCfg);
	setGraphicsCfg(*this, mImpl->mGraphicsCfg);

	mImpl->mLoaded = true;
	return true;
}

bool GlobalConfig::save() noexcept
{
	sfz_assert_debug(mImpl != nullptr);
	IniParser& ini = mImpl->mIni;
	
	// Update internal ini with the current values of the setting
	for (auto& setting : mImpl->mSettings) {
		switch (setting->type()) {
		case SettingType::INT:
			ini.setInt(setting->section().str, setting->key().str, setting->intValue());
			break;
		case SettingType::FLOAT:
			ini.setFloat(setting->section().str, setting->key().str, setting->floatValue());
			break;
		case SettingType::BOOL:
			ini.setBool(setting->section().str, setting->key().str, setting->boolValue());
			break;
		}
	}

	// Write to ini
	if (ini.save()) {
		return true;
	}

	return false;
}

Setting* GlobalConfig::getCreateSetting(const char* section, const char* key, bool* created) noexcept
{
	Setting* setting = this->getSetting(section, key);
	
	if (setting != nullptr) {
		if (created != nullptr) *created = false;
		return setting;
	}

	mImpl->mSettings.add(makeUnique<Setting>(section, key));
	if (created != nullptr) *created = true;
	return mImpl->mSettings.last().get();
}

// GlobalConfig: Getters
// ------------------------------------------------------------------------------------------------

Setting* GlobalConfig::getSetting(const char* section, const char* key) noexcept
{
	sfz_assert_debug(mImpl != nullptr);
	for (auto& setting : mImpl->mSettings) {
		if (setting->section() == section && setting->key() == key) {
			return setting.get();
		}
	}
	return nullptr;
}

Setting* GlobalConfig::getSetting(const char* key) noexcept
{
	return this->getSetting("", key);
}

void GlobalConfig::getSettings(DynArray<Setting*>& settings) noexcept
{
	sfz_assert_debug(mImpl != nullptr);
	settings.ensureCapacity(mImpl->mSettings.size());
	for (auto& setting : mImpl->mSettings) {
		settings.add(setting.get());
	}
}

const WindowConfig& GlobalConfig::windowCfg() const noexcept
{
	sfz_assert_debug(mImpl != nullptr);
	return mImpl->mWindowCfg;
}

const GraphicsConfig& GlobalConfig::graphcisCfg() const noexcept
{
	sfz_assert_debug(mImpl != nullptr);
	return mImpl->mGraphicsCfg;
}

// GlobalConfig: Sanitizers
// ------------------------------------------------------------------------------------------------

Setting* GlobalConfig::sanitizeInt(const char* section, const char* key,
                                   int32_t defaultValue,
                                   int32_t minValue,
                                   int32_t maxValue) noexcept
{
	sfz_assert_debug(mImpl != nullptr);

	bool created = false;
	Setting* setting = getCreateSetting(section, key, &created);

	// Set default value if created
	if (created) {
		setting->setInt(defaultValue);
		return setting;
	}

	// Make sure setting is of correct type
	if (setting->type() != SettingType::INT) {
		if (setting->type() == SettingType::FLOAT) {
			setting->setInt(setting->intValue());
		} else {
			setting->setInt(defaultValue);
			return setting;
		}
	}

	// Ensure value is in range
	int32_t val = setting->intValue();
	val = std::min(std::max(val, minValue), maxValue);
	setting->setInt(val);

	return setting;
}

Setting* GlobalConfig::sanitizeFloat(const char* section, const char* key,
                                     float defaultValue,
                                     float minValue,
                                     float maxValue) noexcept
{
	bool created = false;
	Setting* setting = getCreateSetting(section, key, &created);

	// Set default value if created
	if (created) {
		setting->setFloat(defaultValue);
		return setting;
	}

	// Make sure setting is of correct type
	if (setting->type() != SettingType::FLOAT) {
		if (setting->type() == SettingType::INT) {
			setting->setFloat(setting->floatValue());
		} else {
			setting->setFloat(defaultValue);
			return setting;
		}
	}

	// Ensure value is in range
	float val = setting->floatValue();
	val = std::min(std::max(val, minValue), maxValue);
	setting->setFloat(val);

	return setting;
}

Setting* GlobalConfig::sanitizeBool(const char* section, const char* key,
                                    bool defaultValue) noexcept
{
	bool created = false;
	Setting* setting = getCreateSetting(section, key, &created);

	// Set default value if created
	if (created) {
		setting->setBool(defaultValue);
		return setting;
	}

	// Make sure setting is of correct type
	if (setting->type() != SettingType::BOOL) {
		if (setting->type() == SettingType::INT) {
			setting->setBool(setting->boolValue());
		} else {
			setting->setBool(defaultValue);
		}
	}
	return setting;
}

// GlobalConfig: Private constructors & destructors
// ------------------------------------------------------------------------------------------------

GlobalConfig::~GlobalConfig() noexcept
{
	this->destroy();
}

} // namespace sfz
