// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include <sfz/math/Vector.hpp>

namespace phe {

using sfz::vec4;

// Material
// ------------------------------------------------------------------------------------------------

struct Material final {
	// Albedo is defined in gamma space, transparency is linear.
	vec4 albedoValue = vec4(0.0f);
	uint32_t albedoIndex = uint32_t(~0);

	// Roughness is defined in linear space
	float roughnessValue = 0.0f;
	uint32_t roughnessIndex = uint32_t(~0);

	// Metallic is defined in linear space
	float metallicValue = 0.0f; // Should be 0 or 1 for most materials
	uint32_t metallicIndex = uint32_t(~0);
};

} // namespace phe
