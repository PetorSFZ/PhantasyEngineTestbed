// Copyright (c) Peter Hillerström (skipifzero.com, peter@hstroem.se)

#pragma once

#include <sfz/math/Vector.hpp>

#include "resources/RawImage.hpp"

namespace sfz {

// RawMaterial
// ------------------------------------------------------------------------------------------------

struct RawMaterial final {
	vec3 albedoValue = vec3(0.0f);
	RawImage albedoImage;
	float roughnessValue = 0.0f;
	RawImage roughnessImage;
	float metallicValue = 0.0f; // Should be 0 or 1 for most materials
	RawImage metallicImage;
	float specularValue = 0.5f; // Should be 0.5 for 99% of materials according to UE4 docs
	RawImage specularImage;
};

} // namespace sfz
