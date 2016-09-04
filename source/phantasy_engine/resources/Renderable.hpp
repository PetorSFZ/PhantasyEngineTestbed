// Copyright (c) Peter Hillerström (skipifzero.com, peter@hstroem.se)

#pragma once

#include <sfz/containers/DynArray.hpp>

#include "resources/GLModel.hpp"
#include "resources/GLTexture.hpp"
#include "resources/RawGeometry.hpp"
#include "resources/RawImage.hpp"

namespace sfz {

// Materials
// ------------------------------------------------------------------------------------------------

struct Material final {
	// Albedo is defined in gamma space
	vec3 albedoValue = vec3(0.0f);
	uint32_t albedoIndex = uint32_t(~0);

	// Roughness is defined in linear space
	float roughnessValue = 0.0f;
	uint32_t roughnessIndex = uint32_t(~0);

	// Metallic is defined in linear space
	float metallicValue = 0.0f; // Should be 0 or 1 for most materials
	uint32_t metallicIndex = uint32_t(~0);
};

// Renderable class
// ------------------------------------------------------------------------------------------------

struct RenderableComponent final {
	RawGeometry geometry;
	GLModel glModel;
	Material material;
};

struct Renderable final {
	DynArray<RawImage> images;
	DynArray<GLTexture> textures;
	DynArray<RenderableComponent> components;
};

// Renderable creation functions
// ------------------------------------------------------------------------------------------------

Renderable assimpLoadSponza(const char* basePath, const char* fileName) noexcept;

} // namespace sfz
