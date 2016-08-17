// Copyright (c) Peter Hillerström (skipifzero.com, peter@hstroem.se)

#pragma once

#include "resources/GLTexture.hpp"

namespace sfz {

// GLMaterial
// ------------------------------------------------------------------------------------------------

struct GLMaterial final {
	GLTexture albedoTexture;
	GLTexture roughnessTexture;
	GLTexture metallicTexture;
	GLTexture specularTexture;
};

} // namespace sfz
