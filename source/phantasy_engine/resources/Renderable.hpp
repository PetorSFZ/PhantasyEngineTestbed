// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include <sfz/containers/DynArray.hpp>
#include <sfz/math/Matrix.hpp>

#include "phantasy_engine/resources/GLModel.hpp"
#include "phantasy_engine/resources/GLTexture.hpp"
#include "phantasy_engine/rendering/Material.hpp"
#include "phantasy_engine/rendering/RawGeometry.hpp"
#include "phantasy_engine/rendering/RawImage.hpp"

namespace phe {

using sfz::DynArray;
using sfz::mat4;
using sfz::vec3;

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

// Other functions
// ------------------------------------------------------------------------------------------------

void modelToWorldSpace(Renderable& renderable, const mat4& modelMatrix) noexcept;

} // namespace phe
