// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include <sfz/containers/DynArray.hpp>

#include "phantasy_engine/resources/Renderable.hpp"
#include "phantasy_engine/level/PointLight.hpp"

namespace phe {

using sfz::DynArray;

// Static scene
// ------------------------------------------------------------------------------------------------

/// The definition of a static scene
/// In general a static scene is not expected to change during gameplay, as it may result in a
/// longer or shorter load time depending on the renderer used.
/// All renderable components must be defined in world space, i.e. any transforms must be preapplied.
/// Transparent renderable components may be stored in the opaqueComponents list, but they may be
/// completely skipped or rendered incorrectly.
struct StaticScene {
	DynArray<RawImage> images;
	DynArray<GLTexture> textures;
	DynArray<RenderableComponent> opaqueComponents;
	DynArray<RenderableComponent> transparentComponents;

	DynArray<PointLight> pointLights;
};

// Other functions
// ------------------------------------------------------------------------------------------------

void assimpLoadSponza(const char* basePath, const char* fileName, StaticScene& staticScene) noexcept;
void assimpLoadSponza(const char* basePath, const char* fileName, StaticScene& staticScene, const mat4& modelMatrix) noexcept;

} // namespace phe
