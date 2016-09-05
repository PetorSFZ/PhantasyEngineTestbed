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
/// All renderables must be defined in world space, i.e. any transforms must be preapplied.
/// Transparent renderables may be stored in the opaqueRenderables list, but they may be
/// completely skipped or rendered incorrectly.
struct StaticScene {
	DynArray<Renderable> opaqueRenderables;
	DynArray<Renderable> transparentRenderables;
	DynArray<PointLight> pointLights;
};

} // namespace phe
