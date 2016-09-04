// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include <sfz/containers/DynArray.hpp>

#include "phantasy_engine/resources/Renderable.hpp"
#include "phantasy_engine/level/PointLight.hpp"

namespace sfz {

struct Scene {
	DynArray<PointLight> staticPointLights, dynamicPointLights;
	DynArray<Renderable> staticRenderables, dynamicRenderables;
};

}
