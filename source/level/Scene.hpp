#pragma once

#include <level/PointLight.hpp>
#include <sfz/containers/DynArray.hpp>
#include <resources/Renderable.hpp>

namespace sfz {

struct Scene {
	DynArray<PointLight> staticPointLights, dynamicPointLights;
	DynArray<Renderable> staticRenderables, dynamicRenderables;
};

}
