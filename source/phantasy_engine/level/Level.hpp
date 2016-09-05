// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include <sfz/memory/SmartPointers.hpp>

#include "phantasy_engine/level/StaticScene.hpp"

namespace phe {

using sfz::SharedPtr;

struct Level final {
	SharedPtr<StaticScene> staticScene;

	//DynArray<Renderable&transform> opaqueObjects;
	//DynArray<Renderable&transform> transparentObjects;
	//DynArray<PointLight> pointLights;
};

} // namespace phe
