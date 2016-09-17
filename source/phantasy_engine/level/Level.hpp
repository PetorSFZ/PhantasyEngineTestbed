// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include <sfz/containers/DynArray.hpp>
#include <sfz/memory/SmartPointers.hpp>

#include "phantasy_engine/level/StaticScene.hpp"
#include "phantasy_engine/rendering/Material.hpp"
#include "phantasy_engine/rendering/RawImage.hpp"

namespace phe {

using sfz::DynArray;
using sfz::SharedPtr;

struct Level final {
	DynArray<RawImage> textures;
	DynArray<Material> materials;
	StaticScene staticScene;

	//DynArray<Renderable&transform> opaqueObjects;
	//DynArray<Renderable&transform> transparentObjects;
	//DynArray<PointLight> pointLights;
};

} // namespace phe
