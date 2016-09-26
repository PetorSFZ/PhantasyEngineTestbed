// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include <sfz/containers/DynArray.hpp>

#include "phantasy_engine/level/DynObject.hpp"
#include "phantasy_engine/level/StaticScene.hpp"
#include "phantasy_engine/rendering/Material.hpp"
#include "phantasy_engine/rendering/RawImage.hpp"
#include "phantasy_engine/rendering/RawMesh.hpp"

namespace phe {

using sfz::DynArray;

struct Level final {
	// Textures and materials, shared between static and dynamic objects
	DynArray<RawImage> textures;
	DynArray<Material> materials;

	// Static scene, contains static mesh and lights
	StaticScene staticScene;

	// Dynamic meshes
	DynArray<RawMesh> meshes;
	DynArray<SphereLight> sphereLights;

	// Temporary list of dynamic objects, will not necessarily be stored
	// like this in the future
	DynArray<DynObject> objects;
};

} // namespace phe
