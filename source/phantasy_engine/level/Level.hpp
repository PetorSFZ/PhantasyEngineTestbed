// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include <sfz/containers/DynArray.hpp>
#include <sfz/containers/HashMap.hpp>

#include "phantasy_engine/level/EcsWrapper.hpp"
#include "phantasy_engine/level/StaticScene.hpp"
#include "phantasy_engine/rendering/Material.hpp"
#include "phantasy_engine/rendering/RawImage.hpp"
#include "phantasy_engine/rendering/RawMesh.hpp"

namespace phe {

using sfz::DynArray;
using sfz::HashMap;

// Level struct
// ------------------------------------------------------------------------------------------------

struct Level final {
	// Textures and materials, shared between static and dynamic objects
	DynArray<RawImage> textures;
	HashMap<size_t, uint32_t> texMapping;
	DynArray<Material> materials;

	// Static scene, contains static mesh and lights
	StaticScene staticScene;

	// Dynamic meshes
	DynArray<RawMesh> meshes;
	DynArray<SphereLight> sphereLights;

	// Entity component system
	EcsWrapper ecs;
};

} // namespace phe
