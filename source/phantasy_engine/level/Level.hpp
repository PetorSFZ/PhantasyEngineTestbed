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

struct DynObject {
	uint32_t meshIndex, numMeshes;
};

struct DynObjectInstance {
	uint32_t id, objectIndex;
	sfz::mat4 transform;
};

struct Level final {
	DynArray<RawImage> textures;
	DynArray<Material> materials;
	StaticScene staticScene;

	DynArray<RawMesh> dynamicMeshes;
	// A specific object type has a set number of meshes and points to its index in the dynamic mesh list
	DynArray<DynObject> dynamicObjects;
	// Every instance of an object has a unique id, points to its object type and has a transform
	DynArray<DynObjectInstance> dynamicObjectInstances;

	//DynArray<Renderable&transform> opaqueObjects;
	//DynArray<Renderable&transform> transparentObjects;
	//DynArray<PointLight> pointLights;
};

} // namespace phe
