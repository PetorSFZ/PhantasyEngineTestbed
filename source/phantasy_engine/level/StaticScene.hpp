// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include <sfz/containers/DynArray.hpp>

#include "phantasy_engine/level/SphereLight.hpp"
#include "phantasy_engine/ray_tracer_common/BVH.hpp"
#include "phantasy_engine/rendering/RawMesh.hpp"

namespace phe {

using sfz::DynArray;

// Static scene
// ------------------------------------------------------------------------------------------------

/// In general a static scene is not expected to change during gameplay, as it may result in a
/// longer or shorter load time depending on the renderer used.
/// All meshes must be defined in world space, i.e. any transforms must be preapplied. Transparent
/// objects may be completely skipped or rendered incorrectly.
struct StaticScene {
	DynArray<RawMesh> meshes;
	DynArray<SphereLight> sphereLights;
	BVH bvh; // BVH to be used for raycasting
};

} // namespace phe
