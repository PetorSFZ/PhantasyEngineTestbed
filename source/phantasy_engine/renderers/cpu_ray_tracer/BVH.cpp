// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#include "phantasy_engine/renderers/cpu_ray_tracer/BVH.hpp"

namespace phe {

// C++ container
// ------------------------------------------------------------------------------------------------

BVH buildBVHFromStaticScene(const StaticScene& scene) noexcept
{
	// TODO: Implement
	BVH tmp;

	BVHNode node;
	setAABB(node, vec3(1.0f, 0.0f, 1.0f), vec3(1.0f, 1.0f, 1.0f));
	tmp.nodes.add(node);

	return tmp;
}

} // namespace phe
