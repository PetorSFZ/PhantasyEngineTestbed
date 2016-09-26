// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include <sfz/geometry/Sphere.hpp>
#include <sfz/math/Vector.hpp>

namespace phe {

using sfz::Sphere;
using sfz::vec3;

struct SphereLight final {
	vec3 pos = vec3(0.0f);
	float radius = 0.0f; // Size of the light emitter, 0 makes it a point light
	float range = 0.0f; // Range of the emitted light
	vec3 strength = vec3(0.0f); // Strength (and color) of light source
	
	// Disjoint shadow states, both should be true for all shadows
	bool staticShadows = false; // Static objects casts shadows
	bool dynamicShadows = false; // Dynamic objects casts shadows

	// Helper methods to get Sphere structs
	inline Sphere emittingSphere() const noexcept { return Sphere(pos, radius); }
	inline Sphere rangeSphere() const noexcept { return Sphere(pos, range); }
};

} // namespace phe
