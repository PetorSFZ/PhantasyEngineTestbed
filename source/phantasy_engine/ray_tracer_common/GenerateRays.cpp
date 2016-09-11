// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#include "phantasy_engine/ray_tracer_common/GenerateRays.hpp"

#include <cmath>

#include <sfz/math/MathHelpers.hpp>

namespace phe {

CameraDef generateCameraDef(vec3 camPos, vec3 camDir, vec3 camUp, float vertFovRad, vec2 res) noexcept
{
	camDir = normalize(camDir);
	camUp = normalize(camUp);
	sfz_assert_debug(approxEqual(dot(camDir, camUp), 0.0f));

	// Calculate camRight
	vec3 camRight = normalize(cross(camDir, camUp));
	sfz_assert_debug(approxEqual(dot(camDir, camRight), 0.0f));
	sfz_assert_debug(approxEqual(dot(camUp, camRight), 0.0f));

	// Calculate offset variables
	float aspect = res.x / res.y;
	float yMaxOffset = std::tan(vertFovRad * 0.5f);
	float xMaxOffset = aspect * yMaxOffset;

	// Create and return the CameraDef
	CameraDef cam;
	cam.origin = camPos;
	cam.dir = camDir;
	cam.dX = camRight * xMaxOffset;
	cam.dY = camUp * yMaxOffset;
	return cam;
}

} // namespace phe
