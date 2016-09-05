// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#include <sfz/PushWarnings.hpp>
#include <catch.hpp>
#include <sfz/PopWarnings.hpp>

#include <sfz/math/MathHelpers.hpp>

#include <phantasy_engine/renderers/ViewFrustum.hpp>

TEST_CASE("ViewFrustum: Getters", "[ViewFrustum]")
{
	using namespace sfz;

	ViewFrustum frustum(vec3(0.0f, 0.0f, -1.0f), vec3(1.0f, 0.0f, 0.0f), vec3(0.0f, 1.0f, 0.0f), 45.0f, 1.0f, 0.01f, 100.0f);


	REQUIRE(approxEqual(frustum.pos(), vec3(0.0f, 0.0f, -1.0f)));
	REQUIRE(approxEqual(frustum.dir(), vec3(1.0f, 0.0f, 0.0f)));
	REQUIRE(approxEqual(frustum.up(), vec3(0.0f, 1.0f, 0.0f)));
	REQUIRE(approxEqual(frustum.verticalFov(), 45.0f));
	REQUIRE(approxEqual(frustum.aspectRatio(), 1.0f));
	REQUIRE(approxEqual(frustum.near(), 0.01f));
	REQUIRE(approxEqual(frustum.far(), 100.0f));
}
