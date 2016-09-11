// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#include <sfz/PushWarnings.hpp>
#include <catch.hpp>
#include <sfz/PopWarnings.hpp>

#include <sfz/math/MathHelpers.hpp>

#include "phantasy_engine/ray_tracer_common/BVHNode.hpp"
#include <phantasy_engine/renderers/cpu_ray_tracer/BVH.hpp>

TEST_CASE("isLeaf() && numTriangles()", "[BVHNode]")
{
	using namespace sfz;
	using namespace phe;

	BVHNode node;
	node.indices[1] = 0u;

	node.indices[0] = 0u;
	REQUIRE(!node.isLeaf());
	REQUIRE(node.numTriangles() == 0u);

	node.indices[0] = 0x80000000u;
	REQUIRE(node.isLeaf());
	REQUIRE(node.numTriangles() == 0u);

	node.indices[0] = 0x80000001u;
	REQUIRE(node.isLeaf());
	REQUIRE(node.numTriangles() == 1u);

	node.indices[0] = 2147483648u;
	REQUIRE(node.isLeaf());
	REQUIRE(node.numTriangles() == 0u);

	node.indices[0] = 2147483647u;
	REQUIRE(!node.isLeaf());
	REQUIRE(node.numTriangles() == 2147483647u);
}

TEST_CASE("setLeaf()", "[BVHNode]")
{
	using namespace phe;

	BVHNode node;

	node.setLeaf(0u, 1u);
	REQUIRE(node.indices[0] == 0x80000000u);
	REQUIRE(node.indices[1] == 1u);

	node.setLeaf(1u, 20u);
	REQUIRE(node.indices[0] == 0x80000001u);
	REQUIRE(node.indices[1] == 20u);
}
