// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#include <sfz/PushWarnings.hpp>
#include <catch.hpp>
#include <sfz/PopWarnings.hpp>

#include <sfz/math/MathHelpers.hpp>

#include <phantasy_engine/renderers/cpu_ray_tracer/BVH.hpp>

TEST_CASE("isLeaf() && numTriangles()", "[BVH]")
{
	using namespace sfz;
	using namespace phe;

	BVHNode node;
	node.indices[1] = 0u;

	node.indices[0] = 0u;
	REQUIRE(!isLeaf(node));
	REQUIRE(numTriangles(node) == 0u);

	node.indices[0] = 0x80000000u;
	REQUIRE(isLeaf(node));
	REQUIRE(numTriangles(node) == 0u);

	node.indices[0] = 0x80000001u;
	REQUIRE(isLeaf(node));
	REQUIRE(numTriangles(node) == 1u);

	node.indices[0] = 2147483648u;
	REQUIRE(isLeaf(node));
	REQUIRE(numTriangles(node) == 0u);

	node.indices[0] = 2147483647u;
	REQUIRE(!isLeaf(node));
	REQUIRE(numTriangles(node) == 2147483647u);
}

TEST_CASE("setLeaf()", "[BVH]")
{
	using namespace phe;

	BVHNode node;

	setLeaf(node, 0u, 1u);
	REQUIRE(node.indices[0] == 0x80000000u);
	REQUIRE(node.indices[1] == 1u);

	setLeaf(node, 1u, 20u);
	REQUIRE(node.indices[0] == 0x80000001u);
	REQUIRE(node.indices[1] == 20u);
}
