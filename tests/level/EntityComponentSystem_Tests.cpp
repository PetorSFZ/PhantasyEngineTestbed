// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#include <sfz/PushWarnings.hpp>
#include <catch.hpp>
#include <sfz/PopWarnings.hpp>

#include "phantasy_engine/level/EntityComponentSystem.hpp"

using namespace phe;

TEST_CASE("Entity creation/deletion", "[EntityComponentSystem]")
{
	EntityComponentSystem ecs(10);
	REQUIRE(ecs.maxNumEntities() == 10);

	for (uint32_t i = 0; i < 10; i++) {
		REQUIRE(ecs.createEntity() == i);
		REQUIRE(ecs.currentNumEntities() == (i+1));
	}
	REQUIRE(ecs.currentNumEntities() == 10);
	
	for (uint32_t i = 0; i < 10; i++) {
		ecs.deleteEntity(i);
		REQUIRE(ecs.currentNumEntities() == (10-i-1));
	}
	REQUIRE(ecs.currentNumEntities() == 0);

	for (uint32_t i = 0; i < 10; i++) {
		ecs.createEntity(); // Note: Not defined which index we will get after deletion
		REQUIRE(ecs.currentNumEntities() == (i+1));
	}
	REQUIRE(ecs.currentNumEntities() == 10);
}

TEST_CASE("Component creation/deletion", "[EntityComponentSystem]")
{
	EntityComponentSystem ecs(10);
	const uint32_t e1 = ecs.createEntity();
	const uint32_t e2 = ecs.createEntity();
	const uint32_t e3 = ecs.createEntity();

	REQUIRE(ecs.currentNumComponentTypes() == 0);
	const uint32_t byteComponent = ecs.createComponentTypeRaw(1);
	REQUIRE(ecs.numComponents(byteComponent) == 0);
	REQUIRE(ecs.currentNumComponentTypes() == 1);
	const uint32_t uintComponent = ecs.createComponentTypeRaw(4);
	REQUIRE(ecs.numComponents(uintComponent) == 1);
	REQUIRE(ecs.currentNumComponentTypes() == 2);

	uint8_t tmpByte = 'a';
	ecs.addComponentRaw(e1, byteComponent, &tmpByte);
	REQUIRE(ecs.numComponents(byteComponent) == 1);
	tmpByte = 'c';
	ecs.addComponentRaw(e3, byteComponent, &tmpByte);
	REQUIRE(ecs.numComponents(byteComponent) == 2);

	REQUIRE(*ecs.getComponentRaw(e1, byteComponent) == 'a');
	REQUIRE(*ecs.getComponentRaw(e3, byteComponent) == 'c');
}
