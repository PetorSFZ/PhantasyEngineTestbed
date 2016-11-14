// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#include <sfz/PushWarnings.hpp>
#include <catch.hpp>
#include <sfz/PopWarnings.hpp>

#include "phantasy_engine/level/EntityComponentSystem.hpp"

using namespace phe;

TEST_CASE("ComponentMask", "[EntityComponentSystem]")
{
	SECTION("fromRawValue()") {
		auto m1 = ComponentMask::fromRawValue(0, 0);
		for (uint32_t i = 0; i < 16; i++) {
			REQUIRE(m1.rawMask[i] == uint8_t(0));
		}

		auto m2 = ComponentMask::fromRawValue(0, 1);
		REQUIRE(m2.rawMask[0] == uint8_t(1));
		for (uint32_t i = 1; i < 16; i++) {
			REQUIRE(m2.rawMask[i] == uint8_t(0));
		}

		auto m3 = ComponentMask::fromRawValue(0, uint64_t(0x8000000000000000));
		for (uint32_t i = 0; i < 7; i++) {
			REQUIRE(m3.rawMask[i] == uint8_t(0));
		}
		REQUIRE(m3.rawMask[7] == uint8_t(0x80));
		for (uint32_t i = 8; i < 16; i++) {
			REQUIRE(m3.rawMask[i] == uint8_t(0));
		}

		auto m4 = ComponentMask::fromRawValue(1, 0);
		for (uint32_t i = 0; i < 8; i++) {
			REQUIRE(m4.rawMask[i] == uint8_t(0));
		}
		REQUIRE(m4.rawMask[8] == uint8_t(1));
		for (uint32_t i = 9; i < 16; i++) {
			REQUIRE(m4.rawMask[i] == uint8_t(0));
		}

		auto m5 = ComponentMask::fromRawValue(uint64_t(0x8000000000000000), 0);
		for (uint32_t i = 0; i < 15; i++) {
			REQUIRE(m5.rawMask[i] == uint8_t(0));
		}
		REQUIRE(m5.rawMask[15] == uint8_t(0x80));
	}
	SECTION("fromType()") {
		for (uint64_t i = 0; i < 64; i++) {
			REQUIRE(ComponentMask::fromRawValue(0, uint64_t(1) << i) == ComponentMask::fromType(i));
		}
		for (uint64_t i = 0; i < 64; i++) {
			REQUIRE(ComponentMask::fromRawValue(uint64_t(1) << i, 0) == ComponentMask::fromType(i + 64));
		}
	}
}

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

	REQUIRE(ecs.currentNumComponentTypes() == 1);
	const uint32_t byteComponent = ecs.createComponentTypeRaw(1);
	REQUIRE(ecs.numComponents(byteComponent) == 0);
	REQUIRE(ecs.currentNumComponentTypes() == 2);
	const uint32_t uintComponent = ecs.createComponentTypeRaw(4);
	REQUIRE(ecs.numComponents(uintComponent) == 0);
	REQUIRE(ecs.currentNumComponentTypes() == 3);

	uint8_t tmpByte = 'a';
	ecs.addComponentRaw(e1, byteComponent, &tmpByte);
	REQUIRE(ecs.numComponents(byteComponent) == 1);
	tmpByte = 'c';
	ecs.addComponentRaw(e3, byteComponent, &tmpByte);
	REQUIRE(ecs.numComponents(byteComponent) == 2);

	REQUIRE(*(const uint8_t*)ecs.getComponentRaw(e1, byteComponent) == 'a');
	REQUIRE(*(const uint8_t*)ecs.getComponentRaw(e3, byteComponent) == 'c');
}
