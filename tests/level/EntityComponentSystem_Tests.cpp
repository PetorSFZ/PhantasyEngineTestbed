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
	ComponentMask existenceMask = ComponentMask::fromType(ECS_EXISTENCE_COMPONENT_TYPE);
	REQUIRE(ecs.componentMask(e1) == existenceMask);
	REQUIRE(ecs.componentMask(e2) == existenceMask);
	REQUIRE(ecs.componentMask(e3) == existenceMask);

	REQUIRE(ecs.currentNumComponentTypes() == 1);
	const uint32_t byteComponent = ecs.createComponentTypeRaw(1);
	REQUIRE(ecs.numComponents(byteComponent) == 0);
	REQUIRE(ecs.currentNumComponentTypes() == 2);
	const uint32_t uintComponent = ecs.createComponentTypeRaw(4);
	REQUIRE(ecs.numComponents(uintComponent) == 0);
	REQUIRE(ecs.currentNumComponentTypes() == 3);

	ComponentMask byteMask = ComponentMask::fromType(byteComponent);
	ComponentMask uintMask = ComponentMask::fromType(uintComponent);
	REQUIRE(existenceMask != byteMask);
	REQUIRE(existenceMask != uintMask);
	REQUIRE(byteMask != uintMask);

	ComponentMask existenceByteMask = existenceMask | byteMask;
	ComponentMask existenceUintMask = existenceMask | uintMask;
	ComponentMask byteUintMask = byteMask | uintMask;
	ComponentMask existenceByteUintMask = existenceMask | byteUintMask;
	REQUIRE(existenceByteMask.fulfills(existenceMask));
	REQUIRE(!existenceByteMask.fulfills(existenceByteUintMask));

	ecs.addComponent<uint8_t>(e1, byteComponent, 'a');
	REQUIRE(ecs.numComponents(byteComponent) == 1);
	ecs.addComponent<uint8_t>(e3, byteComponent, 'c');
	REQUIRE(ecs.numComponents(byteComponent) == 2);
	REQUIRE(*ecs.getComponent<uint8_t>(e1, byteComponent) == 'a');
	REQUIRE(*ecs.getComponent<uint8_t>(e3, byteComponent) == 'c');
	REQUIRE(ecs.componentMask(e1) == existenceByteMask);
	REQUIRE(ecs.componentMask(e2) == existenceMask);
	REQUIRE(ecs.componentMask(e3) == existenceByteMask);

	const uint8_t* bytePtr = ecs.componentArrayPtr<uint8_t>(byteComponent);
	REQUIRE(bytePtr[0] == 'a');
	REQUIRE(bytePtr[1] == 'c');

	ecs.addComponent<uint32_t>(e1, uintComponent, ~0u);
	ecs.addComponent<uint32_t>(e3, uintComponent, 42u);
	REQUIRE(ecs.numComponents(uintComponent) == 2);
	REQUIRE(*ecs.getComponent<uint32_t>(e1, uintComponent) == ~0u);
	REQUIRE(*ecs.getComponent<uint32_t>(e3, uintComponent) == 42u);
	ecs.addComponent(e3, uintComponent, 37u);
	REQUIRE(ecs.numComponents(uintComponent) == 2);
	REQUIRE(*ecs.getComponent<uint32_t>(e3, uintComponent) == 37u);
	REQUIRE(ecs.componentMask(e1) == existenceByteUintMask);
	REQUIRE(ecs.componentMask(e2) == existenceMask);
	REQUIRE(ecs.componentMask(e3) == existenceByteUintMask);
	
	const uint32_t* uintPtr = ecs.componentArrayPtr<uint32_t>(uintComponent);
	REQUIRE(uintPtr[0] == ~0u);
	REQUIRE(uintPtr[1] == 37u);

	ecs.removeComponent(e1, uintComponent);
	REQUIRE(ecs.numComponents(uintComponent) == 1);
	REQUIRE(*ecs.getComponent<uint32_t>(e3, uintComponent) == 37u);
	REQUIRE(ecs.componentMask(e1) == existenceByteMask);
	REQUIRE(ecs.componentMask(e2) == existenceMask);
	REQUIRE(ecs.componentMask(e3) == existenceByteUintMask);
	REQUIRE(uintPtr[0] == 37u);

	ecs.addComponent<uint32_t>(e2, uintComponent, 42u);
	REQUIRE(ecs.numComponents(uintComponent) == 2);
	REQUIRE(*ecs.getComponent<uint32_t>(e2, uintComponent) == 42u);
	REQUIRE(*ecs.getComponent<uint32_t>(e3, uintComponent) == 37u);
	REQUIRE(uintPtr[0] == 37u);
	REQUIRE(uintPtr[1] == 42u);
}
