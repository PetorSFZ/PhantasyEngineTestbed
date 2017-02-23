// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#include <sfz/PushWarnings.hpp>
#include <catch.hpp>
#include <sfz/PopWarnings.hpp>

#include "phantasy_engine/ecs/EcsComponentAccessor.hpp"
#include "phantasy_engine/ecs/EcsWrapper.hpp"
#include "phantasy_engine/ecs/EntityComponentSystem.hpp"

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
			REQUIRE(ComponentMask::fromRawValue(0, uint64_t(1) << i) == ComponentMask::fromType(uint32_t(i)));
		}
		for (uint64_t i = 0; i < 64; i++) {
			REQUIRE(ComponentMask::fromRawValue(uint64_t(1) << i, 0) == ComponentMask::fromType(uint32_t(i + 64)));
		}
	}
}

TEST_CASE("Entity creation/deletion", "[EntityComponentSystem]")
{
	EntityComponentSystem ecs(10, sfz::getDefaultAllocator());
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
	sfz::SharedPtr<EntityComponentSystem> ecs = sfz::makeSharedDefault<EntityComponentSystem>(10, sfz::getDefaultAllocator());

	const uint32_t e1 = ecs->createEntity();
	const uint32_t e2 = ecs->createEntity();
	const uint32_t e3 = ecs->createEntity();
	ComponentMask existenceMask = ComponentMask::fromType(ECS_EXISTENCE_COMPONENT_TYPE);
	REQUIRE(ecs->componentMask(e1) == existenceMask);
	REQUIRE(ecs->componentMask(e2) == existenceMask);
	REQUIRE(ecs->componentMask(e3) == existenceMask);

	REQUIRE(ecs->currentNumComponentTypes() == 1);
	const uint32_t byteComponent = ecs->createComponentTypeRaw(1);
	REQUIRE(ecs->numComponents(byteComponent) == 0);
	REQUIRE(ecs->currentNumComponentTypes() == 2);

	EcsComponentAccessor<uint32_t> uintAccessor(ecs);
	REQUIRE(uintAccessor.numComponents() == 0);
	REQUIRE(ecs->currentNumComponentTypes() == 3);

	ComponentMask byteMask = ComponentMask::fromType(byteComponent);
	ComponentMask uintMask = uintAccessor.mask();
	REQUIRE(existenceMask != byteMask);
	REQUIRE(existenceMask != uintMask);
	REQUIRE(byteMask != uintMask);

	ComponentMask existenceByteMask = existenceMask | byteMask;
	ComponentMask existenceUintMask = existenceMask | uintMask;
	ComponentMask byteUintMask = byteMask | uintMask;
	ComponentMask existenceByteUintMask = existenceMask | byteUintMask;
	REQUIRE(existenceByteMask.fulfills(existenceMask));
	REQUIRE(!existenceByteMask.fulfills(existenceByteUintMask));

	uint8_t byteTmp = 'a';
	ecs->addComponentRaw(e1, byteComponent, &byteTmp);
	REQUIRE(ecs->numComponents(byteComponent) == 1);
	byteTmp = 'c';
	ecs->addComponentRaw(e3, byteComponent, &byteTmp);
	REQUIRE(ecs->numComponents(byteComponent) == 2);
	REQUIRE(*(const uint8_t*)ecs->getComponentRaw(e1, byteComponent) == 'a');
	REQUIRE(*(const uint8_t*)ecs->getComponentRaw(e3, byteComponent) == 'c');
	REQUIRE(ecs->componentMask(e1) == existenceByteMask);
	REQUIRE(ecs->componentMask(e2) == existenceMask);
	REQUIRE(ecs->componentMask(e3) == existenceByteMask);

	const uint8_t* bytePtr = (const uint8_t*)ecs->componentArrayPtrRaw(byteComponent);
	REQUIRE(bytePtr[0] == 'a');
	REQUIRE(bytePtr[1] == 'c');

	uintAccessor.add(e1, ~0u);
	uintAccessor.add(e3, 42u);
	REQUIRE(uintAccessor.numComponents() == 2);
	REQUIRE(*uintAccessor.get(e1) == ~0u);
	REQUIRE(*uintAccessor.get(e3) == 42u);
	uintAccessor.add(e3, 37u);
	REQUIRE(uintAccessor.numComponents() == 2);
	REQUIRE(*uintAccessor.get(e3) == 37u);
	REQUIRE(ecs->componentMask(e1) == existenceByteUintMask);
	REQUIRE(ecs->componentMask(e2) == existenceMask);
	REQUIRE(ecs->componentMask(e3) == existenceByteUintMask);
	
	const uint32_t* uintPtr = uintAccessor.arrayPtr();
	REQUIRE(uintPtr[0] == ~0u);
	REQUIRE(uintPtr[1] == 37u);

	uintAccessor.remove(e1);
	REQUIRE(uintAccessor.numComponents() == 1);
	REQUIRE(*uintAccessor.get(e3) == 37u);
	REQUIRE(ecs->componentMask(e1) == existenceByteMask);
	REQUIRE(ecs->componentMask(e2) == existenceMask);
	REQUIRE(ecs->componentMask(e3) == existenceByteUintMask);
	REQUIRE(uintPtr[0] == 37u);

	uintAccessor.add(e2, 42u);
	REQUIRE(uintAccessor.numComponents() == 2);
	REQUIRE(*uintAccessor.get(e2) == 42u);
	REQUIRE(*uintAccessor.get(e3) == 37u);
	REQUIRE(uintPtr[0] == 37u);
	REQUIRE(uintPtr[1] == 42u);
}
