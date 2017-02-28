// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#include "phantasy_engine/ecs/EntityComponentSystem.hpp"

#include <algorithm>

#include <sfz/Assert.hpp>
#include <sfz/containers/DynArray.hpp>
#include <sfz/memory/New.hpp>

namespace phe {

using namespace sfz;

// EntityComponentSystemImpl
// ------------------------------------------------------------------------------------------------

struct ComponentType {
	uint32_t bytesPerComponent;
	uint32_t numComponents;
	DynArray<uint32_t> entityTable;
	DynArray<uint8_t> data;
};

class EntityComponentSystemImpl final {
public:
	Allocator* allocator;

	// The maximum number of entities allowed
	uint32_t maxNumEntities;

	// Keeps track of the next index to allocate, and any eventual unallocated slots
	uint32_t nextFreeEntity;
	DynArray<uint32_t> freeSlots;

	// Bitmasks used to determine what components an entity has.
	DynArray<ComponentMask> masks;

	// The components
	DynArray<ComponentType> components;

	EntityComponentSystemImpl(Allocator* allocator, uint32_t maxNumEntities) noexcept
	{
		this->allocator = allocator;
		this->maxNumEntities = maxNumEntities;

		// Initialize number of entities
		this->nextFreeEntity = 0;
		freeSlots.create(1024, allocator);

		// Initialize all bitmasks to 0
		masks.create(maxNumEntities, allocator);
		masks.setSize(maxNumEntities);
		memset(masks.data(), 0, maxNumEntities * sizeof(ComponentMask));

		// Allocate memory for the components and set "existence component"
		components.create(ECS_MAX_NUM_COMPONENT_TYPES, allocator);
		components.add(ComponentType());
		components[0].bytesPerComponent = 0;
		components[0].numComponents = 0;
	}

	~EntityComponentSystemImpl() noexcept
	{

	}
};

// EntityComponentSystem: Constructors & destructors
// ------------------------------------------------------------------------------------------------

EntityComponentSystem::EntityComponentSystem(uint32_t maxNumEntities, Allocator* allocator) noexcept
{
	this->create(maxNumEntities, allocator);
}

EntityComponentSystem::~EntityComponentSystem() noexcept
{
	this->destroy();
}

// EntityComponentSystem: State methods
// ------------------------------------------------------------------------------------------------

void EntityComponentSystem::create(uint32_t maxNumEntities, Allocator* allocator) noexcept
{
	this->destroy();
	mImpl = sfzNew<EntityComponentSystemImpl>(allocator, allocator, maxNumEntities);
}

void EntityComponentSystem::destroy() noexcept
{
	if (mImpl == nullptr) return;
	sfzDelete(mImpl, mImpl->allocator);
	mImpl = nullptr;
}

Allocator* EntityComponentSystem::allocator() const noexcept
{
	if (mImpl == nullptr) return nullptr;
	return mImpl->allocator;
}

// EntityComponentSystem: Entity methods
// ------------------------------------------------------------------------------------------------

uint32_t EntityComponentSystem::createEntity() noexcept
{
	sfz_assert_debug(this->currentNumEntities() < this->maxNumEntities());

	// Get free entity (prefer from free list)
	uint32_t entity;
	if (mImpl->freeSlots.size() > 0u) {
		entity = mImpl->freeSlots.last();
		mImpl->freeSlots.removeLast();
	}
	else {
		entity = mImpl->nextFreeEntity;
		mImpl->nextFreeEntity += 1u;
	}

	// Sets existence bit
	ComponentMask& mask = mImpl->masks[entity];
	mask = ComponentMask::fromType(0);

	return entity;
}

void EntityComponentSystem::deleteEntity(uint32_t entity) noexcept
{
	sfz_assert_debug(entity < mImpl->maxNumEntities);

	// Retrieve mask
	ComponentMask& mask = mImpl->masks[entity];
	
	// Check if entity does not exist (first bit is reserved for entity existence)
	if (mask == ComponentMask::empty()) {
		printErrorMessage("EntityComponentSystem: Trying to delete entity that does not exist.");
		return;
	}

	// Remove all associated components (skip 0, it is existence flag)
	for (uint32_t i = 1; i < this->currentNumComponentTypes(); i++) {
		if (mask.hasComponentType(i)) {
			this->removeComponent(entity, i);
		}
	}

	// Clear mask
	mask = ComponentMask::empty();

	// Add entity to free list if it is not the highest entity index
	if ((entity + 1) == mImpl->nextFreeEntity) {
		mImpl->nextFreeEntity = entity;
	}
	else {
		mImpl->freeSlots.add(entity);
	}
}

const ComponentMask& EntityComponentSystem::componentMask(uint32_t entity) const noexcept
{
	return mImpl->masks[entity];
}

const ComponentMask* EntityComponentSystem::componentMaskArrayPtr() const noexcept
{
	return mImpl->masks.data();
}

uint32_t EntityComponentSystem::maxNumEntities() const noexcept
{
	return mImpl->maxNumEntities;
}

uint32_t EntityComponentSystem::currentNumEntities() const noexcept
{
	return mImpl->nextFreeEntity - mImpl->freeSlots.size();
}

uint32_t EntityComponentSystem::entityIndexUpperBound() const noexcept
{
	return mImpl->nextFreeEntity;
}

// EntityComponentSystem: Raw (non-typesafe) component methods
// ------------------------------------------------------------------------------------------------

uint32_t EntityComponentSystem::createComponentTypeRaw(uint32_t bytesPerComponent) noexcept
{
	// Allocating memory for component type
	ComponentType tmp;
	tmp.bytesPerComponent = bytesPerComponent;
	tmp.entityTable.create(maxNumEntities(), mImpl->allocator);
	tmp.entityTable.addMany(maxNumEntities(), ~0u);
	tmp.data.create(bytesPerComponent * maxNumEntities(), mImpl->allocator);
	tmp.data.setSize(bytesPerComponent * maxNumEntities());
	tmp.numComponents = 0u;

	// Add component type and return index for it
	mImpl->components.add(std::move(tmp));
	return mImpl->components.size() - 1u;
}

uint32_t EntityComponentSystem::currentNumComponentTypes() const noexcept
{
	return mImpl->components.size();
}

void EntityComponentSystem::addComponentRaw(uint32_t entity, uint32_t componentType,
                                            const void* component) noexcept
{
	sfz_assert_debug(componentType < mImpl->components.size());
	ComponentType& compType = mImpl->components[componentType];
	
	// Retrieve component location for given entity
	sfz_assert_debug(entity < mImpl->maxNumEntities);
	uint32_t componentLoc = compType.entityTable[entity];

	// If entity does not have component it needs to be added
	if (componentLoc == ~0u) {
		componentLoc = compType.numComponents;
		compType.numComponents += 1u;
		compType.entityTable[entity] = componentLoc;
		mImpl->masks[entity] = mImpl->masks[entity] | ComponentMask::fromType(componentType);
	}

	// Copy data
	uint8_t* componentPtr = compType.data.data() + (componentLoc * compType.bytesPerComponent);
	memcpy(componentPtr, component, compType.bytesPerComponent);
}

void EntityComponentSystem::removeComponent(uint32_t entity, uint32_t componentType) noexcept
{
	sfz_assert_debug(componentType < mImpl->components.size());
	ComponentType& compType = mImpl->components[componentType];

	// Retrieve component location for given entity
	sfz_assert_debug(entity < mImpl->maxNumEntities);
	uint32_t componentLoc = compType.entityTable[entity];

	// Check if component exists
	if (componentLoc == ~0u) {
		printErrorMessage("EntityComponentSystem: Trying to delete component that does not exist.");
		return;
	}

	// Remove component
	uint8_t* dstPtr = compType.data.data() + compType.bytesPerComponent * componentLoc;
	uint8_t* srcPtr = compType.data.data() + compType.bytesPerComponent * (componentLoc + 1u);
	uint32_t numBytesToMove = (mImpl->maxNumEntities - componentLoc - 1u) * compType.bytesPerComponent;
	std::memmove(dstPtr, srcPtr, numBytesToMove);
	compType.numComponents -= 1;

	compType.entityTable[entity] = ~0u;
	mImpl->masks[entity] = mImpl->masks[entity] & (~ComponentMask::fromType(componentType));

	// Update all indices larger than removed index
	for (uint32_t i = 0; i < this->entityIndexUpperBound(); i++) {
		uint32_t compLoc = compType.entityTable[i];
		if (compLoc != ~0u && compLoc > componentLoc) {
			compType.entityTable[i] = compLoc - 1u;
		}
	}
}

void* EntityComponentSystem::componentArrayPtrRaw(uint32_t componentType) noexcept
{
	sfz_assert_debug(componentType < mImpl->components.size());
	return mImpl->components[componentType].data.data();
}

const void* EntityComponentSystem::componentArrayPtrRaw(uint32_t componentType) const noexcept
{
	sfz_assert_debug(componentType < mImpl->components.size());
	return mImpl->components[componentType].data.data();
}

uint32_t EntityComponentSystem::numComponents(uint32_t componentType) noexcept
{
	sfz_assert_debug(componentType < mImpl->components.size());
	return mImpl->components[componentType].numComponents;
}

void* EntityComponentSystem::getComponentRaw(uint32_t entity, uint32_t componentType) noexcept
{
	sfz_assert_debug(componentType < mImpl->components.size());
	ComponentType& compType = mImpl->components[componentType];

	// Retrieve component location for given entity
	sfz_assert_debug(entity < mImpl->maxNumEntities);
	uint32_t componentLoc = compType.entityTable[entity];

	// Check if component exists
	if (componentLoc == ~0u) {
		printErrorMessage("EntityComponentSystem: Trying to access component that does not exist.");
		return nullptr;
	}

	return compType.data.data() + (componentLoc * compType.bytesPerComponent);
}

const void* EntityComponentSystem::getComponentRaw(uint32_t entity,
                                                   uint32_t componentType) const noexcept
{
	sfz_assert_debug(componentType < mImpl->components.size());
	ComponentType& compType = mImpl->components[componentType];

	// Retrieve component location for given entity
	sfz_assert_debug(entity < mImpl->maxNumEntities);
	uint32_t componentLoc = compType.entityTable[entity];

	// Check if component exists
	if (componentLoc == ~0u) {
		printErrorMessage("EntityComponentSystem: Trying to access component that does not exist.");
		return nullptr;
	}

	return compType.data.data() + (componentLoc * compType.bytesPerComponent);
}

} // namespace phe
