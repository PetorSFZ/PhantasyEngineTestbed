// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#include "phantasy_engine/level/EntityComponentSystem.hpp"

#include <algorithm>

#include <sfz/Assert.hpp>
#include <sfz/containers/DynArray.hpp>
#include <sfz/memory/New.hpp>

namespace phe {

using namespace sfz;

// EntityComponentSystemImpl
// ------------------------------------------------------------------------------------------------

struct ComponentStruct {
	DynArray<uint8_t> data;
	uint32_t bytesPerComponent;
};

class EntityComponentSystemImpl final {
public:
	// The maximum number of entities allowed
	uint32_t maxNumEntities;

	// The current number of entities
	uint32_t currentNumEntities = 0;

	//The current number of component types
	uint32_t currentNumComponentTypes = 0;

	// All currently free entity indices in this system
	DynArray<uint32_t> freeEntitiesStack;



	DynArray<uint8_t> mask;
	DynArray<ComponentStruct> components;


	EntityComponentSystemImpl(uint32_t maxNumEntities) noexcept
	{
		this->maxNumEntities = maxNumEntities;

		// Create and fill freeEntities with all free indices
		freeEntitiesStack.setCapacity(maxNumEntities);
		freeEntitiesStack.setSize(maxNumEntities);
		for (uint32_t i = 0; i < maxNumEntities; i++) {
			freeEntitiesStack[i] = maxNumEntities - i - 1u;
		}


	}

	~EntityComponentSystemImpl() noexcept
	{

	}
};

// EntityComponentSystem: Constructors & destructors
// ------------------------------------------------------------------------------------------------

EntityComponentSystem::EntityComponentSystem(uint32_t maxNumEntitiesIn) noexcept
{
	mImpl = sfz_new<EntityComponentSystemImpl>(maxNumEntitiesIn);
}

EntityComponentSystem::EntityComponentSystem(EntityComponentSystem&& other) noexcept
{
	this->swap(other);
}

EntityComponentSystem& EntityComponentSystem::operator= (EntityComponentSystem&& other) noexcept
{
	this->swap(other);
	return *this;
}

EntityComponentSystem::~EntityComponentSystem() noexcept
{
	this->destroy();
}

// EntityComponentSystem: State methods & getters
// ------------------------------------------------------------------------------------------------

void EntityComponentSystem::destroy() noexcept
{
	sfz_delete(mImpl);
	mImpl = nullptr;
}

void EntityComponentSystem::swap(EntityComponentSystem& other) noexcept
{
	std::swap(this->mImpl, other.mImpl);
}

uint32_t EntityComponentSystem::maxNumEntities() const noexcept
{
	return mImpl->maxNumEntities;
}

uint32_t EntityComponentSystem::currentNumEntities() const noexcept
{
	return mImpl->currentNumEntities;
}

uint32_t EntityComponentSystem::currentNumComponentTypes() const noexcept
{
	return mImpl->currentNumComponentTypes;
}

// EntityComponentSystem: Entity creation/deletion
// ------------------------------------------------------------------------------------------------

uint32_t EntityComponentSystem::createEntity() noexcept
{
	sfz_assert_release(mImpl->freeEntitiesStack.size() > 0);
	uint32_t entity = mImpl->freeEntitiesStack.last();
	mImpl->freeEntitiesStack.removeLast();
	return entity;
}

void EntityComponentSystem::deleteEntity(uint32_t entity) noexcept
{
	// TODO:
	// - Remove all associated components
	// - Clear associated mask
	// - Add to freeEntitiesStack
}

// EntityComponentSystem: Raw (non-typesafe) methods
// ------------------------------------------------------------------------------------------------

uint32_t EntityComponentSystem::createComponentTypeRaw(uint32_t bytesPerComponent) noexcept
{
	return 0;
}

void* EntityComponentSystem::getComponentRaw(uint32_t entity, uint32_t componentType) noexcept
{
	return nullptr;
}

const void* EntityComponentSystem::getComponentRaw(uint32_t entity, uint32_t componentType) const noexcept
{
	return nullptr;
}

} // namespace phe
