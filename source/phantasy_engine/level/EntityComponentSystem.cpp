// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#include "phantasy_engine/level/EntityComponentSystem.hpp"

#include <algorithm>

#include <sfz/containers/DynArray.hpp>
#include <sfz/memory/New.hpp>

namespace phe {

using namespace sfz;

// EntityComponentSystemImpl
// ------------------------------------------------------------------------------------------------

class EntityComponentSystemImpl final {
public:

	uint32_t maxNumEntities = 0;
	uint32_t maxNumComponents = 0;
	uint32_t currentNumComponents = 0;
	DynArray<DynArray<uint8_t>> data;


	EntityComponentSystemImpl(uint32_t maxNumEntities) noexcept
	{

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

uint32_t EntityComponentSystem::maxNumComponents() const noexcept
{
	return mImpl->maxNumComponents;
}

uint32_t EntityComponentSystem::currentNumComponents() const noexcept
{
	return mImpl->currentNumComponents;
}

// EntityComponentSystem: Raw (non-typesafe) methods
// ------------------------------------------------------------------------------------------------

void* EntityComponentSystem::getComponentRaw(uint32_t entity, uint32_t componentType,
                                             uint32_t bytesPerComponent) noexcept
{
	// TODO: Implement
	return nullptr;
}

const void* EntityComponentSystem::getComponentRaw(uint32_t entity, uint32_t componentType,
                                                   uint32_t bytesPerComponent) const noexcept
{
	// TODO: Implement
	return nullptr;
}

} // namespace phe
