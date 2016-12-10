// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#include "phantasy_engine/ecs/EcsWrapper.hpp"

#include <algorithm>

namespace phe {

// EcsWrapper: Constructors & destructors
// ------------------------------------------------------------------------------------------------

EcsWrapper::EcsWrapper(uint32_t maxNumEntities) noexcept
{
	rawEcsPtr = sfz::makeShared<EntityComponentSystem>(maxNumEntities);
	renderComponents = EcsComponentAccessor<RenderComponent>(rawEcsPtr);
}

EcsWrapper::EcsWrapper(EcsWrapper&& other) noexcept
{
	this->swap(other);
}

EcsWrapper& EcsWrapper::operator= (EcsWrapper&& other) noexcept
{
	this->swap(other);
	return *this;
}

EcsWrapper::~EcsWrapper() noexcept
{
	this->destroy();
}

// EcsWrapper: State methods
// ------------------------------------------------------------------------------------------------

void EcsWrapper::destroy() noexcept
{
	rawEcsPtr = nullptr;
	renderComponents = EcsComponentAccessor<RenderComponent>();
}

void EcsWrapper::swap(EcsWrapper& other) noexcept
{
	this->rawEcsPtr.swap(other.rawEcsPtr);
	std::swap(this->renderComponents, other.renderComponents);
}

} // namepace phe
