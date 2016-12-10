// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include <cstdint>

#include <sfz/math/Matrix.hpp>
#include <sfz/math/Vector.hpp>
#include <sfz/memory/SmartPointers.hpp>

#include "phantasy_engine/ecs/EntityComponentSystem.hpp"

namespace phe {

using sfz::mat4;
using sfz::SharedPtr;
using sfz::vec3;

// EcsComponentAccessor
// ------------------------------------------------------------------------------------------------

/// Support class for EntityComponentSystem responsible for providing typesafe access to
/// components with types known at compile time.
template<typename T>
class EcsComponentAccessor final {
public:
	// Constructors & destructors
	// --------------------------------------------------------------------------------------------

	EcsComponentAccessor() noexcept = default;
	EcsComponentAccessor(const EcsComponentAccessor&) noexcept = default;
	EcsComponentAccessor& operator= (const EcsComponentAccessor&) noexcept = default;
	EcsComponentAccessor(EcsComponentAccessor&&) noexcept = default;
	EcsComponentAccessor& operator= (EcsComponentAccessor&&) noexcept = default;
	~EcsComponentAccessor() noexcept = default;
	
	/// Creates a new component type of the specified class in the parameter ecs.
	explicit EcsComponentAccessor(const SharedPtr<EntityComponentSystem>& ecsPtr) noexcept;

	// Component accessor methods
	// --------------------------------------------------------------------------------------------

	/// Returns the component type index
	uint32_t typeIndex() const noexcept;

	/// Returns the ComponentMask for the component type associated with this accessor
	ComponentMask mask() const noexcept;

	/// Adds a component to the specified entity
	void add(uint32_t entity, const T& component) noexcept;

	/// Removes the component associated with the specified entity
	void remove(uint32_t entity) noexcept;

	/// Returns the pointer to the internal array of a given type of component
	T* arrayPtr() noexcept;
	const T* arrayPtr() const noexcept;

	/// Returns the number of components of this type
	uint32_t numComponents() const noexcept;

	/// Returns pointer to the component associated with the specified entity. Returns nullptr if
	/// component does not exist.
	T* get(uint32_t entity) noexcept;
	const T* get(uint32_t entity) const noexcept;

private:
	// Private members
	// --------------------------------------------------------------------------------------------

	SharedPtr<EntityComponentSystem> mEcsPtr;
	uint32_t mComponentType = ~0u;
};

// EcsComponentAccessor implementation: Constructors & destructors
// ------------------------------------------------------------------------------------------------

template<typename T>
EcsComponentAccessor<T>::EcsComponentAccessor(const SharedPtr<EntityComponentSystem>& ecsPtr) noexcept
{
	static_assert(std::is_pod<T>::value, "Type is not POD");
	mEcsPtr = ecsPtr;
	mComponentType = mEcsPtr->createComponentTypeRaw(sizeof(T));
}

// EcsComponentAccessor implementation: Component accessor methods
// ------------------------------------------------------------------------------------------------

template<typename T>
uint32_t EcsComponentAccessor<T>::typeIndex() const noexcept
{
	return mComponentType;
}

template<typename T>
ComponentMask EcsComponentAccessor<T>::mask() const noexcept
{
	return ComponentMask::fromType(mComponentType);
}

template<typename T>
void EcsComponentAccessor<T>::add(uint32_t entity, const T& component) noexcept
{
	mEcsPtr->addComponentRaw(entity, mComponentType, reinterpret_cast<const void*>(&component));
}

template<typename T>
void EcsComponentAccessor<T>::remove(uint32_t entity) noexcept
{
	mEcsPtr->removeComponent(entity, mComponentType);
}

template<typename T>
T* EcsComponentAccessor<T>::arrayPtr() noexcept
{
	return reinterpret_cast<T*>(mEcsPtr->componentArrayPtrRaw(mComponentType));
}

template<typename T>
const T* EcsComponentAccessor<T>::arrayPtr() const noexcept
{
	return reinterpret_cast<const T*>(mEcsPtr->componentArrayPtrRaw(mComponentType));
}

template<typename T>
uint32_t EcsComponentAccessor<T>::numComponents() const noexcept
{
	return mEcsPtr->numComponents(mComponentType);
}

template<typename T>
T* EcsComponentAccessor<T>::get(uint32_t entity) noexcept
{
	return reinterpret_cast<T*>(mEcsPtr->getComponentRaw(entity, mComponentType));
}

template<typename T>
const T* EcsComponentAccessor<T>::get(uint32_t entity) const noexcept
{
	return reinterpret_cast<const T*>(mEcsPtr->getComponentRaw(entity, mComponentType));
}

} // namespace phe
