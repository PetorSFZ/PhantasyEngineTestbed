// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include <sfz/memory/SmartPointers.hpp>

#include "phantasy_engine/level/EcsComponentAccessor.hpp"
#include "phantasy_engine/level/EntityComponentSystem.hpp"
#include "phantasy_engine/rendering/RenderComponent.hpp"

namespace phe {

using sfz::SharedPtr;

// EcsWrapper
// ------------------------------------------------------------------------------------------------

/// A wrapper around the entity component system for Phantasy Engine.
///
/// Provides typesafe easy access to built-in component types provided by Phantasy Engine itself.
/// See EntityComponentSystem for documentation.
class EcsWrapper final {
public:
	// Public members
	// --------------------------------------------------------------------------------------------

	SharedPtr<EntityComponentSystem> rawEcsPtr;
	EcsComponentAccessor<RenderComponent> renderComponents;

	// Constructors & destructors
	// --------------------------------------------------------------------------------------------
	
	EcsWrapper() noexcept = default;
	EcsWrapper(const EcsWrapper&) = delete;
	EcsWrapper& operator= (const EcsWrapper&) = delete;
	
	explicit EcsWrapper(uint32_t maxNumEntities) noexcept;
	EcsWrapper(EcsWrapper&& other) noexcept;
	EcsWrapper& operator= (EcsWrapper&& other) noexcept;
	~EcsWrapper() noexcept;
	
	// State methods
	// --------------------------------------------------------------------------------------------

	void destroy() noexcept;
	void swap(EcsWrapper& other) noexcept;

	// Entity method wrappers
	// --------------------------------------------------------------------------------------------

	uint32_t createEntity() noexcept { return rawEcsPtr->createEntity(); }
	void deleteEntity(uint32_t entity) noexcept { rawEcsPtr->deleteEntity(entity); }
	const ComponentMask& componentMask(uint32_t entity) const noexcept { return rawEcsPtr->componentMask(entity); }
	const ComponentMask* componentMaskArrayPtr() const noexcept { return rawEcsPtr->componentMaskArrayPtr(); }
	uint32_t maxNumEntities() const noexcept { return rawEcsPtr->maxNumEntities(); }
	uint32_t currentNumEntities() const noexcept { return rawEcsPtr->currentNumEntities(); }
	uint32_t entityIndexUpperBound() const noexcept { return rawEcsPtr->entityIndexUpperBound(); }
};

} // namepace phe
