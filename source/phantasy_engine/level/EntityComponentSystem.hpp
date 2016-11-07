// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include <cstdint>

namespace phe {

using std::uint32_t;

// EntityComponentSystem
// ------------------------------------------------------------------------------------------------

class EntityComponentSystemImpl; // Pimpl

class EntityComponentSystem final {
public:
	// Constants
	// --------------------------------------------------------------------------------------------

	constexpr static uint32_t MAX_NUM_COMPONENT_TYPES = 128u;

	// Constructors & destructors
	// --------------------------------------------------------------------------------------------

	EntityComponentSystem() noexcept = default;
	EntityComponentSystem(const EntityComponentSystem&) = delete;
	EntityComponentSystem& operator= (const EntityComponentSystem&) = delete;

	EntityComponentSystem(uint32_t maxNumEntities) noexcept;
	EntityComponentSystem(EntityComponentSystem&& other) noexcept;
	EntityComponentSystem& operator= (EntityComponentSystem&& other) noexcept;
	~EntityComponentSystem() noexcept;

	// State methods & getters
	// --------------------------------------------------------------------------------------------

	/// Destroys this EntityComponentSystem, called automatically in the destructor.
	void destroy() noexcept;

	/// Swaps the contents of this EntityComponentSystem with another, same as using a move
	/// constructor.
	void swap(EntityComponentSystem& other) noexcept;

	/// Checks whether this EntityComponentSystem is valid or not. If it is not valid then all
	/// methods except for destroy() and swap() is undefined and should under no circumstances be
	/// called.
	bool isValid() const noexcept { return mImpl != nullptr; }

	/// Returns the maximum number of entites allowed by this system
	uint32_t maxNumEntities() const noexcept;

	/// Returns the current number of entities in this system
	uint32_t currentNumEntities() const noexcept;

	/// Returns the current number of component types in this system. The maximum number is defined
	/// by the MAX_NUM_COMPONENT_TYPES constant.
	uint32_t currentNumComponentTypes() const noexcept;

	// Entity creation/deletion
	// --------------------------------------------------------------------------------------------

	/// Creates a new entity with no associated components. Index is guaranteed to be smaller than
	/// the systems maximum number of entities. Indices used for removed entities will be used
	uint32_t createEntity() noexcept;

	/// Deletes an entity. Will remove all associated components and free the index to be reused
	/// for future entities.
	void deleteEntity(uint32_t entity) noexcept;

	// Raw (non-typesafe) methods
	// --------------------------------------------------------------------------------------------

	/// Specifies a new type of component, returns its index to be used when accessing
	uint32_t createComponentTypeRaw(uint32_t bytesPerComponent) noexcept;
	
	/// Returns pointer to the component of specified type for a given entity. Returns nullptr if
	/// it does not exist.
	void* getComponentRaw(uint32_t entity, uint32_t componentType) noexcept;
	const void* getComponentRaw(uint32_t entity, uint32_t componentType) const noexcept;

private:
	// Private members
	// --------------------------------------------------------------------------------------------

	EntityComponentSystemImpl* mImpl = nullptr;
};

} // namespace phe
