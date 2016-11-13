// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include <cstdint>

namespace phe {

using std::uint8_t;
using std::uint32_t;
using std::uint64_t;

// Constants
// ------------------------------------------------------------------------------------------------

constexpr uint32_t ECS_MAX_NUM_COMPONENT_TYPES = 128;
constexpr uint32_t ECS_NUM_BYTES_PER_MASK = ECS_MAX_NUM_COMPONENT_TYPES / 8u;
constexpr uint64_t ECS_ENTITY_EXISTENCE_MASK = 1u;

// EntityMask
// ------------------------------------------------------------------------------------------------

struct alignas(ECS_NUM_BYTES_PER_MASK) EntityMask final {
	// Members
	// --------------------------------------------------------------------------------------------

	uint8_t rawMask[ECS_NUM_BYTES_PER_MASK];

	// Constructors & destructors
	// --------------------------------------------------------------------------------------------

	EntityMask() noexcept = default;
	EntityMask(const EntityMask&) noexcept = default;
	EntityMask& operator= (const EntityMask&) noexcept = default;
	~EntityMask() noexcept = default;

	static EntityMask empty() noexcept;
	static EntityMask fromComponentType(uint32_t componentType) noexcept;

	// Operators
	// --------------------------------------------------------------------------------------------

	bool operator== (const EntityMask& other) const noexcept;
	bool operator!= (const EntityMask& other) const noexcept;

	EntityMask operator& (const EntityMask& other) const noexcept;
	EntityMask operator| (const EntityMask& other) const noexcept;

	// Methods
	// --------------------------------------------------------------------------------------------

	/// Checks whether this mask contains the specified component type or not
	bool hasComponentType(uint32_t componentType) const noexcept;

	/// Checks whether this mask has all the components in the specified parameter mask
	bool fulfills(const EntityMask& constraints) const noexcept;
};

static_assert(sizeof(EntityMask) == ECS_NUM_BYTES_PER_MASK, "EntityMask is padded");

// EntityComponentSystem
// ------------------------------------------------------------------------------------------------

class EntityComponentSystemImpl; // Pimpl

class EntityComponentSystem final {
public:
	
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

	// Entity methods
	// --------------------------------------------------------------------------------------------

	/// Creates a new entity with no associated components. Index is guaranteed to be smaller than
	/// the systems maximum number of entities. Indices used for removed entities will be used.
	uint32_t createEntity() noexcept;

	/// Deletes an entity. Will remove all associated components and free the index to be reused
	/// for future entities.
	void deleteEntity(uint32_t entity) noexcept;

	/// Returns the component mask for a given entity.
	const EntityMask& componentMask(uint32_t entity) const noexcept;

	// Component methods
	// --------------------------------------------------------------------------------------------


	// Raw (non-typesafe) component methods
	// --------------------------------------------------------------------------------------------

	/// Specifies a new type of component, returns its index to be used when accessing.
	uint32_t createComponentTypeRaw(uint32_t bytesPerComponent) noexcept;
	
	/// Adds a component of the specified type to the specified entity.
	void addComponentRaw(uint32_t entity, uint32_t componentType, const void* component) noexcept;

	/// Removes a component from an entity.
	void removeComponentRaw(uint32_t entity, uint32_t componentType) noexcept;

	/// Returns the pointer to the internal array of a given type of component.
	void* componentArrayRaw(uint32_t componentType) noexcept;
	const void* componentArrayRaw(uint32_t componentType) const noexcept;

	/// Returns the number of components of a specific type.
	uint32_t numComponents(uint32_t componentType) noexcept;

	/// Returns pointer to the component of specified type for a given entity. Returns nullptr if
	/// component does not exist.
	void* getComponentRaw(uint32_t entity, uint32_t componentType) noexcept;
	const void* getComponentRaw(uint32_t entity, uint32_t componentType) const noexcept;

private:
	// Private members
	// --------------------------------------------------------------------------------------------

	EntityComponentSystemImpl* mImpl = nullptr;
};

} // namespace phe
