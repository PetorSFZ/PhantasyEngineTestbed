// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include <cstdint>

#include <sfz/memory/Allocator.hpp>

namespace phe {

using std::uint8_t;
using std::uint32_t;
using std::uint64_t;
using sfz::Allocator;

// ComponentMask
// ------------------------------------------------------------------------------------------------

constexpr uint32_t ECS_MAX_NUM_COMPONENT_TYPES = 64;

/// A mask specifying a number of components.
struct ComponentMask final {
	// Members
	// --------------------------------------------------------------------------------------------

	uint64_t rawMask;

	// Constructor methods
	// --------------------------------------------------------------------------------------------

	static ComponentMask fromRawValue(uint64_t bits) noexcept { return { bits }; }
	static ComponentMask empty() noexcept { return ComponentMask::fromRawValue(0); }
	static ComponentMask fromType(uint32_t componentType) noexcept
	{
		return ComponentMask::fromRawValue(uint64_t(1) << uint64_t(componentType));
	}
	static ComponentMask existenceMask() noexcept { return ComponentMask::fromRawValue(1); }

	// Operators
	// ------------------------------------------------------------------------------------------------

	bool operator== (const ComponentMask& o) const noexcept { return rawMask == o.rawMask; }
	bool operator!= (const ComponentMask& o) const noexcept { return rawMask != o.rawMask; }
	ComponentMask operator& (const ComponentMask& o) const noexcept { return { rawMask & o.rawMask }; }
	ComponentMask operator| (const ComponentMask& o) const noexcept { return { rawMask | o.rawMask }; }
	ComponentMask operator~ () const noexcept { return { ~rawMask }; }

	// Methods
	// --------------------------------------------------------------------------------------------

	/// Checks whether this mask contains the specified component type or not
	bool hasComponentType(uint32_t componentType) const noexcept
	{
		return this->fulfills(ComponentMask::fromType(componentType));
	}

	/// Checks whether this mask has all the components in the specified parameter mask
	bool fulfills(const ComponentMask& constraints) const noexcept
	{
		return (this->rawMask & constraints.rawMask) == constraints.rawMask;
	}
};

// EntityComponentSystem
// ------------------------------------------------------------------------------------------------

class EntityComponentSystemImpl; // Pimpl

/// An entity component system
///
/// Specifically designed to hide as much of the implementation as possible in the translation
/// unit in order to improve compile times. To accomplish this each component type is only defined
/// by how many bytes each component uses. In practice this means that the system is not even
/// slightly typesafe, and components can only be PODs.
///
/// An entity is simply a number (uint32_t). Each entity have a mask that specifies which
/// components it has. In addition, each mask has a bit specifying that the entity exists (0th bit).
///
/// A component type is a type of component stored in the system, each type has an associated bit
/// in the EntityMask. In addition, each component type also has an index (referred to as
/// 'componentType' in parameters). This index specifies which bit in the mask refers to this type.
/// In addition, this index is used when retrieving components from the system.
///
/// In general, the system is designed to use large components. All existing components are
/// packed together, not necessarily in the order of the entities' numbers. This means that it is
/// very cheap and efficient to iterate over all components of a specific type using
/// 'componentArrayPtr()' and 'numComponents()'.
///
/// When retrieving a component for a specific entity ('getComponent()') it is necessary to check
/// in a lookup table where this entity stores its component. This means that it is very
/// inefficient to iterate over multiple types of components at the same time. Accessing a specific
/// entity's component is not an operation that should be performed often if it can be avoided.
///
/// If it truly is necessary to iterate over all entities (instead of over all components of a 
/// specific type), the 'entityIndexUpperBound()' method should be used. This method returns
/// the highest possible value an entity can currently have + 1. It is not the same as
/// 'currentNumEntities()' as there may be holes from deleted entities.
class EntityComponentSystem final {
public:
	
	// Constructors & destructors
	// --------------------------------------------------------------------------------------------

	EntityComponentSystem() noexcept = default;
	EntityComponentSystem(const EntityComponentSystem&) = delete;
	EntityComponentSystem& operator= (const EntityComponentSystem&) = delete;
	EntityComponentSystem(EntityComponentSystem&&) = delete;
	EntityComponentSystem& operator= (EntityComponentSystem&&) = delete;

	/// Creates an ECS using the default constructor and then calls create() on it.
	explicit EntityComponentSystem(uint32_t maxNumEntities, Allocator* allocator) noexcept;
	~EntityComponentSystem() noexcept;

	// State methods
	// --------------------------------------------------------------------------------------------

	/// Creates a new EntityComponentSystem. Will first destroy() the existing one.
	void create(uint32_t maxNumEntities, Allocator* allocator) noexcept;

	/// Destroys this EntityComponentSystem, called automatically in the destructor.
	void destroy() noexcept;

	/// Checks whether this EntityComponentSystem is valid or not. If it is not valid then all
	/// methods except for destroy() and swap() is undefined and should under no circumstances be
	/// called.
	bool isValid() const noexcept { return mImpl != nullptr; }

	/// Returns the allocator associated with this EntityComponentSystem
	Allocator* allocator() const noexcept;

	// Entity methods
	// --------------------------------------------------------------------------------------------

	/// Creates a new entity with no associated components. Index is guaranteed to be smaller than
	/// the systems maximum number of entities. Indices used for removed entities will be used.
	/// Complexity: O(1) operation, should be very fast.
	uint32_t createEntity() noexcept;

	/// Deletes an entity. Will remove all associated components and free the index to be reused
	/// for future entities.
	/// Complexity: O(NUM_COMPONENT_TYPES * NUM_COMPONENTS) operation, essentialy removeComponent()
	///             is called for each component the entity has.
	void deleteEntity(uint32_t entity) noexcept;

	/// Returns the component mask for a given entity.
	/// Complexity: O(1) operation, however componentMaskArrayPtr() should be used if more than
	///             one ComponentMask is to be accessed.
	const ComponentMask& componentMask(uint32_t entity) const noexcept;

	/// Returns pointer to the intenral array of component masks. The ComponentMask for entity
	/// 'i' is accessed by returnedPointer[i].
	/// Complexity: O(1) operation
	const ComponentMask* componentMaskArrayPtr() const noexcept;

	/// Returns the maximum number of entites allowed by this system
	/// Complexity: O(1) operation
	uint32_t maxNumEntities() const noexcept;

	/// Returns the current number of entities in this system
	/// Complexity: O(1) operation
	uint32_t currentNumEntities() const noexcept;

	/// Gives an upper bound for the index values of the current entities. This value should be
	/// used when iterating over all entities. Guaranteed to be at least 1 bigger than the
	/// currently highest index, but not guaranteed to be lower than currentNumEntities().
	/// Complexity: O(1) operation
	uint32_t entityIndexUpperBound() const noexcept;

	// Raw (non-typesafe) component methods
	// --------------------------------------------------------------------------------------------

	/// Specifies a new type of component, returns its index to be used when accessing.
	/// Complexity: O(NUM_INITIAL_COMPONENTS_CAPACITY) memory allocation operation
	/// TODO: Parameter for initial number of components to allocate space for
	uint32_t createComponentTypeRaw(uint32_t bytesPerComponent) noexcept;
	
	/// Returns the current number of component types in this system. The maximum number is defined
	/// by the MAX_NUM_COMPONENT_TYPES constant.
	/// Complexity: O(1) operation
	uint32_t currentNumComponentTypes() const noexcept;

	/// Adds a component of the specified type to the specified entity.
	/// Complexity: O(1) operation
	/// TODO: Potentially amortized O(1) operation if memory allocation scheme is changed
	void addComponentRaw(uint32_t entity, uint32_t componentType, const void* component) noexcept;

	/// Removes a component from an entity.
	/// Complexity: O(NUM_COMPONENTS) operation, moves all components after this one back in the
	///             list.
	/// TODO: Could be changed to an O(1) operation by swapping component to be removed and
	/// component at last position in list. However, this has implications for iteration and list
	/// order. I.e. it will be harder to keep list sorted.
	void removeComponent(uint32_t entity, uint32_t componentType) noexcept;

	/// Returns the pointer to the internal array of a given type of component.
	/// Complexity: O(1) operation
	void* componentArrayPtrRaw(uint32_t componentType) noexcept;
	const void* componentArrayPtrRaw(uint32_t componentType) const noexcept;

	/// Returns the number of components of a specific type.
	/// Complexity: O(1) operation
	uint32_t numComponents(uint32_t componentType) noexcept;

	/// Returns pointer to the component of specified type for a given entity. Returns nullptr if
	/// component does not exist.
	/// Complexity: O(1) operation, however componentArrayPtrRaw() should be used instead if more
	///             than one component is to be accessed.
	void* getComponentRaw(uint32_t entity, uint32_t componentType) noexcept;
	const void* getComponentRaw(uint32_t entity, uint32_t componentType) const noexcept;

private:
	// Private members
	// --------------------------------------------------------------------------------------------

	EntityComponentSystemImpl* mImpl = nullptr;
};

} // namespace phe
