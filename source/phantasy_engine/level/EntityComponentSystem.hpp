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
constexpr uint64_t ECS_EXISTENCE_COMPONENT_TYPE = 0u;

// ComponentMask
// ------------------------------------------------------------------------------------------------

struct alignas(ECS_NUM_BYTES_PER_MASK) ComponentMask final {
	// Members
	// --------------------------------------------------------------------------------------------

	uint8_t rawMask[ECS_NUM_BYTES_PER_MASK];

	// Constructors & destructors
	// --------------------------------------------------------------------------------------------

	ComponentMask() noexcept = default;
	ComponentMask(const ComponentMask&) noexcept = default;
	ComponentMask& operator= (const ComponentMask&) noexcept = default;
	~ComponentMask() noexcept = default;

	static ComponentMask empty() noexcept;
	static ComponentMask fromType(uint32_t componentType) noexcept;
	static ComponentMask fromRawValue(uint64_t highBits, uint64_t lowBits) noexcept;

	// Operators
	// --------------------------------------------------------------------------------------------

	bool operator== (const ComponentMask& other) const noexcept;
	bool operator!= (const ComponentMask& other) const noexcept;

	ComponentMask operator& (const ComponentMask& other) const noexcept;
	ComponentMask operator| (const ComponentMask& other) const noexcept;
	ComponentMask operator~ () const noexcept;

	// Methods
	// --------------------------------------------------------------------------------------------

	/// Checks whether this mask contains the specified component type or not
	bool hasComponentType(uint32_t componentType) const noexcept;

	/// Checks whether this mask has all the components in the specified parameter mask
	bool fulfills(const ComponentMask& constraints) const noexcept;
};

static_assert(sizeof(ComponentMask) == ECS_NUM_BYTES_PER_MASK, "ComponentMask is padded");

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
	const ComponentMask& componentMask(uint32_t entity) const noexcept;

	// Raw (non-typesafe) component methods
	// --------------------------------------------------------------------------------------------

	/// Specifies a new type of component, returns its index to be used when accessing.
	uint32_t createComponentTypeRaw(uint32_t bytesPerComponent) noexcept;
	
	/// Adds a component of the specified type to the specified entity.
	void addComponentRaw(uint32_t entity, uint32_t componentType, const void* component) noexcept;

	/// Removes a component from an entity.
	void removeComponent(uint32_t entity, uint32_t componentType) noexcept;

	/// Returns the pointer to the internal array of a given type of component.
	void* componentArrayPtrRaw(uint32_t componentType) noexcept;
	const void* componentArrayPtrRaw(uint32_t componentType) const noexcept;

	/// Returns the number of components of a specific type.
	uint32_t numComponents(uint32_t componentType) noexcept;

	/// Returns pointer to the component of specified type for a given entity. Returns nullptr if
	/// component does not exist.
	void* getComponentRaw(uint32_t entity, uint32_t componentType) noexcept;
	const void* getComponentRaw(uint32_t entity, uint32_t componentType) const noexcept;

	// Component helper methods (not typesafe either)
	// --------------------------------------------------------------------------------------------

	template<typename T>
	uint32_t createComponentType() noexcept
	{
		static_assert(std::is_pod<T>::value, "Type is not POD");
		return this->createComponentTypeRaw(sizeof(T));
	}

	template<typename T>
	void addComponent(uint32_t entity, uint32_t componentType, const T& component) noexcept
	{
		static_assert(std::is_pod<T>::value, "Type is not POD");
		this->addComponentRaw(entity, componentType, reinterpret_cast<const void*>(&component));
	}

	template<typename T>
	T* componentArrayPtr(uint32_t componentType) noexcept
	{
		static_assert(std::is_pod<T>::value, "Type is not POD");
		return reinterpret_cast<T*>(this->componentArrayPtrRaw(componentType));
	}

	template<typename T>
	const T* componentArrayPtr(uint32_t componentType) const noexcept
	{
		static_assert(std::is_pod<T>::value, "Type is not POD");
		return reinterpret_cast<const T*>(this->componentArrayPtrRaw(componentType));
	}

	template<typename T>
	T* getComponent(uint32_t entity, uint32_t componentType) noexcept
	{
		static_assert(std::is_pod<T>::value, "Type is not POD");
		return reinterpret_cast<T*>(this->getComponentRaw(entity, componentType));
	}

	template<typename T>
	const T* getComponent(uint32_t entity, uint32_t componentType) const noexcept
	{
		static_assert(std::is_pod<T>::value, "Type is not POD");
		return reinterpret_cast<const T*>(this->getComponentRaw(entity, componentType));
	}

private:
	// Private members
	// --------------------------------------------------------------------------------------------

	EntityComponentSystemImpl* mImpl = nullptr;
};

} // namespace phe
