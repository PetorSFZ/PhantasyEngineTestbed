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

	void destroy() noexcept;
	void swap(EntityComponentSystem& other) noexcept;
	bool isValid() const noexcept { return mImpl != nullptr; }

	uint32_t maxNumEntities() const noexcept;
	uint32_t maxNumComponents() const noexcept;
	uint32_t currentNumComponents() const noexcept;

	// Raw (non-typesafe) methods
	// --------------------------------------------------------------------------------------------

	void* getComponentRaw(uint32_t entity, uint32_t componentType,
	                      uint32_t bytesPerComponent) noexcept;
	const void* getComponentRaw(uint32_t entity, uint32_t componentType,
	                            uint32_t bytesPerComponent) const noexcept;

private:
	// Private members
	// --------------------------------------------------------------------------------------------

	EntityComponentSystemImpl* mImpl = nullptr;
};

inline size_t &component_counter()
{
	static size_t counter = 0;
	return counter;
}

template<typename C>
size_t component_index()
{
	static size_t index = inc_component_counter();
	return index;
}

inline size_t inc_component_counter()
{
	size_t index = component_counter()++;
	return index;
}

} // namespace phe
