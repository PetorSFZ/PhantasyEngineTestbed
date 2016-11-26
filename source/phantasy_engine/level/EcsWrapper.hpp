// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include <cstdint>

#include <sfz/math/Matrix.hpp>
#include <sfz/math/Vector.hpp>

#include "phantasy_engine/level/EntityComponentSystem.hpp"

namespace phe {

using sfz::vec3;
using sfz::mat4;

// Built-in component types
// ------------------------------------------------------------------------------------------------

struct GraphicsComponent final {
	uint32_t meshIndex;
	mat4 transform;
	vec3 velocity; // TODO: Should perhaps not be here?
};

// EcsWrapper
// ------------------------------------------------------------------------------------------------

/// Support class for EntityComponentSystem responsible for providing typesafe access to
/// components with types known at compile time.
template<typename T>
class EcsWrapper final {
public:
	// Constructors & destructors
	// --------------------------------------------------------------------------------------------

	EcsWrapper() noexcept = default;
	EcsWrapper(const EcsWrapper&) noexcept = default;
	EcsWrapper& operator= (const EcsWrapper&) noexcept = default;
	~EcsWrapper() noexcept = default;
	
	/// Creates a new component type of the specified class in the parameter ecs.
	EcsWrapper(EntityComponentSystem& ecs) noexcept
	:
		mEcsPtr(&ecs)
	{
		
	}

private:
	// Private members
	// --------------------------------------------------------------------------------------------

	EntityComponentSystem* mEcsPtr = nullptr;
	uint32_t mComponentType = ~0u;
};

} // namespace phe
