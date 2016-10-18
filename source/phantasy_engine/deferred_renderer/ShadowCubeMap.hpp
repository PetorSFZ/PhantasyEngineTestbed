// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include <cstdint>

#include <sfz/math/Vector.hpp>

namespace phe {

using sfz::vec2u;
using std::uint32_t;

// ShadowCubeMap
// ------------------------------------------------------------------------------------------------
	
class ShadowCubeMap final {
public:
	// Constructors & destructors
	// --------------------------------------------------------------------------------------------

	ShadowCubeMap() noexcept = default;
	ShadowCubeMap(const ShadowCubeMap&) = delete;
	ShadowCubeMap& operator= (const ShadowCubeMap&) = delete;

	ShadowCubeMap(uint32_t res) noexcept;
	ShadowCubeMap(ShadowCubeMap&& other) noexcept;
	ShadowCubeMap& operator= (ShadowCubeMap&& other) noexcept;
	~ShadowCubeMap() noexcept;

	// Methods
	// --------------------------------------------------------------------------------------------

	void destroy() noexcept;
	void swap(ShadowCubeMap& other) noexcept;

	// Getters
	// --------------------------------------------------------------------------------------------

	vec2u resolution() const noexcept { return mRes; }
	uint32_t fbo() const noexcept { return mFbo; }
	uint32_t shadowCubeMap() const noexcept { return mShadowCubeMap; }

private:
	// Private members
	// --------------------------------------------------------------------------------------------
	
	vec2u mRes = vec2u(0u);
	uint32_t mFbo = 0u;
	uint32_t mShadowCubeMap = 0u;
};

} // namespace phe
