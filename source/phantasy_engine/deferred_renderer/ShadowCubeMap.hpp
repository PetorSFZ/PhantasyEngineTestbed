// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include <cstdint>

#include <sfz/gl/Framebuffer.hpp>
#include <sfz/math/Vector.hpp>

namespace phe {

using sfz::gl::FBDepthFormat;
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

	ShadowCubeMap(vec2u res, FBDepthFormat depthFormat, bool pcf = true) noexcept;
	ShadowCubeMap(ShadowCubeMap&& other) noexcept;
	ShadowCubeMap& operator= (ShadowCubeMap&& other) noexcept;
	~ShadowCubeMap() noexcept;

	// Methods
	// --------------------------------------------------------------------------------------------

	void destroy() noexcept;
	void swap(ShadowCubeMap& other) noexcept;


private:
	// Private members
	// --------------------------------------------------------------------------------------------
	
	uint32_t mFBO = 0u;
	uint32_t mDepthTexture = 0;
};

} // namespace phe
