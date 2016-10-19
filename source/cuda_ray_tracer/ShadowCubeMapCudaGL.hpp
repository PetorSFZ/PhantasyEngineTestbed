// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include <cstdint>

#include <sfz/math/Vector.hpp>

namespace phe {

using sfz::vec2u;
using std::uint32_t;

// ShadowCubeMapCudaGL
// ------------------------------------------------------------------------------------------------
	
class ShadowCubeMapCudaGL final {
public:
	// Constructors & destructors
	// --------------------------------------------------------------------------------------------

	ShadowCubeMapCudaGL() noexcept = default;
	ShadowCubeMapCudaGL(const ShadowCubeMapCudaGL&) = delete;
	ShadowCubeMapCudaGL& operator= (const ShadowCubeMapCudaGL&) = delete;

	ShadowCubeMapCudaGL(uint32_t res) noexcept;
	ShadowCubeMapCudaGL(ShadowCubeMapCudaGL&& other) noexcept;
	ShadowCubeMapCudaGL& operator= (ShadowCubeMapCudaGL&& other) noexcept;
	~ShadowCubeMapCudaGL() noexcept;

	// Methods
	// --------------------------------------------------------------------------------------------

	void destroy() noexcept;
	void swap(ShadowCubeMapCudaGL& other) noexcept;

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
	uint32_t mCubeDepthTexture = 0u;
	uint32_t mShadowCubeMap = 0u;
};

} // namespace phe
