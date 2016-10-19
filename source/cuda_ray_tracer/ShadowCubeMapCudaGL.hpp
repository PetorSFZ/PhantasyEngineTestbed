// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include <cstdint>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

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
	uint32_t shadowCubeMapGL() const noexcept { return mShadowCubeMap; }
	cudaSurfaceObject_t shadowCubeMapCuda() const noexcept { return mShadowSurface; }

private:
	// Private members
	// --------------------------------------------------------------------------------------------
	
	vec2u mRes = vec2u(0u);
	uint32_t mFbo = 0u;
	uint32_t mCubeDepthTexture = 0u;

	uint32_t mShadowCubeMap = 0u;
	cudaGraphicsResource_t mShadowResource = 0;
	cudaSurfaceObject_t mShadowSurface = 0;
};

} // namespace phe
