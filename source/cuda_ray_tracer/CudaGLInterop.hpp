// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <sfz/gl/Framebuffer.hpp>
#include <sfz/math/Vector.hpp>

namespace phe {

using sfz::gl::Framebuffer;
using sfz::vec2i;

// CudaGLGBuffer
// ------------------------------------------------------------------------------------------------

class CudaGLGBuffer final {
public:
	// Constructors & destructors
	// --------------------------------------------------------------------------------------------

	CudaGLGBuffer() noexcept = default;
	CudaGLGBuffer(const CudaGLGBuffer&) = delete;
	CudaGLGBuffer& operator= (const CudaGLGBuffer&) = delete;
	
	CudaGLGBuffer(vec2i resolution) noexcept;
	CudaGLGBuffer(CudaGLGBuffer&& other) noexcept;
	CudaGLGBuffer& operator= (CudaGLGBuffer&& other) noexcept;
	~CudaGLGBuffer() noexcept;

	// Methods
	// --------------------------------------------------------------------------------------------

	void bindViewportClearColorDepth() noexcept;

	// Getters
	// --------------------------------------------------------------------------------------------

	inline vec2i resolution() const noexcept { return mFramebuffer.dimensions(); }

	uint32_t fbo() const noexcept { return mFramebuffer.fbo(); }

	uint32_t depthTextureGL() const noexcept;
	uint32_t positionTextureGL() const noexcept;
	uint32_t normalTextureGL() const noexcept;
	uint32_t materialIdTextureGL() const noexcept;

	cudaSurfaceObject_t positionSurfaceCuda() const noexcept;
	cudaSurfaceObject_t normalSurfaceCuda() const noexcept;
	cudaSurfaceObject_t materialIdSurfaceCuda() const noexcept;

private:
	// Private members
	// --------------------------------------------------------------------------------------------

	Framebuffer mFramebuffer;

	cudaGraphicsResource_t mPosResource = 0;
	cudaSurfaceObject_t mPosSurface = 0;

	cudaGraphicsResource_t mNormalResource = 0;
	cudaSurfaceObject_t mNormalSurface = 0;
	
	cudaGraphicsResource_t mMaterialIdResource = 0;
	cudaSurfaceObject_t mMaterialIdSurface = 0;
};

// CudaGLTexture
// ------------------------------------------------------------------------------------------------

// OpenGL texture that is read/writeable in Cuda.
// Always float rgba
class CudaGLTexture final {
public:
	// Constructors & destructors
	// --------------------------------------------------------------------------------------------

	CudaGLTexture() noexcept = default;
	CudaGLTexture(const CudaGLTexture&) = delete;
	CudaGLTexture& operator= (const CudaGLTexture&) = delete;
	
	CudaGLTexture(vec2i resolution) noexcept;
	CudaGLTexture(CudaGLTexture&& other) noexcept;
	CudaGLTexture& operator= (CudaGLTexture&& other) noexcept;
	~CudaGLTexture() noexcept;

	// Getters
	// --------------------------------------------------------------------------------------------

	inline vec2i resolution() const noexcept { return mRes; }
	inline uint32_t glTexture() const noexcept { return mGLTex; }
	inline cudaSurfaceObject_t cudaSurface() const noexcept { return mCudaSurface; }

private:
	// Private members
	// --------------------------------------------------------------------------------------------

	vec2i mRes = vec2i(0);
	uint32_t mGLTex = 0;
	cudaGraphicsResource_t mCudaResource = 0;
	cudaSurfaceObject_t mCudaSurface = 0;
};

} // namespace phe
