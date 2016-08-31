#pragma once

#include <cstdint>

#include <sfz/math/Vector.hpp>

#include <sfz/gl/IncludeOpenGL.hpp>
#include <cuda_gl_interop.h>

namespace sfz {

// CUDAGLTexture
// ------------------------------------------------------------------------------------------------

class CUDAGLTexture final {
public:
	// Constructors & destructors
	// --------------------------------------------------------------------------------------------

	CUDAGLTexture(const CUDAGLTexture&) = delete;
	CUDAGLTexture& operator= (const CUDAGLTexture&) = delete;
	CUDAGLTexture(CUDAGLTexture&& other) = delete;
	CUDAGLTexture& operator= (CUDAGLTexture&& other) = delete;

	CUDAGLTexture() noexcept;
	~CUDAGLTexture() noexcept;

	// Methods
	// --------------------------------------------------------------------------------------------

	void setSize(vec2i resolution) noexcept;

	void cudaRegister() noexcept;
	void cudaUnregister() noexcept;
	cudaSurfaceObject_t cudaMap() noexcept;
	void cudaUnmap() noexcept;

	// Getters
	// --------------------------------------------------------------------------------------------
	
	inline uint32_t glTexture() const noexcept { return mGLTex; }
	inline vec2i resolution() const noexcept { return mRes; }

private:
	// Private members
	// --------------------------------------------------------------------------------------------

	uint32_t mGLTex = 0;
	vec2i mRes = vec2i(0);

	bool mCudaRegistered = false;
	bool mCudaMapped = false;

	cudaGraphicsResource_t mCudaResource;
	cudaSurfaceObject_t mCudaSurfaceObject;
};

} // namespace sfz
