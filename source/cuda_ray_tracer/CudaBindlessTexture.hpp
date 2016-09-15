// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include <phantasy_engine/resources/RawImage.hpp>

#include "cuda_runtime.h"

namespace phe {

// CudaBindlessTexture
// ------------------------------------------------------------------------------------------------

class CudaBindlessTexture final {
public:
	// Constructors & destructors
	// --------------------------------------------------------------------------------------------

	CudaBindlessTexture() noexcept = default;
	CudaBindlessTexture(const CudaBindlessTexture&) = delete;
	CudaBindlessTexture& operator= (const CudaBindlessTexture&) = delete;

	CudaBindlessTexture(CudaBindlessTexture&& other) noexcept;
	CudaBindlessTexture& operator= (CudaBindlessTexture&& other) noexcept;
	~CudaBindlessTexture() noexcept;

	// Methods
	// --------------------------------------------------------------------------------------------

	void load(const RawImage& image) noexcept;

	void destroy() noexcept;

	void swap(CudaBindlessTexture& other) noexcept;

	// Getters
	// --------------------------------------------------------------------------------------------

	inline cudaTextureObject_t textureObject() const noexcept { return mTextureObject; }

private:
	// Private members
	// --------------------------------------------------------------------------------------------

	cudaArray* mCudaArray = nullptr; // CUDA memory for texture
	cudaTextureObject_t mTextureObject = 0;
};

} // namespace phe
