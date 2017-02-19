// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include <sfz/cuda/Buffer.hpp>

namespace phe {

// CudaTextureBuffer
// ------------------------------------------------------------------------------------------------

template<typename T>
class CudaTextureBuffer final {
public:
	// Constructors & destructors
	// --------------------------------------------------------------------------------------------

	CudaTextureBuffer() noexcept = default;
	CudaTextureBuffer(const CudaTextureBuffer&) = delete;
	CudaTextureBuffer& operator= (const CudaTextureBuffer&) = delete;

	CudaTextureBuffer(const T* dataPtr, uint32_t numElements) noexcept;
	CudaTextureBuffer(CudaTextureBuffer&& other) noexcept;
	CudaTextureBuffer& operator= (CudaTextureBuffer&& other) noexcept;
	~CudaTextureBuffer() noexcept;

	// Getters
	// --------------------------------------------------------------------------------------------
	
	inline T* cudaPtr() noexcept { return mBuffer.cudaPtr(); }
	inline const T* cudaPtr() const noexcept { return mBuffer.cudaPtr(); }
	inline cudaTextureObject_t cudaTexture() const noexcept { return mTexture; }

private:
	// Private members
	// --------------------------------------------------------------------------------------------

	sfz::cuda::Buffer<T> mBuffer;
	cudaTextureObject_t mTexture = 0;
};

// CudaTextureBuffer implementation: Constructors & destructors
// ------------------------------------------------------------------------------------------------

template<typename T>
CudaTextureBuffer<T>::CudaTextureBuffer(const T* dataPtr, uint32_t numElements) noexcept
{
	static_assert((sizeof(T) % 16) == 0, "Type is not 16-byte aligned");

	mBuffer.create(numElements);
	mBuffer.upload(dataPtr, 0, numElements);

	cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(cudaResourceDesc));
	resDesc.resType = cudaResourceTypeLinear;
	resDesc.res.linear.devPtr = mBuffer.data();
	resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
	resDesc.res.linear.desc.x = 32; // bits per channel
	resDesc.res.linear.desc.y = 32; // bits per channel
	resDesc.res.linear.desc.z = 32; // bits per channel
	resDesc.res.linear.desc.w = 32; // bits per channel
	resDesc.res.linear.sizeInBytes = mBuffer.capacity() * sizeof(T);

	cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(cudaTextureDesc));
	texDesc.readMode = cudaReadModeElementType;

	CHECK_CUDA_ERROR(cudaCreateTextureObject(&mTexture, &resDesc, &texDesc, NULL));
}

template<typename T>
CudaTextureBuffer<T>::CudaTextureBuffer(CudaTextureBuffer&& other) noexcept
{
	std::swap(this->mBuffer, other.mBuffer);
	std::swap(this->mTexture, other.mTexture);
}

template<typename T>
CudaTextureBuffer<T>& CudaTextureBuffer<T>::operator= (CudaTextureBuffer&& other) noexcept
{
	std::swap(this->mBuffer, other.mBuffer);
	std::swap(this->mTexture, other.mTexture);
	return *this;
}

template<typename T>
CudaTextureBuffer<T>::~CudaTextureBuffer() noexcept
{
	if (mTexture != 0) {
		CHECK_CUDA_ERROR(cudaDestroyTextureObject(mTexture));
		mTexture = 0;
	}
	mBuffer.destroy();
}

} // namespace phe
