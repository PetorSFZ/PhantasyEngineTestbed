// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include <algorithm>
#include <cstdint>

#include "cuda_runtime.h"

#include <sfz/Assert.hpp>

#include "CudaHelpers.hpp"

namespace phe {

using std::uint32_t;

// CudaBuffer
// ------------------------------------------------------------------------------------------------

template<typename T>
class CudaBuffer final {
public:
	// Constructors & destructors
	// --------------------------------------------------------------------------------------------

	CudaBuffer() noexcept = default;
	CudaBuffer(const CudaBuffer&) = delete;
	CudaBuffer& operator= (const CudaBuffer&) = delete;

	CudaBuffer(const T* dataPtr, uint32_t numElements) noexcept;
	CudaBuffer(CudaBuffer&& other) noexcept;
	CudaBuffer& operator= (CudaBuffer&& other) noexcept;
	~CudaBuffer() noexcept;

	// Methods
	// --------------------------------------------------------------------------------------------

	void create(uint32_t capacity) noexcept;
	void upload(const T* dataPtr, uint32_t numElements) noexcept;
	void destroy() noexcept;
	void swap(CudaBuffer& other) noexcept;

	// Getters
	// --------------------------------------------------------------------------------------------

	inline T* cudaPtr() noexcept { return mCudaPtr; }
	inline const T* cudaPtr() const noexcept { return mCudaPtr; }
	inline uint32_t capacity() const noexcept { return mCapacity; }
	inline uint32_t size() const noexcept { return mCapacity; }

private:
	// Private members
	// --------------------------------------------------------------------------------------------

	T* mCudaPtr = nullptr;
	uint32_t mCapacity = 0u;
	uint32_t mSize = 0u;
};

// CudaBuffer implementation: Constructors & destructors
// ------------------------------------------------------------------------------------------------

template<typename T>
CudaBuffer<T>::CudaBuffer(const T* dataPtr, uint32_t numElements) noexcept
{
	this->create(numElements);
	this->upload(dataPtr, numElements);
}

template<typename T>
CudaBuffer<T>::CudaBuffer(CudaBuffer&& other) noexcept
{
	this->swap(other);
}

template<typename T>
CudaBuffer<T>& CudaBuffer<T>::operator= (CudaBuffer&& other) noexcept
{
	this->swap(other);
	return *this;
}

template<typename T>
CudaBuffer<T>::~CudaBuffer() noexcept
{
	this->destroy();
}

// CudaBuffer implementation: Methods
// ------------------------------------------------------------------------------------------------

template<typename T>
void CudaBuffer<T>::create(uint32_t capacity) noexcept
{
	if (mCapacity != 0u) this->destroy();
	size_t numBytes = capacity * sizeof(T);
	CHECK_CUDA_ERROR(cudaMalloc(&mCudaPtr, numBytes));
	mCapacity = capacity;
}

template<typename T>
void CudaBuffer<T>::upload(const T* dataPtr, uint32_t numElements) noexcept
{
	sfz_assert_debug(numElements <= mCapacity);
	size_t numBytes = numElements * sizeof(T);
	CHECK_CUDA_ERROR(cudaMemcpy(mCudaPtr, dataPtr, numBytes, cudaMemcpyHostToDevice));
}

template<typename T>
void CudaBuffer<T>::destroy() noexcept
{
	CHECK_CUDA_ERROR(cudaFree(mCudaPtr));
	mCudaPtr = nullptr;
	mCapacity = 0u;
	mSize = 0u;
}

template<typename T>
void CudaBuffer<T>::swap(CudaBuffer& other) noexcept
{
	std::swap(this->mCudaPtr, other.mCudaPtr);
	std::swap(this->mCapacity, other.mCapacity);
	std::swap(this->mSize, other.mSize);
}

} // namespace phe
