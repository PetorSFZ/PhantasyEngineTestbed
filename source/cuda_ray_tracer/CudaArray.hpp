// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include <algorithm>
#include <cstdint>

#include "cuda_runtime.h"

#include <sfz/Assert.hpp>

#include "CudaHelpers.hpp"

namespace phe {

using std::uint32_t;

// DeviceArray
// ------------------------------------------------------------------------------------------------

template<typename T> class HostArray; // Forward declare HostArray

template<typename T>
struct DeviceArray final {
	// Members
	// --------------------------------------------------------------------------------------------
	
	T* const ptr;
	const uint32_t capacity;
	uint32_t size;

	// Constructors & destructors
	// --------------------------------------------------------------------------------------------

	__host__ __device__ DeviceArray(const DeviceArray&) noexcept = default;
	__host__ __device__ DeviceArray& operator= (const DeviceArray&) noexcept = default;
	__host__ __device__ ~DeviceArray() noexcept = default;

	// Default empty DeviceArray
	__host__ __device__ DeviceArray() noexcept;

	// Implicit conversion from HostArray
	__host__ DeviceArray(const HostArray<T>& hostArray) noexcept;
};

// HostArray
// ------------------------------------------------------------------------------------------------

template<typename T>
class HostArray final {
public:
	// Constructors & destructors
	// --------------------------------------------------------------------------------------------

	HostArray() noexcept = default;
	HostArray(const HostArray&) = delete;
	HostArray& operator= (const HostArray&) = delete;

	HostArray(T* dataPtr, uint32_t numElements) noexcept;
	HostArray(HostArray&& other) noexcept;
	HostArray& operator= (HostArray&& other) noexcept;
	~HostArray() noexcept;

	// Methods
	// --------------------------------------------------------------------------------------------

	void create(uint32_t capacity) noexcept;
	void upload(T* dataPtr, uint32_t numElements) noexcept;
	void destroy() noexcept;
	void swap(HostArray& other) noexcept;

	// Getters
	// --------------------------------------------------------------------------------------------

	inline T* cudaPtr() noexcept { return mCudaPtr; }
	inline const T* cudaPtr() const noexcept { return mCudaPtr; }
	inline uint32_t capacity() const noexcept { return mCapacity; }
	inline uint32_t size() const noexcept { return mCapacity; }

private:
	// Private members
	// --------------------------------------------------------------------------------------------

	friend struct DeviceArray<T>; // DeviceArray has direct access to members

	T* mCudaPtr = nullptr;
	uint32_t mCapacity = 0u;
	uint32_t mSize = 0u;
};

// DeviceArray implementation:
// ------------------------------------------------------------------------------------------------

template<typename T>
__host__ __device__ DeviceArray<T>::DeviceArray() noexcept
:
	ptr(nullptr),
	capacity(0u),
	size(0u)
{ }

template<typename T>
__host__ DeviceArray<T>::DeviceArray(const HostArray<T>& hostArray) noexcept
:
	ptr(hostArray.mCudaPtr),
	capacity(hostArray.mCapacity),
	size(hostArray.mSize)
{ }

// HostArray implementation: Constructors & destructors
// ------------------------------------------------------------------------------------------------

template<typename T>
HostArray<T>::HostArray(T* dataPtr, uint32_t numElements) noexcept
{
	this->create(numElements);
	this->upload(dataPtr, numElements);
}

template<typename T>
HostArray<T>::HostArray(HostArray&& other) noexcept
{
	this->swap(other);
}

template<typename T>
HostArray<T>& HostArray<T>::operator= (HostArray&& other) noexcept
{
	this->swap(other);
}

template<typename T>
HostArray<T>::~HostArray() noexcept
{
	this->destroy();
}

// HostArray implementation: Methods
// ------------------------------------------------------------------------------------------------

template<typename T>
void HostArray<T>::create(uint32_t capacity) noexcept
{
	if (mCapacity != 0u) this->destroy();
	size_t numBytes = capacity * sizeof(T);
	CHECK_CUDA_ERROR(cudaMalloc(&mCudaPtr, numBytes));
	mCapacity = capacity;
}

template<typename T>
void HostArray<T>::upload(T* dataPtr, uint32_t numElements) noexcept
{
	sfz_assert_debug(numElements <= mCapacity);
	size_t numBytes = numElements * sizeof(T);
	CHECK_CUDA_ERROR(cudaMemcpy(mCudaPtr, dataPtr, numBytes, cudaMemcpyHostToDevice));
}

template<typename T>
void HostArray<T>::destroy() noexcept
{
	CHECK_CUDA_ERROR(cudaFree(mCudaPtr));
	mCudaPtr = nullptr;
	mCapacity = 0u;
	mSize = 0u;
}

template<typename T>
void HostArray<T>::swap(HostArray& other) noexcept
{
	std::swap(this->mCudaPtr, other.mCudaPtr);
	std::swap(this->mCapacity, other.mCapacity);
	std::swap(this->mSize, other.mSize);
}

} // namespace phe
