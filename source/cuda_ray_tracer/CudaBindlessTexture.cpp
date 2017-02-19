// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#include "CudaBindlessTexture.hpp"

#include <algorithm>

#include <sfz/cuda/CudaUtils.hpp>

namespace phe {

// CudaBindlessTexture: Constructors & destructors
// ------------------------------------------------------------------------------------------------

CudaBindlessTexture::CudaBindlessTexture(CudaBindlessTexture&& other) noexcept
{
	this->swap(other);
}

CudaBindlessTexture& CudaBindlessTexture::operator= (CudaBindlessTexture&& other) noexcept
{
	this->swap(other);
	return *this;
}

CudaBindlessTexture::~CudaBindlessTexture() noexcept
{
	this->destroy();
}

// CudaBindlessTexture: Methods
// ------------------------------------------------------------------------------------------------

void CudaBindlessTexture::load(const RawImage& image) noexcept
{
	// Make sure texture doesn't already exist
	this->destroy();

	// Allocate CUDA array in device memory
	cudaChannelFormatDesc channelFormatDesc;
	switch (image.bytesPerPixel) {
	case 1:
		channelFormatDesc = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
		break;
	case 2:
		channelFormatDesc = cudaCreateChannelDesc(8, 8, 0, 0, cudaChannelFormatKindUnsigned);
		break;
	case 4:
		channelFormatDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
		break;
	default:
		sfz::error("Invalid number of components, can't create cudaChannelFormatDesc.");
	}
	CHECK_CUDA_ERROR(cudaMallocArray(&mCudaArray, &channelFormatDesc, image.dim.x, image.dim.y));

	// Copy image to CUDA array
	CHECK_CUDA_ERROR(cudaMemcpyToArray(mCudaArray, 0, 0, image.imgData.data(),
	                                   image.imgData.size(), cudaMemcpyHostToDevice));

	// Specify texture
	struct cudaResourceDesc resourceDesc;
	memset(&resourceDesc, 0, sizeof(cudaResourceDesc));
	resourceDesc.resType = cudaResourceTypeArray;
	resourceDesc.res.array.array = mCudaArray;

	// Specify texture object parameters
	struct cudaTextureDesc textureDesc;
	memset(&textureDesc, 0, sizeof(cudaTextureDesc));
	textureDesc.addressMode[0] = cudaAddressModeWrap;
	textureDesc.addressMode[1] = cudaAddressModeWrap;
	textureDesc.filterMode = cudaFilterModePoint;//cudaFilterModeLinear;
	textureDesc.readMode = cudaReadModeElementType;
	textureDesc.normalizedCoords = 1; // Use coord in [0,1] instead of [0, size]

	// Create texture object
	CHECK_CUDA_ERROR(cudaCreateTextureObject(&mTextureObject, &resourceDesc, &textureDesc, NULL));
}

void CudaBindlessTexture::destroy() noexcept
{
	// Destroy texture object
	CHECK_CUDA_ERROR(cudaDestroyTextureObject(mTextureObject));
	mTextureObject = 0;

	// Free device memory
	CHECK_CUDA_ERROR(cudaFreeArray(mCudaArray));
	mCudaArray = nullptr;
}

void CudaBindlessTexture::swap(CudaBindlessTexture& other) noexcept
{
	std::swap(this->mCudaArray, other.mCudaArray);
	std::swap(this->mTextureObject, other.mTextureObject);
}

} // namespace phe
