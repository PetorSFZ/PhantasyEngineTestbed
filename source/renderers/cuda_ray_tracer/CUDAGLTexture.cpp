#include "renderers/cuda_ray_tracer/CUDAGLTexture.hpp"

#include "errorCheck.cuh"

namespace sfz {

// CUDAGLTexture: Constructors & destructors
// ------------------------------------------------------------------------------------------------

CUDAGLTexture::CUDAGLTexture() noexcept
{
	glGenTextures(1, &mGLTex);
	glBindTexture(GL_TEXTURE_2D, mGLTex);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
}

CUDAGLTexture::~CUDAGLTexture() noexcept
{
	this->cudaUnmap();
	this->cudaUnregister();
	glDeleteTextures(1, &mGLTex);
}

// CUDAGLTexture: Methods
// ------------------------------------------------------------------------------------------------

void CUDAGLTexture::setSize(vec2i resolution) noexcept
{
	this->cudaUnmap();
	this->cudaUnregister();
	this->mRes = resolution;

	glBindTexture(GL_TEXTURE_2D, mGLTex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, resolution.x, resolution.y, 0, GL_RGBA, GL_FLOAT, nullptr);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
}

void CUDAGLTexture::cudaRegister() noexcept
{
	if (!mCudaRegistered) {
		glBindTexture(GL_TEXTURE_2D, mGLTex);
		CUDA_CHECKED_CALL cudaGraphicsGLRegisterImage(&mCudaResource, mGLTex, GL_TEXTURE_2D,
		                                              cudaGraphicsRegisterFlagsSurfaceLoadStore);
		mCudaRegistered = true;
	}
}

void CUDAGLTexture::cudaUnregister() noexcept
{
	if (mCudaRegistered) {
		CUDA_CHECKED_CALL cudaGraphicsUnregisterResource(mCudaResource);
		mCudaRegistered = false;
	}
}

cudaSurfaceObject_t CUDAGLTexture::cudaMap() noexcept
{
	this->cudaRegister();
	if (!mCudaMapped) {
		CUDA_CHECKED_CALL cudaGraphicsMapResources(1, &mCudaResource);
		cudaArray_t cuda_array;
		CUDA_CHECKED_CALL cudaGraphicsSubResourceGetMappedArray(&cuda_array, mCudaResource, 0, 0);
		cudaResourceDesc cuda_array_resource_desc;
		memset(&cuda_array_resource_desc, 0, sizeof(cudaResourceDesc));
		cuda_array_resource_desc.resType = cudaResourceTypeArray;
		cuda_array_resource_desc.res.array.array = cuda_array;
		CUDA_CHECKED_CALL cudaCreateSurfaceObject(&mCudaSurfaceObject, &cuda_array_resource_desc);
		mCudaMapped = true;
	}
	return mCudaSurfaceObject;
}

void CUDAGLTexture::cudaUnmap() noexcept
{
	if (mCudaMapped) {
		CUDA_CHECKED_CALL cudaDestroySurfaceObject(mCudaSurfaceObject);
		CUDA_CHECKED_CALL cudaGraphicsUnmapResources(1, &mCudaResource);
		mCudaMapped = false;
	}
}

} // namespace sfz
