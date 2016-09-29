// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#include "CudaGLInterop.hpp"

#include <algorithm>

#include <sfz/gl/IncludeOpenGL.hpp>
#include <cuda_gl_interop.h>

#include "CudaHelpers.hpp"

namespace phe {

// Statics
// ------------------------------------------------------------------------------------------------

static const uint32_t GBUFFER_POSITION = 0u; // uv "u" coordinate stored in w
static const uint32_t GBUFFER_NORMAL = 1u; // uv "v" coordinate stored in w
static const uint32_t GBUFFER_MATERIAL_ID = 2u;

// CudaGLGBuffer: Constructors & destructors
// ------------------------------------------------------------------------------------------------

CudaGLGBuffer::CudaGLGBuffer(vec2i resolution) noexcept
{
	using namespace sfz::gl;

	mFramebuffer = FramebufferBuilder(resolution)
	    .addDepthTexture(FBDepthFormat::F32, FBTextureFiltering::NEAREST)
	    .addTexture(GBUFFER_POSITION, FBTextureFormat::RGBA_F32, FBTextureFiltering::NEAREST)
	    .addTexture(GBUFFER_NORMAL, FBTextureFormat::RGBA_F32, FBTextureFiltering::LINEAR)
	    .addTexture(GBUFFER_MATERIAL_ID, FBTextureFormat::R_INT_U16, FBTextureFiltering::NEAREST)
	    .build();

	// Position texture

	// https://github.com/nvpro-samples/gl_cuda_interop_pingpong_st
	CHECK_CUDA_ERROR(cudaGraphicsGLRegisterImage(
	    &mPosResource, mFramebuffer.texture(GBUFFER_POSITION), GL_TEXTURE_2D,
	    cudaGraphicsRegisterFlagsSurfaceLoadStore));
	CHECK_CUDA_ERROR(cudaGraphicsMapResources(1, &mPosResource, 0));
	cudaArray_t posCudaArray = 0;
	CHECK_CUDA_ERROR(cudaGraphicsSubResourceGetMappedArray(&posCudaArray, mPosResource, 0, 0));
	CHECK_CUDA_ERROR(cudaGraphicsUnmapResources(1, &mPosResource, 0));

	// Create cuda surface object from binding
	cudaResourceDesc posResDesc;
	memset(&posResDesc, 0, sizeof(cudaResourceDesc));
	posResDesc.resType = cudaResourceTypeArray;
	posResDesc.res.array.array = posCudaArray;
	CHECK_CUDA_ERROR(cudaCreateSurfaceObject(&mPosSurface, &posResDesc));

	// Normal texture

	// https://github.com/nvpro-samples/gl_cuda_interop_pingpong_st
	CHECK_CUDA_ERROR(cudaGraphicsGLRegisterImage(
	    &mNormalResource, mFramebuffer.texture(GBUFFER_NORMAL), GL_TEXTURE_2D,
	    cudaGraphicsRegisterFlagsSurfaceLoadStore));
	CHECK_CUDA_ERROR(cudaGraphicsMapResources(1, &mNormalResource, 0));
	cudaArray_t normalCudaArray = 0;
	CHECK_CUDA_ERROR(cudaGraphicsSubResourceGetMappedArray(&normalCudaArray, mNormalResource, 0, 0));
	CHECK_CUDA_ERROR(cudaGraphicsUnmapResources(1, &mNormalResource, 0));

	// Create cuda surface object from binding
	cudaResourceDesc normalResDesc;
	memset(&normalResDesc, 0, sizeof(cudaResourceDesc));
	normalResDesc.resType = cudaResourceTypeArray;
	normalResDesc.res.array.array = normalCudaArray;
	CHECK_CUDA_ERROR(cudaCreateSurfaceObject(&mNormalSurface, &normalResDesc));

	// Material id texture

	// https://github.com/nvpro-samples/gl_cuda_interop_pingpong_st
	CHECK_CUDA_ERROR(cudaGraphicsGLRegisterImage(
	    &mMaterialIdResource, mFramebuffer.texture(GBUFFER_MATERIAL_ID), GL_TEXTURE_2D,
	    cudaGraphicsRegisterFlagsSurfaceLoadStore));
	CHECK_CUDA_ERROR(cudaGraphicsMapResources(1, &mMaterialIdResource, 0));
	cudaArray_t matCudaArray = 0;
	CHECK_CUDA_ERROR(cudaGraphicsSubResourceGetMappedArray(&matCudaArray, mMaterialIdResource, 0, 0));
	CHECK_CUDA_ERROR(cudaGraphicsUnmapResources(1, &mMaterialIdResource, 0));

	// Create cuda surface object from binding
	cudaResourceDesc matResDesc;
	memset(&matResDesc, 0, sizeof(cudaResourceDesc));
	matResDesc.resType = cudaResourceTypeArray;
	matResDesc.res.array.array = matCudaArray;
	CHECK_CUDA_ERROR(cudaCreateSurfaceObject(&mMaterialIdSurface, &matResDesc));
}

CudaGLGBuffer::CudaGLGBuffer(CudaGLGBuffer&& other) noexcept
{
	std::swap(this->mFramebuffer, other.mFramebuffer);

	std::swap(this->mPosResource, other.mPosResource);
	std::swap(this->mNormalResource, other.mNormalResource);
	std::swap(this->mMaterialIdResource, other.mMaterialIdResource);

	std::swap(this->mPosSurface, other.mPosSurface);
	std::swap(this->mNormalSurface, other.mNormalSurface);
	std::swap(this->mMaterialIdSurface, other.mMaterialIdSurface);
}

CudaGLGBuffer& CudaGLGBuffer::operator= (CudaGLGBuffer&& other) noexcept
{
	std::swap(this->mFramebuffer, other.mFramebuffer);

	std::swap(this->mPosResource, other.mPosResource);
	std::swap(this->mNormalResource, other.mNormalResource);
	std::swap(this->mMaterialIdResource, other.mMaterialIdResource);

	std::swap(this->mPosSurface, other.mPosSurface);
	std::swap(this->mNormalSurface, other.mNormalSurface);
	std::swap(this->mMaterialIdSurface, other.mMaterialIdSurface);

	return *this;
}

CudaGLGBuffer::~CudaGLGBuffer() noexcept
{
	// Position texture
	if (mPosSurface != 0) {
		CHECK_CUDA_ERROR(cudaDestroySurfaceObject(mPosSurface));
	}
	if (mPosResource != 0) {
		CHECK_CUDA_ERROR(cudaGraphicsUnregisterResource(mPosResource));
	}
	mPosResource = 0;
	mPosSurface = 0;

	// Normal texture
	if (mNormalSurface != 0) {
		CHECK_CUDA_ERROR(cudaDestroySurfaceObject(mNormalSurface));
	}
	if (mNormalResource != 0) {
		CHECK_CUDA_ERROR(cudaGraphicsUnregisterResource(mNormalResource));
	}
	mNormalResource = 0;
	mNormalSurface = 0;

	// Material id texture
	if (mMaterialIdSurface != 0) {
		CHECK_CUDA_ERROR(cudaDestroySurfaceObject(mMaterialIdSurface));
	}
	if (mMaterialIdResource != 0) {
		CHECK_CUDA_ERROR(cudaGraphicsUnregisterResource(mMaterialIdResource));
	}
	mMaterialIdResource = 0;
	mMaterialIdSurface = 0;
}

// CudaGLGBuffer: Methods
// ------------------------------------------------------------------------------------------------

void CudaGLGBuffer::bindViewportClearColorDepth() noexcept
{
	using namespace sfz;
	mFramebuffer.bindViewportClearColorDepth(vec2i(0), mFramebuffer.dimensions(), vec4(0.0f), 0.0f);
}

// CudaGLGBuffer: Getters
// ------------------------------------------------------------------------------------------------

uint32_t CudaGLGBuffer::depthTextureGL() const noexcept
{
	return mFramebuffer.depthTexture();
}

uint32_t CudaGLGBuffer::positionTextureGL() const noexcept
{
	return mFramebuffer.texture(GBUFFER_POSITION);
}

uint32_t CudaGLGBuffer::normalTextureGL() const noexcept
{
	return mFramebuffer.texture(GBUFFER_NORMAL);
}

uint32_t CudaGLGBuffer::materialIdTextureGL() const noexcept
{
	return mFramebuffer.texture(GBUFFER_MATERIAL_ID);
}

cudaSurfaceObject_t CudaGLGBuffer::positionSurfaceCuda() const noexcept
{
	return mPosSurface;
}

cudaSurfaceObject_t CudaGLGBuffer::normalSurfaceCuda() const noexcept
{
	return mNormalSurface;
}

cudaSurfaceObject_t CudaGLGBuffer::materialIdSurfaceCuda() const noexcept
{
	return mMaterialIdSurface;
}

// CudaGLTexture: Constructors & destructors
// ------------------------------------------------------------------------------------------------

CudaGLTexture::CudaGLTexture(vec2i resolution) noexcept
{
	this->mRes = resolution;

	// Creates OpenGL texture
	glGenTextures(1, &mGLTex);
	glBindTexture(GL_TEXTURE_2D, mGLTex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, mRes.x, mRes.y, 0, GL_RGBA, GL_FLOAT, nullptr);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);

	// https://github.com/nvpro-samples/gl_cuda_interop_pingpong_st
	CHECK_CUDA_ERROR(cudaGraphicsGLRegisterImage(&mCudaResource, mGLTex, GL_TEXTURE_2D,
	                 cudaGraphicsRegisterFlagsSurfaceLoadStore));
	CHECK_CUDA_ERROR(cudaGraphicsMapResources(1, &mCudaResource, 0));
	cudaArray_t cudaArray = 0;
	CHECK_CUDA_ERROR(cudaGraphicsSubResourceGetMappedArray(&cudaArray, mCudaResource, 0, 0));
	CHECK_CUDA_ERROR(cudaGraphicsUnmapResources(1, &mCudaResource, 0));

	// Create cuda surface object from binding
	cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(cudaResourceDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = cudaArray;
	CHECK_CUDA_ERROR(cudaCreateSurfaceObject(&mCudaSurface, &resDesc));
}

CudaGLTexture::CudaGLTexture(CudaGLTexture&& other) noexcept
{
	std::swap(this->mRes, other.mRes);
	std::swap(this->mGLTex, other.mGLTex);
	std::swap(this->mCudaResource, other.mCudaResource);
	std::swap(this->mCudaSurface, other.mCudaSurface);
}

CudaGLTexture& CudaGLTexture::operator= (CudaGLTexture&& other) noexcept
{
	std::swap(this->mRes, other.mRes);
	std::swap(this->mGLTex, other.mGLTex);
	std::swap(this->mCudaResource, other.mCudaResource);
	std::swap(this->mCudaSurface, other.mCudaSurface);
	return *this;
}

CudaGLTexture::~CudaGLTexture() noexcept
{
	if (mCudaSurface != 0) {
		CHECK_CUDA_ERROR(cudaDestroySurfaceObject(mCudaSurface));
	}
	if (mCudaResource != 0) {
		CHECK_CUDA_ERROR(cudaGraphicsUnregisterResource(mCudaResource));
	}
	glDeleteTextures(1, &mGLTex);

	mRes = vec2i(0);
	mGLTex = 0;
	mCudaResource = 0;
	mCudaSurface = 0;
}

} // namespace phe
