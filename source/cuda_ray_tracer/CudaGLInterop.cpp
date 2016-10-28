// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#include "CudaGLInterop.hpp"

#include <algorithm>

#include <sfz/gl/IncludeOpenGL.hpp>
#include <cuda_gl_interop.h>

#include "CudaHelpers.hpp"

namespace phe {

// Statics
// ------------------------------------------------------------------------------------------------

static const uint32_t GBUFFER_POSITION = 0u;
static const uint32_t GBUFFER_NORMAL = 1u;
static const uint32_t GBUFFER_ALBEDO = 2u;
static const uint32_t GBUFFER_MATERIAL = 3u;
static const uint32_t GBUFFER_VELOCITY = 4u;

// CudaGLGBuffer: Constructors & destructors
// ------------------------------------------------------------------------------------------------

CudaGLGBuffer::CudaGLGBuffer(vec2i resolution) noexcept
{
	using namespace sfz::gl;

	mFramebuffer = FramebufferBuilder(resolution)
	    .addDepthTexture(FBDepthFormat::F32, FBTextureFiltering::NEAREST)
	    .addTexture(GBUFFER_POSITION, FBTextureFormat::RGBA_F32, FBTextureFiltering::NEAREST)
	    .addTexture(GBUFFER_NORMAL, FBTextureFormat::RGBA_F32, FBTextureFiltering::LINEAR)
	    .addTexture(GBUFFER_ALBEDO, FBTextureFormat::RGBA_U8, FBTextureFiltering::LINEAR)
	    .addTexture(GBUFFER_MATERIAL, FBTextureFormat::RGBA_F32, FBTextureFiltering::LINEAR)
	    .addTexture(GBUFFER_VELOCITY, FBTextureFormat::RGBA_F32, FBTextureFiltering::LINEAR)
	    .build();

	// Position texture

	// https://github.com/nvpro-samples/gl_cuda_interop_pingpong_st
	CHECK_CUDA_ERROR(cudaGraphicsGLRegisterImage(
	    &mPosResource, mFramebuffer.texture(GBUFFER_POSITION), GL_TEXTURE_2D,
	    cudaGraphicsRegisterFlagsReadOnly));
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
	    cudaGraphicsRegisterFlagsReadOnly));
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

	// Albedo texture

	// https://github.com/nvpro-samples/gl_cuda_interop_pingpong_st
	CHECK_CUDA_ERROR(cudaGraphicsGLRegisterImage(
	    &mAlbedoResource, mFramebuffer.texture(GBUFFER_ALBEDO), GL_TEXTURE_2D,
	    cudaGraphicsRegisterFlagsReadOnly));
	CHECK_CUDA_ERROR(cudaGraphicsMapResources(1, &mAlbedoResource, 0));
	cudaArray_t albedoCudaArray = 0;
	CHECK_CUDA_ERROR(cudaGraphicsSubResourceGetMappedArray(&albedoCudaArray, mAlbedoResource, 0, 0));
	CHECK_CUDA_ERROR(cudaGraphicsUnmapResources(1, &mAlbedoResource, 0));

	// Create cuda surface object from binding
	cudaResourceDesc albedoResDesc;
	memset(&albedoResDesc, 0, sizeof(cudaResourceDesc));
	albedoResDesc.resType = cudaResourceTypeArray;
	albedoResDesc.res.array.array = albedoCudaArray;
	CHECK_CUDA_ERROR(cudaCreateSurfaceObject(&mAlbedoSurface, &albedoResDesc));

	// Material id texture

	// https://github.com/nvpro-samples/gl_cuda_interop_pingpong_st
	CHECK_CUDA_ERROR(cudaGraphicsGLRegisterImage(
	    &mMaterialResource, mFramebuffer.texture(GBUFFER_MATERIAL), GL_TEXTURE_2D,
	    cudaGraphicsRegisterFlagsReadOnly));
	CHECK_CUDA_ERROR(cudaGraphicsMapResources(1, &mMaterialResource, 0));
	cudaArray_t matCudaArray = 0;
	CHECK_CUDA_ERROR(cudaGraphicsSubResourceGetMappedArray(&matCudaArray, mMaterialResource, 0, 0));
	CHECK_CUDA_ERROR(cudaGraphicsUnmapResources(1, &mMaterialResource, 0));

	// Create cuda surface object from binding
	cudaResourceDesc matResDesc;
	memset(&matResDesc, 0, sizeof(cudaResourceDesc));
	matResDesc.resType = cudaResourceTypeArray;
	matResDesc.res.array.array = matCudaArray;
	CHECK_CUDA_ERROR(cudaCreateSurfaceObject(&mMaterialSurface, &matResDesc));
}

CudaGLGBuffer::CudaGLGBuffer(CudaGLGBuffer&& other) noexcept
{
	std::swap(this->mFramebuffer, other.mFramebuffer);

	std::swap(this->mPosResource, other.mPosResource);
	std::swap(this->mPosSurface, other.mPosSurface);
	
	std::swap(this->mNormalResource, other.mNormalResource);
	std::swap(this->mNormalSurface, other.mNormalSurface);

	std::swap(this->mAlbedoResource, other.mAlbedoResource);
	std::swap(this->mAlbedoSurface, other.mAlbedoSurface);

	std::swap(this->mMaterialResource, other.mMaterialResource);
	std::swap(this->mMaterialSurface, other.mMaterialSurface);
}

CudaGLGBuffer& CudaGLGBuffer::operator= (CudaGLGBuffer&& other) noexcept
{
	std::swap(this->mFramebuffer, other.mFramebuffer);

	std::swap(this->mPosResource, other.mPosResource);
	std::swap(this->mPosSurface, other.mPosSurface);

	std::swap(this->mNormalResource, other.mNormalResource);
	std::swap(this->mNormalSurface, other.mNormalSurface);

	std::swap(this->mAlbedoResource, other.mAlbedoResource);
	std::swap(this->mAlbedoSurface, other.mAlbedoSurface);

	std::swap(this->mMaterialResource, other.mMaterialResource);
	std::swap(this->mMaterialSurface, other.mMaterialSurface);

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

	// Albedo texture
	if (mAlbedoSurface != 0) {
		CHECK_CUDA_ERROR(cudaDestroySurfaceObject(mAlbedoSurface));
	}
	if (mAlbedoResource != 0) {
		CHECK_CUDA_ERROR(cudaGraphicsUnregisterResource(mAlbedoResource));
	}
	mAlbedoResource = 0;
	mAlbedoSurface = 0;

	// Material texture
	if (mMaterialSurface != 0) {
		CHECK_CUDA_ERROR(cudaDestroySurfaceObject(mMaterialSurface));
	}
	if (mMaterialResource != 0) {
		CHECK_CUDA_ERROR(cudaGraphicsUnregisterResource(mMaterialResource));
	}
	mMaterialResource = 0;
	mMaterialSurface = 0;
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

uint32_t CudaGLGBuffer::albedoTextureGL() const noexcept
{
	return mFramebuffer.texture(GBUFFER_ALBEDO);
}

uint32_t CudaGLGBuffer::materialTextureGL() const noexcept
{
	return mFramebuffer.texture(GBUFFER_MATERIAL);
}

uint32_t CudaGLGBuffer::velocityTextureGL() const noexcept
{
	return mFramebuffer.texture(GBUFFER_VELOCITY);
}

cudaSurfaceObject_t CudaGLGBuffer::positionSurfaceCuda() const noexcept
{
	return mPosSurface;
}

cudaSurfaceObject_t CudaGLGBuffer::normalSurfaceCuda() const noexcept
{
	return mNormalSurface;
}

cudaSurfaceObject_t CudaGLGBuffer::albedoSurfaceCuda() const noexcept
{
	return mAlbedoSurface;
}

cudaSurfaceObject_t CudaGLGBuffer::materialSurfaceCuda() const noexcept
{
	return mMaterialSurface;
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
