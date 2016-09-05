// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#include "CudaRayTracerRenderer.hpp"

#include <sfz/gl/Program.hpp>
#include <sfz/memory/New.hpp>
#include <sfz/util/IO.hpp>

#include <sfz/gl/IncludeOpenGL.hpp>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include "CUDATest.cuh"
#include "phantasy_engine/renderers/FullscreenTriangle.hpp"

// CUDA helpers
// ------------------------------------------------------------------------------------------------

#define CHECK_CUDA_ERROR(error) (checkCudaError(__FILE__, __LINE__, error))
static cudaError_t checkCudaError(const char* file, int line, cudaError_t error) noexcept
{
	if (error == cudaSuccess) return error;
	sfz::printErrorMessage("%s:%i: CUDA error: %s\n", file, line, cudaGetErrorString(error));
	return error;
}

namespace phe {

using namespace sfz;

// CUDARayTracerRendererImpl
// ------------------------------------------------------------------------------------------------

class CUDARayTracerRendererImpl final {
public:
	Framebuffer result;
	gl::Program floatToUintShader;
	FullscreenTriangle fullscreenQuad;

	// Temp
	GLuint glTex = 0;
	cudaGraphicsResource_t cudaResource = 0;
	cudaArray_t cudaArray = 0; // Probably no need to free, since memory is owned by OpenGL
	cudaSurfaceObject_t cudaSurface = 0;

	CUDARayTracerRendererImpl() noexcept
	{
		
	}

	~CUDARayTracerRendererImpl() noexcept
	{
		CHECK_CUDA_ERROR(cudaDestroySurfaceObject(cudaSurface));
		CHECK_CUDA_ERROR(cudaGraphicsUnregisterResource(cudaResource));
		glDeleteTextures(1, &glTex);
	}
};


// CUDARayTracerRenderer: Constructors & destructors
// ------------------------------------------------------------------------------------------------

CUDARayTracerRenderer::CUDARayTracerRenderer() noexcept
{
	mImpl = sfz_new<CUDARayTracerRendererImpl>();

	StackString128 shadersPath;
	shadersPath.printf("%sresources/shaders/", basePath());
	mImpl->floatToUintShader = gl::Program::postProcessFromFile(shadersPath.str, "rgbaf32_to_rgbu8.frag");
}

CUDARayTracerRenderer::~CUDARayTracerRenderer() noexcept
{
	sfz_delete(mImpl);
}

// CUDARayTracerRenderer: Virtual methods from BaseRenderer interface
// ------------------------------------------------------------------------------------------------

RenderResult CUDARayTracerRenderer::render(Framebuffer& resultFB) noexcept
{
	GLuint glTex = mImpl->glTex;
	cudaGraphicsResource_t& resource = mImpl->cudaResource;
	cudaArray_t& array = mImpl->cudaArray;

	writeBlau(mImpl->cudaSurface, mTargetResolution, mTargetResolution);
	cudaDeviceSynchronize();

	// Convert float texture result from cuda into rgb u8
	Framebuffer& result = mImpl->result;
	result.bindViewportClearColorDepth(vec4(0.0f, 0.0f, 0.0f, 0.0f), 0.0f);
	glUseProgram(mImpl->floatToUintShader.handle());

	gl::setUniform(mImpl->floatToUintShader, "uFloatTexture", 0);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, glTex);

	mImpl->fullscreenQuad.render();

	// Return result from cudaaaa shader
	RenderResult tmp;
	tmp.renderedRes = mTargetResolution;
	return tmp;
}

// CUDARayTracerRenderer: Protected virtual methods from BaseRenderer interface
// ------------------------------------------------------------------------------------------------

void CUDARayTracerRenderer::staticSceneChanged() noexcept
{

}

void CUDARayTracerRenderer::targetResolutionUpdated() noexcept
{
	using gl::FBTextureFiltering;
	using gl::FBTextureFormat;
	using gl::FramebufferBuilder;

	mImpl->result = FramebufferBuilder(mTargetResolution)
	                .addTexture(0, FBTextureFormat::RGB_U8, FBTextureFiltering::LINEAR)
	                .build();

	glActiveTexture(GL_TEXTURE0);

	// Cleanup eventual previous texture and bindings
	CHECK_CUDA_ERROR(cudaDestroySurfaceObject(mImpl->cudaSurface));
	CHECK_CUDA_ERROR(cudaGraphicsUnregisterResource(mImpl->cudaResource));
	glDeleteTextures(1, &mImpl->glTex);

	// Create OpenGL texture and allocate memory
	glGenTextures(1, &mImpl->glTex);
	glBindTexture(GL_TEXTURE_2D, mImpl->glTex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, mTargetResolution.x, mTargetResolution.y, 0, GL_RGBA, GL_FLOAT, nullptr);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);

	// https://github.com/nvpro-samples/gl_cuda_interop_pingpong_st
	cudaGraphicsResource_t& resource = mImpl->cudaResource;
	cudaArray_t& array = mImpl->cudaArray;
	CHECK_CUDA_ERROR(cudaGraphicsGLRegisterImage(&resource, mImpl->glTex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore));
	CHECK_CUDA_ERROR(cudaGraphicsMapResources(1, &resource, 0));
	CHECK_CUDA_ERROR(cudaGraphicsSubResourceGetMappedArray(&array, resource, 0, 0));
	CHECK_CUDA_ERROR(cudaGraphicsUnmapResources(1, &resource, 0));

	// Create cuda surface object from binding
	cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(cudaResourceDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = mImpl->cudaArray;
	CHECK_CUDA_ERROR(cudaCreateSurfaceObject(&mImpl->cudaSurface, &resDesc));
}

} // namespace phe
