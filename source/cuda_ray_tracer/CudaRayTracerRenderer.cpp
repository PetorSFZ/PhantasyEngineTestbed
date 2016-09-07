// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#include "CudaRayTracerRenderer.hpp"

#include <sfz/gl/Program.hpp>
#include <sfz/math/MathHelpers.hpp>
#include <sfz/memory/New.hpp>
#include <sfz/util/IO.hpp>

#include <sfz/gl/IncludeOpenGL.hpp>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include <phantasy_engine/renderers/FullscreenTriangle.hpp>

#include "CudaTracerEntry.cuh"

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
	gl::Program transferShader;
	FullscreenTriangle fullscreenTriangle;

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
	mImpl->transferShader = gl::Program::postProcessFromFile(shadersPath.str, "transfer.frag");
	glUseProgram(mImpl->transferShader.handle());
	gl::setUniform(mImpl->transferShader, "uSrcTexture", 0);
}

CUDARayTracerRenderer::~CUDARayTracerRenderer() noexcept
{
	sfz_delete(mImpl);
}

// CUDARayTracerRenderer: Virtual methods from BaseRenderer interface
// ------------------------------------------------------------------------------------------------

RenderResult CUDARayTracerRenderer::render(Framebuffer& resultFB) noexcept
{
	// Calculate camera variables
	CameraDef cam;
	cam.origin = mMatrices.position;
	cam.dir = normalize(mMatrices.forward);
	cam.up = normalize(mMatrices.up);
	sfz_assert_debug(approxEqual(dot(cam.dir, cam.up), 0.0f));
	cam.right = normalize(cross(cam.dir, cam.up));
	cam.vertFovRad; 

	// Run CUDA ray tracer
	runCudaRayTracer(mImpl->cudaSurface, mTargetResolution, cam);
	
	// Transfer result from Cuda texture to result framebuffer
	glUseProgram(mImpl->transferShader.handle());
	resultFB.bindViewportClearColorDepth(vec4(0.0f, 0.0f, 0.0f, 0.0f), 0.0f);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, mImpl->glTex);

	mImpl->fullscreenTriangle.render();

	// Return result
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
	glActiveTexture(GL_TEXTURE0);

	// Cleanup eventual previous texture and bindings
	if (mImpl->cudaSurface != 0) {
		CHECK_CUDA_ERROR(cudaDestroySurfaceObject(mImpl->cudaSurface));
		mImpl->cudaSurface = 0;
	}
	if (mImpl->cudaResource != 0) {
		CHECK_CUDA_ERROR(cudaGraphicsUnregisterResource(mImpl->cudaResource));
		mImpl->cudaResource = 0;
	}
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
