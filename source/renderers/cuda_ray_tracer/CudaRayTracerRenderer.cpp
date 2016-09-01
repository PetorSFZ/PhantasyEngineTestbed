#include "CudaRayTracerRenderer.hpp"

#include <sfz/gl/Program.hpp>
#include <sfz/memory/New.hpp>
#include <sfz/util/IO.hpp>

#include <sfz/gl/IncludeOpenGL.hpp>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include "renderers/cuda_ray_tracer/CUDATest.cuh"
#include "renderers/FullscreenTriangle.hpp"

// CUDA helpers
// ------------------------------------------------------------------------------------------------

#define CHECK_CUDA_ERROR(error) (checkCudaError(__FILE__, __LINE__, error))
static cudaError_t checkCudaError(const char* file, int line, cudaError_t error) noexcept
{
	if (error == cudaSuccess) return error;
	sfz::printErrorMessage("%s:%i: cuda state error %s\n", file, line, cudaGetErrorString(error));
	return error;
}

namespace sfz {

// CUDARayTracerRendererImpl
// ------------------------------------------------------------------------------------------------

class CUDARayTracerRendererImpl final {
public:
	Framebuffer result;
	gl::Program floatToUintShader;
	FullscreenTriangle fullscreenQuad;

	// Temp
	GLuint glTex = 0;
	cudaGraphicsResource_t cudaResource;
	cudaArray_t cudaArray;

	CUDARayTracerRendererImpl() noexcept
	{
		
	}

	~CUDARayTracerRendererImpl() noexcept
	{
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

RenderResult CUDARayTracerRenderer::render(const DynArray<DrawOp>& operations,
                                           const DynArray<PointLight>& pointLights) noexcept
{
	GLuint glTex = mImpl->glTex;
	cudaGraphicsResource_t& resource = mImpl->cudaResource;
	cudaArray_t& array = mImpl->cudaArray;

	// cudaMemcpy color to texture
	uint32_t size = mResolution.x * mResolution.y;
	DynArray<vec4> floats(size, vec4(0.0f, 1.0f, 1.0f, 1.0f), size);
	cudaMemcpyToArray(array, 0, 0, floats.data(), floats.size() * sizeof(vec4), cudaMemcpyHostToDevice);
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
	tmp.colorTex = result.texture(0);
	tmp.colorTexRes = result.dimensions();
	tmp.colorTexRenderedRes = mResolution;
	return tmp;
}

// CUDARayTracerRenderer: Protected virtual methods from BaseRenderer interface
// ------------------------------------------------------------------------------------------------

void CUDARayTracerRenderer::maxResolutionUpdated() noexcept
{
	using gl::FBTextureFiltering;
	using gl::FBTextureFormat;
	using gl::FramebufferBuilder;

	mImpl->result = FramebufferBuilder(mMaxResolution)
	                .addTexture(0, FBTextureFormat::RGB_U8, FBTextureFiltering::LINEAR)
	                .build();


	glActiveTexture(GL_TEXTURE0);
	GLuint& glTex = mImpl->glTex;

	CHECK_CUDA_ERROR(cudaGraphicsUnregisterResource(mImpl->cudaResource));
	glDeleteTextures(1, &glTex);

	glGenTextures(1, &glTex);
	glBindTexture(GL_TEXTURE_2D, glTex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, mMaxResolution.x, mMaxResolution.y, 0, GL_RGBA, GL_FLOAT, nullptr);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);

	// https://github.com/nvpro-samples/gl_cuda_interop_pingpong_st
	cudaGraphicsResource_t& resource = mImpl->cudaResource;
	cudaArray_t& array = mImpl->cudaArray;
	CHECK_CUDA_ERROR(cudaGraphicsGLRegisterImage(&resource, glTex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore));
	CHECK_CUDA_ERROR(cudaGraphicsMapResources(1, &resource, 0));
	CHECK_CUDA_ERROR(cudaGraphicsSubResourceGetMappedArray(&array, resource, 0, 0));
	CHECK_CUDA_ERROR(cudaGraphicsUnmapResources(1, &resource, 0));
}

void CUDARayTracerRenderer::resolutionUpdated() noexcept
{

}

} // namespace sfz
