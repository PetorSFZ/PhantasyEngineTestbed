#include "CudaRayTracerRenderer.hpp"

#include <sfz/gl/Program.hpp>
#include <sfz/memory/New.hpp>
#include <sfz/util/IO.hpp>

#include <sfz/gl/IncludeOpenGL.hpp>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include "renderers/cuda_ray_tracer/CUDAGLTexture.hpp"
#include "renderers/cuda_ray_tracer/CUDATest.cuh"
#include "renderers/FullscreenTriangle.hpp"

namespace sfz {

// CUDARayTracerRendererImpl
// ------------------------------------------------------------------------------------------------

class CUDARayTracerRendererImpl final {
public:
	Framebuffer result;
	CUDAGLTexture cudaGLTex;
	gl::Program cudaaaShader;
	FullscreenTriangle fullscreenQuad;
};


// CUDARayTracerRenderer: Constructors & destructors
// ------------------------------------------------------------------------------------------------

CUDARayTracerRenderer::CUDARayTracerRenderer() noexcept
{
	mImpl = sfz_new<CUDARayTracerRendererImpl>();

	StackString128 shadersPath;
	shadersPath.printf("%sresources/shaders/", basePath());
	mImpl->cudaaaShader = gl::Program::postProcessFromFile(shadersPath.str, "cudaaaa.frag");
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
	/*cudaSurfaceObject_t surface = mImpl->cudaGLTex.cudaMap();

	// Writing blau to surface
	writeBlau(surface, mResolution, mMaxResolution);

	uint32_t cudaTex = mImpl->cudaGLTex.glTexture();

	glBindTexture(GL_TEXTURE_2D, cudaTex);
	


	/*Framebuffer& result = mImpl->result;

	result.bindViewportClearColorDepth(vec4(0.0f, 1.0f, 0.0f, 1.0f), 0.0f);
	doSomething();

	RenderResult tmp;
	tmp.colorTex = result.texture(0);
	tmp.colorTexRes = result.dimensions();
	tmp.colorTexRenderedRes = mResolution;*/


	/*RenderResult tmp;
	tmp.colorTex = mImpl->cudaGLTex.glTexture();
	tmp.colorTexRes = mMaxResolution;
	tmp.colorTexRenderedRes = mResolution;
	return tmp;*/

	////// New thing!@!!
	// https://github.com/nvpro-samples/gl_cuda_interop_pingpong_st
	glActiveTexture(GL_TEXTURE0);

	uint32_t size = mResolution.x * mResolution.y;
	DynArray<vec4> fffea(size, vec4(0.0f, 1.0f, 0.0f, 1.0f), size);

	GLuint glTex;
	glGenTextures(1, &glTex);
	glBindTexture(GL_TEXTURE_2D, glTex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, mResolution.x, mResolution.y, 0, GL_RGBA, GL_FLOAT, fffea.data());
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
	
	cudaGraphicsResource_t resource;
	cudaArray_t array;

	cudaGraphicsGLRegisterImage(&resource, glTex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore);
	cudaGraphicsMapResources(1, &resource, 0);

	cudaGraphicsSubResourceGetMappedArray(&array, resource, 0, 0);

	cudaGraphicsUnmapResources(1, &resource, 0);

	// Coiafj
	DynArray<vec4> fllaota(size, vec4(1.0f, 0.0f, 0.0f, 1.0f), size);
	cudaMemcpyToArray(array, 0, 0, fllaota.data(), fllaota.size() * sizeof(vec4), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();

	// Run result from CUDA through cudaaaa shader
	Framebuffer& result = mImpl->result;
	result.bindViewportClearColorDepth(vec4(0.0f, 0.0f, 0.0f, 0.0f), 0.0f);
	glUseProgram(mImpl->cudaaaShader.handle());

	gl::setUniform(mImpl->cudaaaShader, "uCudaResultTex", 0);
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

	mImpl->cudaGLTex.setSize(mMaxResolution);
}

void CUDARayTracerRenderer::resolutionUpdated() noexcept
{

}

} // namespace sfz
