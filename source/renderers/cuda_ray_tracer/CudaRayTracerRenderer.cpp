#include "CudaRayTracerRenderer.hpp"

#include <sfz/memory/New.hpp>

#include <sfz/gl/IncludeOpenGL.hpp>
#include <cuda_gl_interop.h>

#include "renderers/cuda_ray_tracer/CUDAGLTexture.hpp"
#include "renderers/cuda_ray_tracer/CUDATest.cuh"

namespace sfz {

// CUDARayTracerRendererImpl
// ------------------------------------------------------------------------------------------------

class CUDARayTracerRendererImpl final {
public:
	Framebuffer result;
	CUDAGLTexture cudaGLTex;
};


// CUDARayTracerRenderer: Constructors & destructors
// ------------------------------------------------------------------------------------------------

CUDARayTracerRenderer::CUDARayTracerRenderer() noexcept
{
	mImpl = sfz_new<CUDARayTracerRendererImpl>();
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
	cudaSurfaceObject_t surface = mImpl->cudaGLTex.cudaMap();

	// Writing blau to surface
	writeBlau(surface, mResolution, mMaxResolution);


	/*Framebuffer& result = mImpl->result;

	result.bindViewportClearColorDepth(vec4(0.0f, 1.0f, 0.0f, 1.0f), 0.0f);
	doSomething();

	RenderResult tmp;
	tmp.colorTex = result.texture(0);
	tmp.colorTexRes = result.dimensions();
	tmp.colorTexRenderedRes = mResolution;*/


	RenderResult tmp;
	tmp.colorTex = mImpl->cudaGLTex.glTexture();
	tmp.colorTexRes = mMaxResolution;
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
