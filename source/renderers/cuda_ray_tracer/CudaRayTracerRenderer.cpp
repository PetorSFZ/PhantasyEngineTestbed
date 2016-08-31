#include "CudaRayTracerRenderer.hpp"

#include "renderers/cuda_ray_tracer/CUDATest.cuh"

namespace sfz {

// CUDARayTracerRenderer: Constructors & destructors
// ------------------------------------------------------------------------------------------------

CUDARayTracerRenderer::CUDARayTracerRenderer() noexcept
{

}

// CUDARayTracerRenderer: Virtual methods from BaseRenderer interface
// ------------------------------------------------------------------------------------------------

RenderResult CUDARayTracerRenderer::render(const DynArray<DrawOp>& operations,
                                           const DynArray<PointLight>& pointLights) noexcept
{
	mResult.bindViewportClearColorDepth(vec4(0.0f, 1.0f, 0.0f, 1.0f), 0.0f);
	doSomething();

	RenderResult tmp;
	tmp.colorTex = mResult.texture(0);
	tmp.colorTexRes = mResult.dimensions();
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

	mResult = FramebufferBuilder(mMaxResolution)
	          .addTexture(0, FBTextureFormat::RGB_U8, FBTextureFiltering::LINEAR)
	          .build();
}

void CUDARayTracerRenderer::resolutionUpdated() noexcept
{

}

} // namespace sfz
