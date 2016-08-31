#include "CPURayTracerRenderer.hpp"

namespace sfz {

// CUDARayTracerRenderer: Constructors & destructors
// ------------------------------------------------------------------------------------------------

	CPURayTracerRenderer::CPURayTracerRenderer() noexcept
{

}

// CUDARayTracerRenderer: Virtual methods from BaseRenderer interface
// ------------------------------------------------------------------------------------------------

RenderResult CPURayTracerRenderer::render(const DynArray<DrawOp>& operations, const DynArray<PointLight>& pointLights) noexcept
{
	mResult.bindViewportClearColorDepth(vec4(0.0f, 1.0f, 0.0f, 1.0f), 0.0f);

	RenderResult tmp;
	tmp.colorTex = mResult.texture(0);
	tmp.colorTexRes = mResult.dimensions();
	tmp.colorTexRenderedRes = mResolution;
	return tmp;
}

// CUDARayTracerRenderer: Protected virtual methods from BaseRenderer interface
// ------------------------------------------------------------------------------------------------

void CPURayTracerRenderer::maxResolutionUpdated() noexcept
{
	using gl::FBTextureFiltering;
	using gl::FBTextureFormat;
	using gl::FramebufferBuilder;

	mResult = FramebufferBuilder(mMaxResolution)
	          .addTexture(0, FBTextureFormat::RGB_U8, FBTextureFiltering::LINEAR)
	          .build();
}

void CPURayTracerRenderer::resolutionUpdated() noexcept
{

}

} // namespace sfz
