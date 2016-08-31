#include "CPURayTracerRenderer.hpp"

namespace sfz {

// CUDARayTracerRenderer: Constructors & destructors
// ------------------------------------------------------------------------------------------------

	CPURayTracerRenderer::CPURayTracerRenderer() noexcept
{

}

// CUDARayTracerRenderer: Virtual methods from BaseRenderer interface
// ------------------------------------------------------------------------------------------------

void CPURayTracerRenderer::render(const DynArray<DrawOp>& operations, const DynArray<PointLight>& pointLights) noexcept
{
	mResult.bindViewportClearColorDepth(vec4(0.0f, 1.0f, 0.0f, 1.0f), 0.0f);
}

const Framebuffer& CPURayTracerRenderer::getResult() const noexcept
{
	return mResult;
}

const Framebuffer& CPURayTracerRenderer::getResultVR(uint32_t eye) const noexcept
{
	sfz_assert_debug(eye <= 1);
	return mResultVR[eye];
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
