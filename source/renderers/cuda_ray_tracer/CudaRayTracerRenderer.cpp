#include "CudaRayTracerRenderer.hpp"

namespace sfz {

// CUDARayTracerRenderer: Constructors & destructors
// ------------------------------------------------------------------------------------------------

CUDARayTracerRenderer::CUDARayTracerRenderer() noexcept
{

}

// CUDARayTracerRenderer: Virtual methods from BaseRenderer interface
// ------------------------------------------------------------------------------------------------

void CUDARayTracerRenderer::render(const DynArray<DrawOp>& operations) noexcept
{
	mResult.bindViewportClearColorDepth(vec4(0.0f, 1.0f, 0.0f, 1.0f), 0.0f);
}

const Framebuffer& CUDARayTracerRenderer::getResult() const noexcept
{
	return mResult;
}

const Framebuffer& CUDARayTracerRenderer::getResultVR(uint32_t eye) const noexcept
{
	sfz_assert_debug(eye <= 1);
	return mResultVR[eye];
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
