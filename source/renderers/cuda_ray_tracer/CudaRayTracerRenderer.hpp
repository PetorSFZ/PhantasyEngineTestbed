#pragma

#include "renderers/BaseRenderer.hpp"

namespace sfz {

// CUDARayTracerRenderer
// ------------------------------------------------------------------------------------------------

class CUDARayTracerRenderer final : public BaseRenderer {
public:

	// Constructors & destructors
	// --------------------------------------------------------------------------------------------

	CUDARayTracerRenderer(const CUDARayTracerRenderer&) = delete;
	CUDARayTracerRenderer& operator= (const CUDARayTracerRenderer&) = delete;
	CUDARayTracerRenderer(CUDARayTracerRenderer&&) noexcept = default;
	CUDARayTracerRenderer& operator= (CUDARayTracerRenderer&&) noexcept = default;

	CUDARayTracerRenderer() noexcept;

	// Virtual methods from BaseRenderer interface
	// --------------------------------------------------------------------------------------------

	virtual RenderResult render(const DynArray<DrawOp>& operations,
	                            const DynArray<PointLight>& pointLights) noexcept override final;

protected:
	// Protected virtual methods from BaseRenderer interface
	// --------------------------------------------------------------------------------------------

	void maxResolutionUpdated() noexcept override final;
	void resolutionUpdated() noexcept override final;

private:
	// Private members
	// --------------------------------------------------------------------------------------------
	
	Framebuffer mResult;
};

} // namespace sfz
