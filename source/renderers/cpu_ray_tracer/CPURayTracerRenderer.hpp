#pragma

#include "renderers/BaseRenderer.hpp"

namespace sfz {

// CUDARayTracerRenderer
// ------------------------------------------------------------------------------------------------

class CPURayTracerRenderer final : public BaseRenderer {
public:

	// Constructors & destructors
	// --------------------------------------------------------------------------------------------

	CPURayTracerRenderer(const CPURayTracerRenderer&) = delete;
	CPURayTracerRenderer& operator= (const CPURayTracerRenderer&) = delete;
	CPURayTracerRenderer(CPURayTracerRenderer&&) noexcept = default;
	CPURayTracerRenderer& operator= (CPURayTracerRenderer&&) noexcept = default;

	CPURayTracerRenderer() noexcept;

	// Virtual methods from BaseRenderer interface
	// --------------------------------------------------------------------------------------------

	void render(const DynArray<DrawOp>& operations, const DynArray<PointLight>& pointLights) noexcept override final;
	const Framebuffer& getResult() const noexcept override final;
	const Framebuffer& getResultVR(uint32_t eye) const noexcept override final;

protected:
	// Protected virtual methods from BaseRenderer interface
	// --------------------------------------------------------------------------------------------

	void maxResolutionUpdated() noexcept override final;
	void resolutionUpdated() noexcept override final;

private:
	// Private members
	// --------------------------------------------------------------------------------------------
	
	Framebuffer mResult;
	Framebuffer mResultVR[2];
};

} // namespace sfz
