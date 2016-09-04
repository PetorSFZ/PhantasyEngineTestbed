#pragma

#include "phantasy_engine/renderers/BaseRenderer.hpp"

namespace sfz {

// CUDARayTracerRenderer
// ------------------------------------------------------------------------------------------------

class CUDARayTracerRendererImpl; // Pimpl pattern

class CUDARayTracerRenderer final : public BaseRenderer {
public:

	// Constructors & destructors
	// --------------------------------------------------------------------------------------------

	CUDARayTracerRenderer(const CUDARayTracerRenderer&) = delete;
	CUDARayTracerRenderer& operator= (const CUDARayTracerRenderer&) = delete;
	CUDARayTracerRenderer(CUDARayTracerRenderer&&) = delete;
	CUDARayTracerRenderer& operator= (CUDARayTracerRenderer&&) = delete;
	
	CUDARayTracerRenderer() noexcept;
	~CUDARayTracerRenderer() noexcept;

	// Virtual methods from BaseRenderer interface
	// --------------------------------------------------------------------------------------------

	RenderResult render(Framebuffer& resultFB,
	                    const DynArray<DrawOp>& operations,
	                    const DynArray<PointLight>& pointLights) noexcept override final;
	void prepareForScene(const Scene& scene) noexcept override final;

protected:
	// Protected virtual methods from BaseRenderer interface
	// --------------------------------------------------------------------------------------------

	void targetResolutionUpdated() noexcept override final;

private:
	// Private members
	// --------------------------------------------------------------------------------------------
	
	CUDARayTracerRendererImpl* mImpl = nullptr;
};

} // namespace sfz
