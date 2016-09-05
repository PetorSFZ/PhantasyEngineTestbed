// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma

#include "phantasy_engine/renderers/BaseRenderer.hpp"

namespace phe {

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

	RenderResult render(Framebuffer& resultFB) noexcept override final;

protected:
	// Protected virtual methods from BaseRenderer interface
	// --------------------------------------------------------------------------------------------

	void staticSceneChanged() noexcept override final;

	void targetResolutionUpdated() noexcept override final;

private:
	// Private members
	// --------------------------------------------------------------------------------------------
	
	CUDARayTracerRendererImpl* mImpl = nullptr;
};

} // namespace phe
