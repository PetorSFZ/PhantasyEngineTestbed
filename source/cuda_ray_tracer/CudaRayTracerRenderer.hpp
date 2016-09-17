// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma

#include "phantasy_engine/rendering/BaseRenderer.hpp"

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

	void bakeMaterials(const DynArray<RawImage>& textures,
	                   const DynArray<Material>& materials) noexcept override final;

	void addMaterial(RawImage& texture, Material& material) noexcept override final;

	void bakeStaticScene(const StaticScene& staticScene) noexcept override final;

	RenderResult render(Framebuffer& resultFB) noexcept override final;

protected:
	// Protected virtual methods from BaseRenderer interface
	// --------------------------------------------------------------------------------------------

	void targetResolutionUpdated() noexcept override final;

private:
	// Private members
	// --------------------------------------------------------------------------------------------
	
	CUDARayTracerRendererImpl* mImpl = nullptr;
};

} // namespace phe
