// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include "phantasy_engine/rendering/BaseRenderer.hpp"

namespace phe {

// DeferredRenderer
// ------------------------------------------------------------------------------------------------

class DeferredRendererImpl; // Pimpl pattern

class DeferredRenderer final : public BaseRenderer {
public:

	// Constructors & destructors
	// --------------------------------------------------------------------------------------------

	DeferredRenderer(const DeferredRenderer&) = delete;
	DeferredRenderer& operator= (const DeferredRenderer&) = delete;
	
	DeferredRenderer() noexcept;
	DeferredRenderer(DeferredRenderer&& other) noexcept;
	DeferredRenderer& operator= (DeferredRenderer&& other) noexcept;
	~DeferredRenderer() noexcept;

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
	
	DeferredRendererImpl* mImpl = nullptr;
};

} // namespace phe
