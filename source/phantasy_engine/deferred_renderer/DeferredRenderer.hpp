// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include <sfz/gl/Program.hpp>

#include "phantasy_engine/rendering/BaseRenderer.hpp"
#include "phantasy_engine/rendering/FullscreenTriangle.hpp"

namespace phe {

using sfz::gl::Framebuffer;
using sfz::gl::Program;

// DeferredRenderer
// ------------------------------------------------------------------------------------------------

class DeferredRenderer final : public BaseRenderer {
public:

	// Constructors & destructors
	// --------------------------------------------------------------------------------------------

	DeferredRenderer(const DeferredRenderer&) = delete;
	DeferredRenderer& operator= (const DeferredRenderer&) = delete;
	DeferredRenderer(DeferredRenderer&&) noexcept = default;
	DeferredRenderer& operator= (DeferredRenderer&&) noexcept = default;

	DeferredRenderer() noexcept;

	// Virtual methods from BaseRenderer interface
	// --------------------------------------------------------------------------------------------

	void bakeMaterials(const DynArray<RawImage>& textures,
	                   const DynArray<Material>& materials) noexcept override final;

	void addMaterial(RawImage& texture, Material& material) noexcept override final;

	void bakeStaticScene(const SharedPtr<StaticScene>& staticScene) noexcept override final;

	RenderResult render(Framebuffer& resultFB) noexcept override final;

protected:
	// Protected virtual methods from BaseRenderer interface
	// --------------------------------------------------------------------------------------------

	void targetResolutionUpdated() noexcept override final;

private:
	// Private members
	// --------------------------------------------------------------------------------------------
	
	Program mGBufferGenShader, mShadingShader;
	Framebuffer mGBuffer;
	FullscreenTriangle mFullscreenTriangle;
};

} // namespace phe
