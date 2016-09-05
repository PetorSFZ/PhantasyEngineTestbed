// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include <sfz/gl/Program.hpp>

#include "phantasy_engine/renderers/BaseRenderer.hpp"
#include "phantasy_engine/renderers/FullscreenTriangle.hpp"

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

	RenderResult render(Framebuffer& resultFB) noexcept override final;

protected:
	// Protected virtual methods from BaseRenderer interface
	// --------------------------------------------------------------------------------------------

	void staticSceneChanged() noexcept override final;

	void targetResolutionUpdated() noexcept override final;

private:
	// Private members
	// --------------------------------------------------------------------------------------------
	
	Program mGBufferGenShader, mShadingShader;
	Framebuffer mGBuffer;
	FullscreenTriangle mFullscreenTriangle;
};

} // namespace phe
