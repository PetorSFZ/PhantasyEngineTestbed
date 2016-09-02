// Copyright (c) Peter Hillerström (skipifzero.com, peter@hstroem.se)

#pragma once

#include <sfz/gl/Program.hpp>

#include "renderers/BaseRenderer.hpp"
#include "renderers/FullscreenTriangle.hpp"

namespace sfz {

using gl::Program;

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

	RenderResult render(const DynArray<DrawOp>& operations,
	                            const DynArray<PointLight>& pointLights) noexcept override final;
	void prepareForScene(const Scene& scene) noexcept override final;

protected:
	// Protected virtual methods from BaseRenderer interface
	// --------------------------------------------------------------------------------------------

	void targetResolutionUpdated() noexcept override final;

private:
	// Private members
	// --------------------------------------------------------------------------------------------
	
	Program mGBufferGenShader, mShadingShader;
	Framebuffer mGBuffer, mResult;
	FullscreenTriangle mFullscreenTriangle;
};

} // namespace sfz
