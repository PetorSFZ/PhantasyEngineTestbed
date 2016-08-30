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
	
	Program mGBufferGenShader, mShadingShader;
	Framebuffer mGBuffer, mResult, mResultVR[2];
	FullscreenTriangle mFullscreenTriangle;
};

} // namespace sfz
