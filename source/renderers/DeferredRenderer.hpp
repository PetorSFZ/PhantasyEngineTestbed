// Copyright (c) Peter Hillerström (skipifzero.com, peter@hstroem.se)

#pragma once

#include "renderers/BaseRenderer.hpp"

namespace sfz {

// DeferredRenderer
// ------------------------------------------------------------------------------------------------

class DeferredRenderer final : public BaseRenderer {
public:

	// Constructors & destructors
	// --------------------------------------------------------------------------------------------

	DeferredRenderer() noexcept = default;
	DeferredRenderer(const DeferredRenderer&) = delete;
	DeferredRenderer& operator= (const DeferredRenderer&) = delete;
	DeferredRenderer(DeferredRenderer&&) noexcept = default;
	DeferredRenderer& operator= (DeferredRenderer&&) noexcept = default;

	// Virtual methods from BaseRenderer interface
	// --------------------------------------------------------------------------------------------

	void render(const mat4& viewMatrix, const mat4& projMatrix) noexcept override final;

	const Framebuffer& getResult() const noexcept override final;

	const Framebuffer& getResultVR(uint32_t eye) const noexcept override final;

private:
	// Private members
	// --------------------------------------------------------------------------------------------
	
	Framebuffer mResult, mResultVR[2];
};

} // namespace sfz
