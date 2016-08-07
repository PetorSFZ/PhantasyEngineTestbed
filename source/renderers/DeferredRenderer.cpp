// Copyright (c) Peter Hillerström (skipifzero.com, peter@hstroem.se)

#include "renderers/DeferredRenderer.hpp"

#include <sfz/gl/IncludeOpenGL.hpp>

namespace sfz {

// DeferredRenderer: Virtual methods from BaseRenderer interface
// ------------------------------------------------------------------------------------------------

void DeferredRenderer::render(const mat4& viewMatrix, const mat4& projMatrix) noexcept
{
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClearDepth(1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

const Framebuffer& DeferredRenderer::getResult() const noexcept
{
	return mResult;
}

const Framebuffer& DeferredRenderer::getResultVR(uint32_t eye) const noexcept
{
	sfz_assert_debug(eye <= 1);
	return mResultVR[eye];
}

} // namespace sfz
