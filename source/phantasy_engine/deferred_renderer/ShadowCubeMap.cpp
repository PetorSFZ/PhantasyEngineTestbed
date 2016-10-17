// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#include "phantasy_engine/deferred_renderer/ShadowCubeMap.hpp"

#include <algorithm>

#include <sfz/gl/IncludeOpenGL.hpp>

namespace phe {

// ShadowCubeMap: Constructors & destructors
// ------------------------------------------------------------------------------------------------

ShadowCubeMap::ShadowCubeMap(vec2u res, FBDepthFormat depthFormat, bool pcf) noexcept
{
	// Generate framebuffer
	glGenFramebuffers(1, &mFBO);
	glBindFramebuffer(GL_FRAMEBUFFER, mFBO);

	// Generates depth texture
	glGenTextures(1, &mDepthTexture);
	glBindTexture(GL_TEXTURE_2D, mDepthTexture);
	switch (depthFormat) {
	case FBDepthFormat::F16:
		glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT16, res.x, res.y, 0,
		             GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
		break;
	case FBDepthFormat::F24:
		glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, res.x, res.y, 0,
		             GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
		break;
	case FBDepthFormat::F32:
		glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32, res.x, res.y, 0,
		             GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
		break;
	}

	// Set shadowmap texture min & mag filters (enable/disable pcf)
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, pcf ? GL_LINEAR : GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, pcf ? GL_LINEAR : GL_NEAREST);

	// Set texture wrap mode to CLAMP_TO_BORDER and set border color.
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
	sfz::vec4 borderColor(0.0f);
	glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor.elements);

	// Enable hardware shadow maps (becomes sampler2Dshadow)
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_REF_TO_TEXTURE);

	// Bind texture to framebuffer
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, mDepthTexture, 0);
	glDrawBuffer(GL_NONE); // No color buffer
	glReadBuffer(GL_NONE);

	// Check that framebuffer is okay
	bool status = sfz::gl::checkCurrentFramebufferStatus();
	sfz_assert_debug(status);

	// Cleanup
	glBindTexture(GL_TEXTURE_2D, 0);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

ShadowCubeMap::ShadowCubeMap(ShadowCubeMap&& other) noexcept
{
	this->swap(other);
}

ShadowCubeMap& ShadowCubeMap::operator= (ShadowCubeMap&& other) noexcept
{
	this->swap(other);
	return *this;
}

ShadowCubeMap::~ShadowCubeMap() noexcept
{
	this->destroy();
}

// ShadowCubeMap: Methods
// ------------------------------------------------------------------------------------------------

void ShadowCubeMap::destroy() noexcept
{
	glDeleteFramebuffers(1, &mFBO);
	mFBO = 0u;
	// TODO: MOAR
}

void ShadowCubeMap::swap(ShadowCubeMap& other) noexcept
{
	std::swap(this->mFBO, other.mFBO);
	std::swap(this->mDepthTexture, other.mDepthTexture);
	// TODO: MOAR
}

} // namespace phe
