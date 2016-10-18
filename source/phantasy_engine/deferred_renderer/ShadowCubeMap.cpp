// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#include "phantasy_engine/deferred_renderer/ShadowCubeMap.hpp"

#include <algorithm>

#include <sfz/gl/IncludeOpenGL.hpp>
#include <sfz/gl/Framebuffer.hpp>

namespace phe {

// ShadowCubeMap: Constructors & destructors
// ------------------------------------------------------------------------------------------------

ShadowCubeMap::ShadowCubeMap(uint32_t res) noexcept
{
	mRes = vec2u(res);

	// Generate framebuffer
	glGenFramebuffers(1, &mFbo);
	glBindFramebuffer(GL_FRAMEBUFFER, mFbo);

	// Generate shadow cube map
	glGenTextures(1, &mShadowCubeMap);
	glBindTexture(GL_TEXTURE_CUBE_MAP, mShadowCubeMap);

	// Generates float cube map texture of size width * height for each face
	glTexStorage2D(GL_TEXTURE_CUBE_MAP, 1, GL_DEPTH_COMPONENT32, mRes.x, mRes.y);

	// Texture parameters
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

	// Bind shadow map to framebuffer
	glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, mShadowCubeMap, 0);
	glDrawBuffer(GL_NONE); // No color buffer
	glReadBuffer(GL_NONE);

	// Check that framebuffer is okay
	bool status = sfz::gl::checkCurrentFramebufferStatus();
	sfz_assert_debug(status);

	// Cleanup
	glBindTexture(GL_TEXTURE_CUBE_MAP, 0);
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
	glDeleteTextures(1, &mShadowCubeMap);
	glDeleteFramebuffers(1, &mFbo);
	mRes = vec2u(0u);
	mFbo = 0u;
	mShadowCubeMap = 0u;
}

void ShadowCubeMap::swap(ShadowCubeMap& other) noexcept
{
	std::swap(this->mRes, other.mRes);
	std::swap(this->mFbo, other.mFbo);
	std::swap(this->mShadowCubeMap, other.mShadowCubeMap);
}

} // namespace phe
