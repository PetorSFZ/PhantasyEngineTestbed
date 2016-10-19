// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#include "ShadowCubeMapCudaGL.hpp"

#include <algorithm>

#include <sfz/gl/IncludeOpenGL.hpp>
#include <sfz/gl/Framebuffer.hpp>

namespace phe {

// ShadowCubeMapCudaGL: Constructors & destructors
// ------------------------------------------------------------------------------------------------

ShadowCubeMapCudaGL::ShadowCubeMapCudaGL(uint32_t res) noexcept
{
	mRes = vec2u(res);

	// Generate framebuffer
	glGenFramebuffers(1, &mFbo);
	glBindFramebuffer(GL_FRAMEBUFFER, mFbo);


	// Generate cube map depth bufffer
	glGenTextures(1, &mCubeDepthTexture);
	glBindTexture(GL_TEXTURE_CUBE_MAP, mCubeDepthTexture);

	// Generates float cube map texture of size width * height for each face
	glTexStorage2D(GL_TEXTURE_CUBE_MAP, 1, GL_DEPTH_COMPONENT32, mRes.x, mRes.y);

	// Texture parameters
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

	// Bind depth buffer map to framebuffer
	glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, mCubeDepthTexture, 0);


	// Generate cube map depth bufffer
	glGenTextures(1, &mShadowCubeMap);
	glBindTexture(GL_TEXTURE_CUBE_MAP, mShadowCubeMap);

	// Generates float cube map texture of size width * height for each face
	glTexStorage2D(GL_TEXTURE_CUBE_MAP, 1, GL_R32F, mRes.x, mRes.y);

	// Texture parameters
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

	// Bind depth buffer map to framebuffer
	glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, mShadowCubeMap, 0);

	// Sets up the textures to draw to
	glDrawBuffer(GL_COLOR_ATTACHMENT0);


	// Check that framebuffer is okay
	bool status = sfz::gl::checkCurrentFramebufferStatus();
	sfz_assert_debug(status);


	// Cleanup
	glBindTexture(GL_TEXTURE_CUBE_MAP, 0);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

ShadowCubeMapCudaGL::ShadowCubeMapCudaGL(ShadowCubeMapCudaGL&& other) noexcept
{
	this->swap(other);
}

ShadowCubeMapCudaGL& ShadowCubeMapCudaGL::operator= (ShadowCubeMapCudaGL&& other) noexcept
{
	this->swap(other);
	return *this;
}

ShadowCubeMapCudaGL::~ShadowCubeMapCudaGL() noexcept
{
	this->destroy();
}

// ShadowCubeMapCudaGL: Methods
// ------------------------------------------------------------------------------------------------

void ShadowCubeMapCudaGL::destroy() noexcept
{
	glDeleteRenderbuffers(1, &mCubeDepthTexture);
	glDeleteTextures(1, &mShadowCubeMap);
	glDeleteFramebuffers(1, &mFbo);
	mRes = vec2u(0u);
	mFbo = 0u;
	mCubeDepthTexture = 0u;
	mShadowCubeMap = 0u;
}

void ShadowCubeMapCudaGL::swap(ShadowCubeMapCudaGL& other) noexcept
{
	std::swap(this->mRes, other.mRes);
	std::swap(this->mFbo, other.mFbo);
	std::swap(this->mCubeDepthTexture, other.mCubeDepthTexture);
	std::swap(this->mShadowCubeMap, other.mShadowCubeMap);
}

} // namespace phe
