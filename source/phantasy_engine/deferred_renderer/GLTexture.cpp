// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#include "phantasy_engine/deferred_renderer/GLTexture.hpp"

#include "sfz/gl/IncludeOpenGL.hpp"

namespace phe {

// GLTexture: Constructors & destructors
// ------------------------------------------------------------------------------------------------

GLTexture::GLTexture(const RawImage& image, GLTextureFiltering filtering) noexcept
{
	this->load(image, filtering);
}

GLTexture::GLTexture(GLTexture&& other) noexcept
{
	this->swap(other);
}

GLTexture& GLTexture::operator= (GLTexture&& other) noexcept
{
	this->swap(other);
	return *this;
}

GLTexture::~GLTexture() noexcept
{
	this->destroy();
}

// GLTexture: Methods
// ------------------------------------------------------------------------------------------------

void GLTexture::load(const RawImage& image, GLTextureFiltering filtering) noexcept
{
	sfz_assert_debug(image.pitch == (image.dim.x * image.bytesPerPixel));
	if (image.imgData.data() == nullptr) return;
	if (mTextureHandle != 0) this->destroy();

	// Creating OpenGL texture
	glGenTextures(1, &mTextureHandle);
	glBindTexture(GL_TEXTURE_2D, mTextureHandle);
	
	// Transfer data from raw image
	switch (image.bytesPerPixel) {
	case 1:
		glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, image.dim.x, image.dim.y, 0, GL_RED,
		             GL_UNSIGNED_BYTE, image.imgData.data());
		break;
	case 2:
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RG8, image.dim.x, image.dim.y, 0, GL_RG,
		             GL_UNSIGNED_BYTE, image.imgData.data());
		break;
	case 3:
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, image.dim.x, image.dim.y, 0, GL_RGB,
		             GL_UNSIGNED_BYTE, image.imgData.data());
		break;
	case 4:
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, image.dim.x, image.dim.y, 0, GL_RGBA,
		             GL_UNSIGNED_BYTE, image.imgData.data());
		break;
	}

	// Set filtering format
	mFiltering = filtering != GLTextureFiltering::NEAREST ?
	             GLTextureFiltering::NEAREST : GLTextureFiltering::BILINEAR;
	setFilteringFormat(filtering);

	// Create Bindless handle
	mBindlessHandle = glGetTextureHandleARB(mTextureHandle);
	sfz_assert_debug(mBindlessHandle != 0u);
	this->makeResident();
}

void GLTexture::destroy() noexcept
{
	this->makeNonResident();
	glDeleteTextures(1, &mTextureHandle); // Silently ignores mTextureHandle == 0
	mTextureHandle = 0u;
	mFiltering = GLTextureFiltering::NEAREST;
	mBindlessHandle = 0u;
}

void GLTexture::swap(GLTexture& other) noexcept
{
	std::swap(this->mTextureHandle, other.mTextureHandle);
	std::swap(this->mFiltering, other.mFiltering);
	std::swap(this->mBindlessHandle, other.mBindlessHandle);
}

void GLTexture::makeResident() noexcept
{
	if (mBindlessHandle != 0u) {
		glMakeTextureHandleResidentARB(mBindlessHandle);
	}
}

void GLTexture::makeNonResident() noexcept
{
	if (mBindlessHandle != 0u) {
		if (glIsTextureHandleResidentARB(mBindlessHandle)) {
			glMakeTextureHandleNonResidentARB(mBindlessHandle);
		}
	}
}

// GLTexture: Private methods
// ------------------------------------------------------------------------------------------------

void GLTexture::setFilteringFormat(GLTextureFiltering filtering) noexcept
{
	if (mTextureHandle == 0u) return;
	if (mFiltering == filtering) return;
	mFiltering = filtering;

	glBindTexture(GL_TEXTURE_2D, mTextureHandle);

	// Sets specified texture filtering, generating mipmaps if needed.
	switch (mFiltering) {
	case GLTextureFiltering::NEAREST:
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		break;
	case GLTextureFiltering::BILINEAR:
		glGenerateMipmap(GL_TEXTURE_2D);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		break;
	case GLTextureFiltering::TRILINEAR:
		glGenerateMipmap(GL_TEXTURE_2D);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		break;
	case GLTextureFiltering::ANISOTROPIC:
		glGenerateMipmap(GL_TEXTURE_2D);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		float factor = 0.0f;
		glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &factor);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, factor);
		break;
	}
}

} // namespace phe
