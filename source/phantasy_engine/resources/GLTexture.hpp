// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include "phantasy_engine/rendering/RawImage.hpp"

namespace phe {

// GLTexture class
// ------------------------------------------------------------------------------------------------

enum class GLTextureFiltering {
	NEAREST,
	BILINEAR,
	TRILINEAR,
	ANISOTROPIC
};

// GLTexture class
// ------------------------------------------------------------------------------------------------

class GLTexture final {
public:
	// Constructors & destructors
	// --------------------------------------------------------------------------------------------

	GLTexture() noexcept = default;
	GLTexture(const GLTexture&) = delete;
	GLTexture& operator= (const GLTexture&) = delete;

	GLTexture(const RawImage& image, GLTextureFiltering filtering = GLTextureFiltering::ANISOTROPIC) noexcept;
	GLTexture(GLTexture&& other) noexcept;
	GLTexture& operator= (GLTexture&& other) noexcept;
	~GLTexture() noexcept;

	// Methods
	// --------------------------------------------------------------------------------------------

	/// Loads this GLTexture
	void load(const RawImage& image, GLTextureFiltering filtering = GLTextureFiltering::ANISOTROPIC) noexcept;

	/// Destroys this GLTexture
	void destroy() noexcept;

	/// Swaps this texture with another texture
	void swap(GLTexture& other) noexcept;

	/// Sets the texture filtering format (may generate mipmaps for some formats)
	void setFilteringFormat(GLTextureFiltering filtering) noexcept;

	// Getters
	// --------------------------------------------------------------------------------------------

	inline bool isValid() const noexcept { return mHandle != 0; }

	inline uint32_t handle() const noexcept { return mHandle; }

	inline GLTextureFiltering filteringFormat() const noexcept { return mFiltering; }

private:
	// Private members
	// --------------------------------------------------------------------------------------------

	uint32_t mHandle = 0;
	GLTextureFiltering mFiltering;
};

} // namespace phe
