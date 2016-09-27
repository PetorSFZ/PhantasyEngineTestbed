// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include <cstdint>

#include "phantasy_engine/rendering/RawImage.hpp"

namespace phe {

using std::uint32_t;
using std::uint64_t;

// GLTexture class
// ------------------------------------------------------------------------------------------------

enum class GLTextureFiltering : uint32_t {
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

	void makeResident() noexcept;

	void makeNonResident() noexcept;

	// Getters
	// --------------------------------------------------------------------------------------------

	inline bool isValidTexture() const noexcept { return mTextureHandle != 0u; }
	inline bool isValidBindless() const noexcept { return mBindlessHandle != 0u; }

	// Temporarily disallow access to the normal handle, we are not allowed to change filtering
	// once bindless handle is created
	//inline uint32_t textureHandle() const noexcept { return mTextureHandle; }
	inline uint64_t bindlessHandle() const noexcept { return mBindlessHandle; }

	inline GLTextureFiltering filteringFormat() const noexcept { return mFiltering; }

private:
	// Private methods
	// --------------------------------------------------------------------------------------------

	/// Sets the texture filtering format (may generate mipmaps for some formats)
	void setFilteringFormat(GLTextureFiltering filtering) noexcept;

	// Private members
	// --------------------------------------------------------------------------------------------

	uint32_t mTextureHandle = 0u;
	GLTextureFiltering mFiltering = GLTextureFiltering::NEAREST;
	uint64_t mBindlessHandle = 0u;
};

} // namespace phe
