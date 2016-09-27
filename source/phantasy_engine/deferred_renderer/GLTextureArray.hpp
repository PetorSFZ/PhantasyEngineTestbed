// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include <cstdint>

#include <sfz/containers/DynArray.hpp>

#include "phantasy_engine/rendering/RawImage.hpp"

namespace phe {

using sfz::DynArray;
using std::uint32_t;

// GLTextureArray
// ------------------------------------------------------------------------------------------------

class GLTextureArray final {
public:
	// Constructors & destructors
	// --------------------------------------------------------------------------------------------

	GLTextureArray() noexcept = default;
	GLTextureArray(const GLTextureArray&) = delete;
	GLTextureArray& operator= (const GLTextureArray&) = delete;

	GLTextureArray(GLTextureArray&& other) noexcept;
	GLTextureArray& operator= (GLTextureArray& other) noexcept;
	~GLTextureArray() noexcept;

	// Methods
	// --------------------------------------------------------------------------------------------
	
	void create(const DynArray<RawImage>& images) noexcept;
	void destroy() noexcept;
	void swap(GLTextureArray& other) noexcept;

	inline bool isValid() const noexcept { return mHandle != 0u; }
	inline uint32_t textureHandle() const noexcept { return mHandle; }

private:
	// Private members
	// --------------------------------------------------------------------------------------------

	uint32_t mHandle = 0u;
};

} // namespace phe
