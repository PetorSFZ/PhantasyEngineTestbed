// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include <cstdint>

#include <sfz/containers/DynArray.hpp>
#include <sfz/math/Vector.hpp>

namespace phe {

using std::uint8_t;
using std::uint32_t;
using sfz::DynArray;
using sfz::vec2i;

// RawImage
// ------------------------------------------------------------------------------------------------

/// A struct holding a raw image in memory
struct RawImage final {
	// Members
	// --------------------------------------------------------------------------------------------

	DynArray<uint8_t> imgData;
	vec2i dim = vec2i(0);
	uint32_t bytesPerPixel = 0;
	uint32_t pitch = 0; // Length of row in bytes

	// Methods
	// --------------------------------------------------------------------------------------------

	inline bool isEmpty() const noexcept { return imgData.data() == nullptr; }

	/// Flips the image vertically, i.e. the top row will become the bottom row, etc.
	void flipVertically() noexcept;

	uint8_t* getPixelPtr(int32_t x, int32_t y) noexcept;
	uint8_t* getPixelPtr(vec2i location) noexcept;
	const uint8_t* getPixelPtr(int32_t x, int32_t y) const noexcept;
	const uint8_t* getPixelPtr(vec2i location) const noexcept;
};

/// Loads a RawImage from file using stb_image
RawImage loadImage(const char* basePath, const char* fileName) noexcept;

} // namespace phe
