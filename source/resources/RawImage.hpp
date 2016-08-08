// Copyright (c) Peter Hillerström (skipifzero.com, peter@hstroem.se)

#pragma once

#include <sfz/containers/DynArray.hpp>
#include <sfz/math/Vector.hpp>

namespace sfz {

// RawImage
// ------------------------------------------------------------------------------------------------

/// A class holding a raw image in memory
class RawImage final {
public:
	// Constructors & destructors
	// --------------------------------------------------------------------------------------------

	RawImage() noexcept = default;
	RawImage(const RawImage&) noexcept = default;
	RawImage& operator= (const RawImage&) noexcept = default;
	RawImage(RawImage&&) noexcept = default;
	RawImage& operator= (RawImage&&) noexcept = default;
	~RawImage() noexcept = default;

	// Methods
	// --------------------------------------------------------------------------------------------

	inline bool isEmpty() const noexcept { return mData.data() == nullptr; }
	
	/// Flips the image vertically, i.e. the top row will become the bottom row, etc.
	void flipVertically() noexcept;

	// Getters
	// --------------------------------------------------------------------------------------------

	inline uint8_t* data() noexcept { return mData.data(); }
	inline const uint8_t* data() const noexcept { return mData.data(); }
	inline vec2i dimensions() const noexcept { return mDim; }
	inline int32_t width() const noexcept { return mDim.x; }
	inline int32_t height() const noexcept { return mDim.y; }
	inline uint32_t bytesPerPixel() const noexcept { return mBytesPerPixel; }
	inline uint32_t pitch() const noexcept { return mPitch; }

	uint8_t* getPixelPtr(int32_t x, int32_t y) noexcept;
	uint8_t* getPixelPtr(vec2i location) noexcept;
	const uint8_t* getPixelPtr(int32_t x, int32_t y) const noexcept;
	const uint8_t* getPixelPtr(vec2i location) const noexcept;

private:
	// Private members
	// --------------------------------------------------------------------------------------------

	DynArray<uint8_t> mData;
	vec2i mDim = vec2i(0);
	uint32_t mBytesPerPixel = 0;
	uint32_t mPitch = 0; // Length of row in bytes
};

} // namespace sfz
