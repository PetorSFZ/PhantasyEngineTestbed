// Copyright (c) Peter Hillerström (skipifzero.com, peter@hstroem.se)

#include "resources/RawImage.hpp"

#include <cstring> // std::memcpy

#include <sfz/memory/Allocators.hpp>

namespace sfz {

// RawImage: Methods
// ------------------------------------------------------------------------------------------------

void RawImage::flipVertically() noexcept
{
	sfz_assert_debug(mData.data() != nullptr);
	uint8_t* buffer = static_cast<uint8_t*>(StandardAllocator::allocate(mPitch));
	for (int32_t i = 0; i < (mDim.y/2); i++) {
		uint8_t* begin = mData.data() + i * mPitch;
		uint8_t* end = mData.data() + (mDim.y - i - 1) * mPitch;

		std::memcpy(buffer, begin, mPitch);
		std::memcpy(begin, end, mPitch);
		std::memcpy(end, buffer, mPitch);
	}
	StandardAllocator::deallocate(buffer);
}

// RawImage: Getters
// ------------------------------------------------------------------------------------------------

uint8_t* RawImage::getPixelPtr(int32_t x, int32_t y) noexcept
{
	uint8_t* img = this->data();
	return img + (uint32_t(y) * mPitch + uint32_t(x) * mBytesPerPixel);
}

uint8_t* RawImage::getPixelPtr(vec2i location) noexcept
{
	return this->getPixelPtr(location.x, location.y);
}

const uint8_t* RawImage::getPixelPtr(int32_t x, int32_t y) const noexcept
{
	const uint8_t* img = this->data();
	return img + (uint32_t(y) * mPitch + uint32_t(x) * mBytesPerPixel);
}

const uint8_t* RawImage::getPixelPtr(vec2i location) const noexcept
{
	return this->getPixelPtr(location.x, location.y);
}

} // namespace sfz
