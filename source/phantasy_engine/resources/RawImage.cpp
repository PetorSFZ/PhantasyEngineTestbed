// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#include "phantasy_engine/resources/RawImage.hpp"

#include <cmath>
#include <cstring> // std::memcpy

#include <stb_image.h>

#include <sfz/containers/DynString.hpp>
#include <sfz/math/MathHelpers.hpp>
#include <sfz/memory/Allocators.hpp>

namespace phe {

using namespace sfz;

// RawImage: Methods
// ------------------------------------------------------------------------------------------------

void RawImage::flipVertically() noexcept
{
	sfz_assert_debug(imgData.data() != nullptr);
	uint8_t* buffer = static_cast<uint8_t*>(StandardAllocator::allocate(pitch));
	for (int32_t i = 0; i < (dim.y / 2); i++) {
		uint8_t* begin = imgData.data() + i * pitch;
		uint8_t* end = imgData.data() + (dim.y - i - 1) * pitch;

		std::memcpy(buffer, begin, pitch);
		std::memcpy(begin, end, pitch);
		std::memcpy(end, buffer, pitch);
	}
	StandardAllocator::deallocate(buffer);
}

uint8_t* RawImage::getPixelPtr(int32_t x, int32_t y) noexcept
{
	return imgData.data() + (uint32_t(y) * pitch + uint32_t(x) * bytesPerPixel);
}

uint8_t* RawImage::getPixelPtr(vec2i location) noexcept
{
	return this->getPixelPtr(location.x, location.y);
}

const uint8_t* RawImage::getPixelPtr(int32_t x, int32_t y) const noexcept
{
	return imgData.data() + (uint32_t(y) * pitch + uint32_t(x) * bytesPerPixel);
}

const uint8_t* RawImage::getPixelPtr(vec2i location) const noexcept
{
	return this->getPixelPtr(location.x, location.y);
}

RawImage loadImage(const char* basePath, const char* fileName) noexcept
{
	if (basePath == nullptr && fileName == nullptr) {
		printErrorMessage("RawImage: Invalid path to image");
		return RawImage();
	}

	// Concatenate path
	size_t basePathLen = std::strlen(basePath);
	size_t fileNameLen = std::strlen(fileName);
	DynString path("", uint32_t(basePathLen + fileNameLen + 2));
	path.printf("%s%s", basePath, fileName);

	// Loading image
	int width, height, numChannels;
	uint8_t* img = stbi_load(path.str(), &width, &height, &numChannels, 0);

	// Error checking
	if (img == nullptr) {
		printErrorMessage("RawImage: Unable to load image \"%s\", reason: %s", fileName, stbi_failure_reason());
		return RawImage();
	}

	// Create RawImage from data
	RawImage tmp;
	tmp.dim = vec2i(width, height);
	tmp.bytesPerPixel = uint32_t(numChannels);
	tmp.pitch = uint32_t(width * numChannels);
	tmp.imgData.add(img, uint32_t(width * height * numChannels));

	// Free data and return RawImage
	stbi_image_free(img);
	return std::move(tmp);
}

} // namespace phe
