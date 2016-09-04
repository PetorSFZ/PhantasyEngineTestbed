// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include <cstdint>

namespace sfz {

using std::uint32_t;

class FullscreenTriangle final {
public:

	// Copying not allowed
	FullscreenTriangle(const FullscreenTriangle&) = delete;
	FullscreenTriangle& operator= (const FullscreenTriangle&) = delete;

	FullscreenTriangle() noexcept;
	FullscreenTriangle(FullscreenTriangle&& other) noexcept;
	FullscreenTriangle& operator= (FullscreenTriangle&& other) noexcept;
	~FullscreenTriangle() noexcept;

	void render() noexcept;

private:
	uint32_t mVAO = 0;
	uint32_t mPosBuffer = 0;
	uint32_t mUVBuffer = 0;
	uint32_t mIndexBuffer = 0;
};

}
