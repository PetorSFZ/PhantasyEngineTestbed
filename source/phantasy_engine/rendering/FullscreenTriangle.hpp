// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include <cstdint>

namespace phe {

using std::uint32_t;

/// Class used to render post processes shaders in Phantasy Engine.
///
/// Since Phantasy Engine uses D3D/Vulkan clip space the coordinate are slightly different
/// than usual. The uv coordinates uses right handed coordinate system with (0,0) in lower
/// left corner. The positions themselves uses left-handed clip space with (-1,-1) in upper
/// left corner. 
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

} // namespace phe
