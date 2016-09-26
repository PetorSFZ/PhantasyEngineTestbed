// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include <cstdint>

namespace phe {

using std::uint32_t;

// Shader Storage Buffer Object
// ------------------------------------------------------------------------------------------------

class SSBO final {
public:
	// Constructors & destructors
	// --------------------------------------------------------------------------------------------

	SSBO() noexcept = default;
	SSBO(const SSBO&) = delete;
	SSBO& operator= (const SSBO&) = delete;

	SSBO(uint32_t numBytes) noexcept;
	SSBO(SSBO&& other) noexcept;
	SSBO& operator= (SSBO&& other) noexcept;
	~SSBO() noexcept;

	// Methods
	// --------------------------------------------------------------------------------------------

	void create(uint32_t numBytes) noexcept;
	void destroy() noexcept;
	void swap(SSBO& other) noexcept;

	void uploadData(const void* dataPtr, uint32_t numBytes) noexcept;

	// Binds the SSBO to the specified binding point, corresponds to "binding" in shader
	void bind(uint32_t binding) noexcept;
	void unbind() noexcept;

	inline bool isValid() const noexcept { return mSSBOHandle != 0u; }
	inline uint32_t handle() const noexcept { return mSSBOHandle; }
	inline uint32_t sizeBytes() const noexcept { return mSizeBytes; }

private:
	// Private members
	// --------------------------------------------------------------------------------------------

	uint32_t mSSBOHandle = 0u;
	uint32_t mSizeBytes = 0u;
};

} // namespace phe
