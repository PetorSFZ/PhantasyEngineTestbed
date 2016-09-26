// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#include "phantasy_engine/deferred_renderer/SSBO.hpp"

#include <algorithm>

#include <sfz/Assert.hpp>
#include <sfz/gl/IncludeOpenGL.hpp>

namespace phe {

// SSBO: Constructors & destructors
// ------------------------------------------------------------------------------------------------

SSBO::SSBO(uint32_t numBytes) noexcept
{
	this->create(numBytes);
}

SSBO::SSBO(SSBO&& other) noexcept
{
	this->swap(other);
}

SSBO& SSBO::operator= (SSBO&& other) noexcept
{
	this->swap(other);
	return *this;
}

SSBO::~SSBO() noexcept
{
	this->destroy();
}

// SSBO: Methods
// ------------------------------------------------------------------------------------------------

void SSBO::create(uint32_t numBytes) noexcept
{
	if (!this->isValid()) this->destroy();

	glGenBuffers(1, &mSSBOHandle);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, mSSBOHandle);
	glBufferData(GL_SHADER_STORAGE_BUFFER, (GLsizeiptr)numBytes, NULL, GL_DYNAMIC_COPY);
	mSizeBytes = numBytes;
}

void SSBO::destroy() noexcept
{
	glDeleteBuffers(1, &mSSBOHandle);
	mSSBOHandle = 0u;
	mSizeBytes = 0u;
}

void SSBO::swap(SSBO& other) noexcept
{
	std::swap(this->mSSBOHandle, other.mSSBOHandle);
	std::swap(this->mSizeBytes, other.mSizeBytes);
}

void SSBO::uploadData(const void* dataPtr, uint32_t numBytes) noexcept
{
	sfz_assert_debug(this->isValid());
	sfz_assert_debug(numBytes <= mSizeBytes);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, mSSBOHandle);
	glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, numBytes, dataPtr);
}

void SSBO::bind(uint32_t binding) noexcept
{
	sfz_assert_debug(this->isValid());
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, mSSBOHandle);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding, mSSBOHandle);
}

void SSBO::unbind() noexcept
{
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

} // namespace phe
