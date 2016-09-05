// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#include "phantasy_engine/renderers/FullscreenTriangle.hpp"

#include <algorithm>

#include "sfz/gl/IncludeOpenGL.hpp"

namespace phe {

using std::int32_t;

FullscreenTriangle::FullscreenTriangle() noexcept
{
	const float positions[] = {
		-3.0f, -1.0f, 0.0f, // top-left
		1.0f, -1.0f, 0.0f, // top-right
		1.0f, 3.0f, 0.0f // bottom-right
	};
	
	const float uvCoords[] = {
		// top-left UV
		-1.0f, 0.0f,
		// top-right UV
		1.0f, 0.0f,
		// bottom-right UV
		1.0f, 2.0f
	};	
	const unsigned int indices[] = {
		0, 1, 2 // Clock-wise order
	};

	// Buffer objects
	glGenBuffers(1, &mPosBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, mPosBuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(positions), positions, GL_STATIC_DRAW);

	glGenBuffers(1, &mUVBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, mUVBuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(uvCoords), uvCoords, GL_STATIC_DRAW);

	glGenBuffers(1, &mIndexBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, mIndexBuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

	// Vertex Array Object
	glGenVertexArrays(1, &mVAO);
	glBindVertexArray(mVAO);

	glBindBuffer(GL_ARRAY_BUFFER, mPosBuffer);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(0);

	glBindBuffer(GL_ARRAY_BUFFER, mUVBuffer);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(1);
}

FullscreenTriangle::FullscreenTriangle(FullscreenTriangle&& other) noexcept
{
	std::swap(this->mVAO, other.mVAO);
	std::swap(this->mPosBuffer, other.mPosBuffer);
	std::swap(this->mUVBuffer, other.mUVBuffer);
	std::swap(this->mIndexBuffer, other.mIndexBuffer);
}

FullscreenTriangle& FullscreenTriangle::operator= (FullscreenTriangle&& other) noexcept
{
	std::swap(this->mVAO, other.mVAO);
	std::swap(this->mPosBuffer, other.mPosBuffer);
	std::swap(this->mUVBuffer, other.mUVBuffer);
	std::swap(this->mIndexBuffer, other.mIndexBuffer);
	return *this;
}

FullscreenTriangle::~FullscreenTriangle() noexcept
{
	glDeleteBuffers(1, &mPosBuffer);
	glDeleteBuffers(1, &mUVBuffer);
	glDeleteBuffers(1, &mIndexBuffer);
	glDeleteVertexArrays(1, &mVAO);
}

void FullscreenTriangle::render() noexcept
{
	glBindVertexArray(mVAO);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mIndexBuffer);
	glDrawElements(GL_TRIANGLES, 3, GL_UNSIGNED_INT, 0);
}

} // namespace phe
