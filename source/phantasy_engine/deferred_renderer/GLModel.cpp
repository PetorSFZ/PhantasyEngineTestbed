// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#include "phantasy_engine/deferred_renderer/GLModel.hpp"

#include <algorithm>
#include <cstddef> // offsetof()

#include <sfz/gl/IncludeOpenGL.hpp>

namespace phe {

// GLModel: Constructors & destructors
// ------------------------------------------------------------------------------------------------

GLModel::GLModel(const RawMesh& mesh) noexcept
{
	this->load(mesh);
}

GLModel::GLModel(GLModel&& other) noexcept
{
	this->swap(other);
}

GLModel& GLModel::operator= (GLModel&& other) noexcept
{
	this->swap(other);
	return *this;
}

GLModel::~GLModel() noexcept
{
	this->destroy();
}

// GLModel: Methods
// ------------------------------------------------------------------------------------------------

void GLModel::load(const RawMesh& mesh) noexcept
{
	if (mVAO != 0) this->destroy();

	// Create Vertex Array object
	glGenVertexArrays(1, &mVAO);
	glBindVertexArray(mVAO);

	// Create and fill vertex buffer
	glGenBuffers(1, &mVertexBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, mVertexBuffer);
	glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(mesh.vertices.size() * sizeof(Vertex)),
	             mesh.vertices.data(), GL_STATIC_DRAW);

	// Locate components in vertex buffer
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, pos));
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, normal));
	glEnableVertexAttribArray(2);
	glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, uv));

	// Create and fill index buffer
	glGenBuffers(1, &mIndexBuffer);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mIndexBuffer);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, static_cast<GLsizeiptr>(sizeof(uint32_t) * mesh.indices.size()),
	             mesh.indices.data(), GL_STATIC_DRAW);
	mNumIndices = mesh.indices.size();

	// Cleanup
	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

void GLModel::destroy() noexcept
{
	// Silently ignores values == 0
	glDeleteBuffers(1, &mVertexBuffer);
	glDeleteBuffers(1, &mIndexBuffer);
	glDeleteVertexArrays(1, &mVAO);
	mVertexBuffer = 0;
	mIndexBuffer = 0;
	mVAO = 0;
	mNumIndices = 0;
}

void GLModel::swap(GLModel& other) noexcept
{
	std::swap(this->mVAO, other.mVAO);
	std::swap(this->mVertexBuffer, other.mVertexBuffer);
	std::swap(this->mIndexBuffer, other.mIndexBuffer);
	std::swap(this->mNumIndices, other.mNumIndices);
}

void GLModel::draw() const noexcept
{
	glBindVertexArray(mVAO);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mIndexBuffer);
	glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(mNumIndices), GL_UNSIGNED_INT, 0);
}

} // namespace phe
