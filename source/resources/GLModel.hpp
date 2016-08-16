// Copyright (c) Peter Hillerstr�m (skipifzero.com, peter@hstroem.se)

#pragma once

#include <cstdint>

#include "resources/RawGeometry.hpp"

namespace sfz {

using std::uint32_t;

// GLModel class
// ------------------------------------------------------------------------------------------------

class GLModel final {
public:
	// Constructors & destructors
	// --------------------------------------------------------------------------------------------

	GLModel() noexcept = default;
	GLModel(const GLModel&) = delete;
	GLModel& operator= (const GLModel&) = delete;

	GLModel(const RawGeometry& geometry) noexcept;
	GLModel(GLModel&& other) noexcept;
	GLModel& operator= (GLModel&& other) noexcept;
	~GLModel() noexcept;

	// Methods
	// --------------------------------------------------------------------------------------------

	
	void load(const RawGeometry& geometry) noexcept;

	/// Destroys this GLModel
	void destroy() noexcept;

	/// Swaps this model with another model
	void swap(GLModel& other) noexcept;

	/// Binds the indices and draws the vertices through glDrawElements()
	void draw() const noexcept;

private:
	// Private members
	// --------------------------------------------------------------------------------------------

	uint32_t mVAO = 0;
	uint32_t mVertexBuffer = 0;
	uint32_t mIndexBuffer = 0;
	uint32_t mNumIndices = 0;
};

} // namespace sfz