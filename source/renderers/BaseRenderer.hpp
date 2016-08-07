// Copyright (c) Peter Hillerström (skipifzero.com, peter@hstroem.se)

#pragma once

#include <sfz/gl/Framebuffer.hpp>
#include <sfz/math/Matrix.hpp>

namespace sfz {

using gl::Framebuffer;

// BaseRenderer
// ------------------------------------------------------------------------------------------------

/// Interface for all Phantasy Engine renderers
class BaseRenderer {
public:
	/// TODO: Temporary rendering function, will be removed in future when requirements are better
	/// specified
	virtual void render(const mat4& viewMatrix, const mat4& projMatrix) noexcept = 0;

	/// The resulting framebuffer after rendering
	virtual const Framebuffer& getResult() const noexcept = 0;
	
	/// The resulting framebuffer after rendering for a given eye
	virtual const Framebuffer& getResultVR(uint32_t eye) const noexcept = 0;
};

} // namespace sfz
