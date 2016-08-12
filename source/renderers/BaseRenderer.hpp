// Copyright (c) Peter Hillerström (skipifzero.com, peter@hstroem.se)

#pragma once

#include <sfz/containers/DynArray.hpp>
#include <sfz/gl/Framebuffer.hpp>
#include <sfz/math/Matrix.hpp>

#include "resources/Renderable.hpp"

namespace sfz {

using gl::Framebuffer;

// Helper structs
// ------------------------------------------------------------------------------------------------

/// Struct holding all the camera matrices needed for rendering
///
/// originMatrix: the matrix defining the space the camera's location is defined in, in particular
/// this is the conversion from world space to room space in a vr context.
/// headMatrix: the conversion from room space to head space (or view space in a non-VR context).
/// eyeMatrixVR[eye]: the conversion from head space to eye space (only used for VR)
///
/// This means that in a non-VR context the complete viewMatrix is:
/// viewMatrix = headMatrix * originMatrix
///
/// When performing vr rendering we also need to take into account the eye's locations relative
/// to the head, so the complete viewMatrix for a given eye would be:
/// viewMatrix[eye] = eyeMatrixVR[eye] * headMatrix * originMatrix
struct CameraMatrices final {
	// Shared matrices for both non-vr and vr rendering
	mat4 originMatrix = identityMatrix4<float>();
	mat4 headMatrix = identityMatrix4<float>();

	// Non-vr projection matrices
	mat4 projMatrix = identityMatrix4<float>();

	// VR only matrices
	mat4 eyeMatrixVR[2] = { identityMatrix4<float>(), identityMatrix4<float>() };
	mat4 projMatrixVR[2] = { identityMatrix4<float>(), identityMatrix4<float>() };
};

struct DrawOp final {
	mat4 transform = identityMatrix4<float>();
	const Renderable* renderablePtr = nullptr;

	DrawOp() noexcept = default;
	inline DrawOp(const mat4& transform, const Renderable* renderablePtr) noexcept
	:
		transform(transform),
		renderablePtr(renderablePtr)
	{ }
};

// BaseRenderer
// ------------------------------------------------------------------------------------------------

class BaseRenderer {
public:
	// Virtual methods
	// --------------------------------------------------------------------------------------------
	
	virtual void render(const DynArray<DrawOp>& operations) noexcept = 0;

	/// The resulting framebuffer after rendering
	virtual const Framebuffer& getResult() const noexcept = 0;
	
	/// The resulting framebuffer after rendering for a given eye
	virtual const Framebuffer& getResultVR(uint32_t eye) const noexcept = 0;

	// Non-virtual methods
	// --------------------------------------------------------------------------------------------

	inline void updateMatrices(const CameraMatrices& matrices) noexcept { mMatrices = matrices; }

	inline vec2i maxResolution() const noexcept { return mMaxResolution; }
	inline void setMaxResolution(vec2i maxResolution) noexcept
	{
		sfz_assert_debug(0 < maxResolution.x);
		sfz_assert_debug(0 < maxResolution.y);
		this->mMaxResolution = maxResolution;
		this->maxResolutionUpdated();
	}

	inline vec2i resolution() const noexcept { return mResolution; }
	inline void setResolution(vec2i resolution) noexcept
	{
		sfz_assert_debug(resolution.x <= mMaxResolution.x);
		sfz_assert_debug(resolution.y <= mMaxResolution.y);
		this->mResolution = resolution;
		this->resolutionUpdated();
	}

protected:
	// Protected virtual methods
	// --------------------------------------------------------------------------------------------

	virtual void maxResolutionUpdated() noexcept = 0;
	virtual void resolutionUpdated() noexcept = 0;

	// Protected members
	// --------------------------------------------------------------------------------------------

	CameraMatrices mMatrices;
	vec2i mMaxResolution, mResolution;
};

} // namespace sfz
