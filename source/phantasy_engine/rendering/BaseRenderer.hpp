// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include <sfz/containers/DynArray.hpp>
#include <sfz/gl/Framebuffer.hpp>
#include <sfz/math/Matrix.hpp>
#include <sfz/memory/SmartPointers.hpp>

#include "phantasy_engine/level/StaticScene.hpp"
#include "phantasy_engine/level/SphereLight.hpp"
#include "phantasy_engine/rendering/Material.hpp"
#include "phantasy_engine/rendering/RawImage.hpp"

namespace phe {

using sfz::gl::Framebuffer;
using sfz::mat4;
using sfz::SharedPtr;
using sfz::vec2i;
using sfz::vec3;
using sfz::identityMatrix4;

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

	vec3 position{ 0.0f };
	vec3 forward{ 0.0f };
	vec3 up{ 0.0f };
	float vertFovRad;
};

struct RenderResult final {
	vec2i renderedRes = vec2i(0);
};

// BaseRenderer
// ------------------------------------------------------------------------------------------------

class BaseRenderer {
public:
	// Constructors & destructors
	// --------------------------------------------------------------------------------------------

	virtual ~BaseRenderer() noexcept { }

	// Virtual methods
	// --------------------------------------------------------------------------------------------
	
	virtual void bakeMaterials(const DynArray<RawImage>& textures,
	                           const DynArray<Material>& materials) noexcept = 0;

	virtual void addMaterial(RawImage& texture, Material& material) noexcept = 0;

	virtual void bakeStaticScene(const StaticScene& staticScene) noexcept = 0;

	/// The resultFB framebuffer is required to have a color texture (rgba 16bit float) and a
	/// depth texture of type 32bit float, the resolution of the framebuffer must be the same
	/// as the target resolution of the renderer.
	virtual RenderResult render(Framebuffer& resultFB) noexcept = 0;

	// Non-virtual methods
	// --------------------------------------------------------------------------------------------

	inline void updateMatrices(const CameraMatrices& matrices) noexcept { mMatrices = matrices; }

	inline vec2i targetResolution() const noexcept { return mTargetResolution; }
	inline void setTargetResolution(vec2i targetResolution) noexcept
	{
		sfz_assert_debug(0 < targetResolution.x);
		sfz_assert_debug(0 < targetResolution.y);
		this->mTargetResolution = targetResolution;
		this->targetResolutionUpdated();
	}

protected:
	// Protected virtual methods
	// --------------------------------------------------------------------------------------------

	virtual void targetResolutionUpdated() noexcept = 0;

	// Protected members
	// --------------------------------------------------------------------------------------------

	CameraMatrices mMatrices;
	vec2i mTargetResolution = vec2i(0, 0);
};

} // namespace phe
