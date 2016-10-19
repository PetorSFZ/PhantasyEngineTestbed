// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include <sfz/containers/DynArray.hpp>
#include <sfz/gl/Framebuffer.hpp>
#include <sfz/math/Matrix.hpp>
#include <sfz/memory/SmartPointers.hpp>

#include "phantasy_engine/level/DynObject.hpp"
#include "phantasy_engine/level/StaticScene.hpp"
#include "phantasy_engine/level/SphereLight.hpp"
#include "phantasy_engine/rendering/Material.hpp"
#include "phantasy_engine/rendering/RawImage.hpp"
#include "phantasy_engine/rendering/ViewFrustum.hpp"

namespace phe {

using sfz::gl::Framebuffer;
using sfz::mat4;
using sfz::SharedPtr;
using sfz::vec2i;
using sfz::vec3;
using sfz::identityMatrix4;

// Helper structs
// ------------------------------------------------------------------------------------------------

struct RenderResult final {
	vec2i renderedRes = vec2i(0);
	// Should be a texture with at least three channels, the main output image of the renderer.
	uint32_t colorTexture = 0;
	// Must be a depth texture with internal format GL_DEPTH_COMPONENT32, or 0 if not available.
	uint32_t depthTexture = 0;
	// Should be a texture with roughness in the first channel and metallic the second, or 0 if not
	// available.
	uint32_t materialTexture = 0;
	// Should be a texture with at least three channels containing the world space velocity of the
	// object at each pixel, or 0 if not available.
	uint32_t velocityTexture = 0;
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
	
	virtual void setMaterialsAndTextures(const DynArray<Material>& materials,
	                                     const DynArray<RawImage>& textures) noexcept = 0;

	virtual void addTexture(const RawImage& texture) noexcept = 0;

	virtual void addMaterial(const Material& material) noexcept = 0;

	virtual void setStaticScene(const StaticScene& staticScene) noexcept = 0;
	
	virtual void setDynamicMeshes(const DynArray<RawMesh>& meshes) noexcept = 0;

	virtual void addDynamicMesh(const RawMesh& mesh) noexcept = 0;

	virtual RenderResult render(const DynArray<DynObject>& objects,
	                            const DynArray<SphereLight>& lights) noexcept = 0;

	// Non-virtual methods
	// --------------------------------------------------------------------------------------------

	inline void updateCamera(const ViewFrustum& camera) noexcept { mCamera = camera; }

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

	ViewFrustum mCamera;
	vec2i mTargetResolution = vec2i(0, 0);
};

} // namespace phe
