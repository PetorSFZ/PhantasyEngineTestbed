#pragma once

#include <memory>

#include <sfz/math/Vector.hpp>

#include "renderers/BaseRenderer.hpp"
#include "AabbTree.hpp"

namespace sfz {

// CPURayTracerRenderer
// ------------------------------------------------------------------------------------------------

class CPURayTracerRenderer final : public BaseRenderer {
public:

	// Constructors & destructors
	// --------------------------------------------------------------------------------------------

	CPURayTracerRenderer(const CPURayTracerRenderer&) = delete;
	CPURayTracerRenderer& operator= (const CPURayTracerRenderer&) = delete;
	CPURayTracerRenderer(CPURayTracerRenderer&&) noexcept = default;
	CPURayTracerRenderer& operator= (CPURayTracerRenderer&&) noexcept = default;

	CPURayTracerRenderer() noexcept;

	// Virtual methods from BaseRenderer interface
	// --------------------------------------------------------------------------------------------

	RenderResult render(const DynArray<DrawOp>& operations,
	                    const DynArray<PointLight>& pointLights) noexcept override final;
	void prepareForScene(const Scene& scene) noexcept override final;

protected:
	// Protected virtual methods from BaseRenderer interface
	// --------------------------------------------------------------------------------------------

	void targetResolutionUpdated() noexcept override final;

private:

	vec4 tracePrimaryRays(vec3 origin, vec3 dir) const noexcept;

	// Private members
	// --------------------------------------------------------------------------------------------
	
	Framebuffer mResult;
	std::unique_ptr<vec4[]> mTexture;
	AabbTree aabbBvh;
};

} // namespace sfz
