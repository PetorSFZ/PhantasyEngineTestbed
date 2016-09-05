// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include <thread>

#include <sfz/containers/DynArray.hpp>
#include <sfz/math/Vector.hpp>

#include "phantasy_engine/renderers/BaseRenderer.hpp"
#include "phantasy_engine/renderers/cpu_ray_tracer/AabbTree.hpp"

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

	RenderResult render(Framebuffer& resultFB) noexcept override final;

protected:
	// Protected virtual methods from BaseRenderer interface
	// --------------------------------------------------------------------------------------------

	void staticSceneChanged() noexcept override final;

	void targetResolutionUpdated() noexcept override final;

private:
	// Private methods
	// --------------------------------------------------------------------------------------------

	vec4 tracePrimaryRays(vec3 origin, vec3 dir) const noexcept;

	// Private members
	// --------------------------------------------------------------------------------------------

	DynArray<vec4> mTexture;
	DynArray<std::thread> mThreads;
	AabbTree aabbBvh;
};

} // namespace sfz
