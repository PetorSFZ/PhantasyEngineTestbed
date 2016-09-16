// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include <thread>

#include <sfz/containers/DynArray.hpp>
#include <sfz/math/Vector.hpp>

#include "phantasy_engine/rendering/BaseRenderer.hpp"
#include "phantasy_engine/ray_tracer_common/BVH.hpp"
#include "phantasy_engine/rendering/RawImage.hpp"

namespace phe {

using sfz::DynArray;
using sfz::gl::Framebuffer;
using sfz::vec3;
using sfz::vec4;

// Forward declarations
// ------------------------------------------------------------------------------------------------

struct HitInfo;
struct RayCastResult;
struct Ray;

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

	const uint8_t* sampleImage(const RawImage& image, const vec2 uv) const noexcept;

	vec4 shadeHit(const Ray& ray, const RayCastResult& hit, const HitInfo& info) noexcept;

	// Private members
	// --------------------------------------------------------------------------------------------

	DynArray<vec4> mTexture;
	DynArray<std::thread> mThreads;
	BVH mBVH;
};

} // namespace phe
