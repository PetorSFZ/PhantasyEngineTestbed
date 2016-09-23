// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include "phantasy_engine/rendering/BaseRenderer.hpp"

namespace phe {

// CudaTracerRenderer
// ------------------------------------------------------------------------------------------------

class CudaTracerRendererImpl; // Pimpl pattern

class CudaTracerRenderer final : public BaseRenderer {
public:

	// Constructors & destructors
	// --------------------------------------------------------------------------------------------

	CudaTracerRenderer(const CudaTracerRenderer&) = delete;
	CudaTracerRenderer& operator= (const CudaTracerRenderer&) = delete;
	CudaTracerRenderer(CudaTracerRenderer&&) = delete;
	CudaTracerRenderer& operator= (CudaTracerRenderer&&) = delete;
	
	CudaTracerRenderer() noexcept;
	~CudaTracerRenderer() noexcept;

	// Virtual methods from BaseRenderer interface
	// --------------------------------------------------------------------------------------------

	void bakeMaterials(const DynArray<RawImage>& textures,
	                   const DynArray<Material>& materials) noexcept override final;

	void addMaterial(RawImage& texture, Material& material) noexcept override final;

	void bakeStaticScene(const StaticScene& staticScene) noexcept override final;

	void setDynObjectsForRendering(const DynArray<RawMesh>& meshes, const DynArray<mat4>& transforms) noexcept override final;

	RenderResult render(Framebuffer& resultFB) noexcept override final;

protected:
	// Protected virtual methods from BaseRenderer interface
	// --------------------------------------------------------------------------------------------

	void targetResolutionUpdated() noexcept override final;

private:
	// Private members
	// --------------------------------------------------------------------------------------------
	
	CudaTracerRendererImpl* mImpl = nullptr;
};

} // namespace phe
