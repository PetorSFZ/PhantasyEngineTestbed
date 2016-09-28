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

	void setMaterialsAndTextures(const DynArray<Material>& materials,
	                             const DynArray<RawImage>& textures) noexcept override final;

	void addTexture(const RawImage& texture) noexcept override final;

	void addMaterial(const Material& material) noexcept override final;

	void setStaticScene(const StaticScene& staticScene) noexcept override final;
	
	void setDynamicMeshes(const DynArray<RawMesh>& meshes) noexcept override final;

	void addDynamicMesh(const RawMesh& mesh) noexcept override final;

	RenderResult render(Framebuffer& resultFB,
	                    const DynArray<DynObject>& objects,
	                    const DynArray<SphereLight>& lights) noexcept override final;

	void sendDynamicBvhToCuda();
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
