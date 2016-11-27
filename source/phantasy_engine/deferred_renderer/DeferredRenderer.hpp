// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include "phantasy_engine/rendering/BaseRenderer.hpp"

namespace phe {

// DeferredRenderer
// ------------------------------------------------------------------------------------------------

class DeferredRendererImpl; // Pimpl pattern

class DeferredRenderer final : public BaseRenderer {
public:

	// Constructors & destructors
	// --------------------------------------------------------------------------------------------

	DeferredRenderer(const DeferredRenderer&) = delete;
	DeferredRenderer& operator= (const DeferredRenderer&) = delete;
	
	DeferredRenderer() noexcept;
	DeferredRenderer(DeferredRenderer&& other) noexcept;
	DeferredRenderer& operator= (DeferredRenderer&& other) noexcept;
	~DeferredRenderer() noexcept;

	// Virtual methods from BaseRenderer interface
	// --------------------------------------------------------------------------------------------

	void setMaterialsAndTextures(const DynArray<Material>& materials,
	                             const DynArray<RawImage>& textures) noexcept override final;

	void addTexture(const RawImage& texture) noexcept override final;

	void addMaterial(const Material& material) noexcept override final;

	void setStaticScene(const StaticScene& staticScene) noexcept override final;
	
	void setDynamicMeshes(const DynArray<RawMesh>& meshes) noexcept override final;

	void addDynamicMesh(const RawMesh& mesh) noexcept override final;

	RenderResult render(const RenderComponent* renderComponents, uint32_t numComponents,
	                    const DynArray<SphereLight>& lights) noexcept override final;

protected:
	// Protected virtual methods from BaseRenderer interface
	// --------------------------------------------------------------------------------------------

	void targetResolutionUpdated() noexcept override final;

private:
	// Private members
	// --------------------------------------------------------------------------------------------
	
	DeferredRendererImpl* mImpl = nullptr;
};

} // namespace phe
