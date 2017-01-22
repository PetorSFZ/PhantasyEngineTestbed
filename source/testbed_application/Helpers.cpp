// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#include "Helpers.hpp"

#include <sfz/memory/New.hpp>

#include <phantasy_engine/Config.hpp>
#include <phantasy_engine/DeferredRenderer.hpp>
#ifdef CUDA_TRACER_AVAILABLE
#include <CudaTracerRenderer.hpp>
#endif

using namespace sfz;

DynArray<RendererAndStatus> createRenderers(uint32_t& indexSelected) noexcept
{
	DynArray<RendererAndStatus> renderers;
	renderers.ensureCapacity(16);

	// 0 == Deferred renderer
	renderers.add({makeSharedDefault<phe::DeferredRenderer>().cast<phe::BaseRenderer>(), false});

	// 1 == CUDA ray tracer
#ifdef CUDA_TRACER_AVAILABLE
	renderers.add({makeSharedDefault<phe::CudaTracerRenderer>().cast<phe::BaseRenderer>(), false});
#else
	printf("%s\n", "CUDA not available in this build, using deferred renderer instead.");
	renderers.add({renderers.first().renderer, false});
#endif

	// Retrieving index of selected renderer
	auto& cfg = phe::GlobalConfig::instance();
	phe::Setting* renderingBackendSetting = cfg.getSetting("PhantasyEngineTestbed", "renderingBackend");
	indexSelected = renderingBackendSetting->intValue();
	if (indexSelected >= renderers.size()) {
		sfz::error("Renderer index out of bounds, probably need to add rendering backend to this function.");
	}

	// Selected renderer will be baked in constructor, set baked=true
	renderers[indexSelected].baked = true;

	return renderers;
}
