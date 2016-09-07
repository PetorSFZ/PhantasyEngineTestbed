// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#include "Helpers.hpp"

#include <sfz/memory/New.hpp>

#include <phantasy_engine/Config.hpp>

using namespace sfz;

DynArray<RendererAndStatus> createRenderers(uint32_t& indexSelected) noexcept
{
	DynArray<RendererAndStatus> renderers;
	renderers.ensureCapacity(16);

	// 0 == Deferred renderer
	renderers.add({SharedPtr<BaseRenderer>(sfz_new<phe::DeferredRenderer>()), false});

	// 1 == CUDA ray tracer
#ifdef CUDA_TRACER_AVAILABLE
	renderers.add({SharedPtr<BaseRenderer>(sfz_new<phe::CUDARayTracerRenderer>()), false});
#else
	printf("%s\n", "CUDA not available in this build, using deferred renderer instead.");
	renderers.add({renderers.first().renderer, false});
#endif

	// 2 == CPU ray tracer
	renderers.add({SharedPtr<BaseRenderer>(sfz_new<phe::CPURayTracerRenderer>()), false});

	// Retrieving index of selected renderer
	auto& cfg = phe::GlobalConfig::instance();
	phe::Setting* renderingBackendSetting = cfg.getSetting("PhantasyEngineTestbed", "renderingBackend");
	indexSelected = renderingBackendSetting->intValue();
	if (indexSelected >= renderers.size()) {
		sfz::error("Renderer index out of bounds, probably need to add rendering backend to this function.");
	}

	return renderers;
}
