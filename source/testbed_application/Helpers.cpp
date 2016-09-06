// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#include "Helpers.hpp"

#include <sfz/memory/New.hpp>

#include <phantasy_engine/Config.hpp>

using sfz::sfz_new;

UniquePtr<BaseRenderer> createRendererBasedOnConfig() noexcept
{
	auto& cfg = phe::GlobalConfig::instance();

	phe::Setting* renderingBackendSetting = cfg.getSetting("PhantasyEngineTestbed", "renderingBackend");

	UniquePtr<BaseRenderer> renderer;
	switch (renderingBackendSetting->intValue()) {
	default:
		printf("%s\n", "Something is wrong with the config. Falling back to deferred rendering.");
	case 0:
		renderer = UniquePtr<BaseRenderer>(sfz_new<phe::DeferredRenderer>());
		break;
	case 1:
#ifdef CUDA_TRACER_AVAILABLE
		renderer = UniquePtr<BaseRenderer>(sfz_new<phe::CUDARayTracerRenderer>());
#else
		printf("%s\n", "CUDA not available in this build, using deferred renderer instead.");
		renderer = UniquePtr<BaseRenderer>(sfz_new<phe::DeferredRenderer>());
#endif
		break;
	case 2:
		renderer = UniquePtr<BaseRenderer>(sfz_new<phe::CPURayTracerRenderer>());
		break;
	return renderer;
	}
}
