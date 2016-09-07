// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include <sfz/containers/DynArray.hpp>
#include <sfz/memory/SmartPointers.hpp>

#include <phantasy_engine/Renderers.hpp>
#ifdef CUDA_TRACER_AVAILABLE
#include <CudaRayTracerRenderer.hpp>
#endif

using sfz::DynArray;
using sfz::SharedPtr;
using phe::BaseRenderer;

struct RendererAndStatus final {
	SharedPtr<BaseRenderer> renderer;
	bool baked;
};

DynArray<RendererAndStatus> createRenderers(uint32_t& indexSelected) noexcept;
