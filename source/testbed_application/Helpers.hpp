// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include <sfz/memory/SmartPointers.hpp>

#include <phantasy_engine/Renderers.hpp>
#ifdef CUDA_TRACER_AVAILABLE
#include <CudaRayTracerRenderer.hpp>
#endif

using sfz::UniquePtr;
using phe::BaseRenderer;

UniquePtr<BaseRenderer> createRendererBasedOnConfig() noexcept;
