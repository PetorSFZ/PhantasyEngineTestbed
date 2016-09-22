// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include "phantasy_engine/ray_tracer_common/BVH.hpp"

namespace phe {

// Fixes a few potential issues with the BVH, including:
// * (Potentially) improves cache locality
// * Ensures last triangle in a leaf has the necessary end marker in the padding
// * Ensures all leaf indices are bitwise negated (~)
void sanitizeBVH(BVH& bvh) noexcept;

} // namespace phe
