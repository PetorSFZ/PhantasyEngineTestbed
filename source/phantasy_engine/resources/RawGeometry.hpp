// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include <cstdint>

#include <sfz/containers/DynArray.hpp>
#include <sfz/math/Vector.hpp>

namespace sfz {

using std::uint32_t;

// Vertex struct
// ------------------------------------------------------------------------------------------------

struct Vertex final {
	vec3 pos = vec3(0.0);
	vec3 normal = vec3(0.0);
	vec2 uv = vec2(0.0);
};

static_assert(sizeof(Vertex) == sizeof(float) * 8, "Vertex is padded");

// RawGeometry struct
// ------------------------------------------------------------------------------------------------

struct RawGeometry final {
	DynArray<Vertex> vertices;
	DynArray<uint32_t> indices;
};

} // namespace sfz
