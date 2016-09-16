// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include <cstdint>

#include <sfz/containers/DynArray.hpp>
#include <sfz/math/Vector.hpp>

namespace phe {

using sfz::DynArray;
using sfz::vec2;
using sfz::vec3;
using std::uint16_t;
using std::uint32_t;

// Vertex struct
// ------------------------------------------------------------------------------------------------

struct Vertex final {
	vec3 pos = vec3(0.0);
	vec3 normal = vec3(0.0);
	vec2 uv = vec2(0.0);
};

static_assert(sizeof(Vertex) == sizeof(float) * 8, "Vertex is padded");

// RawMesh struct
// ------------------------------------------------------------------------------------------------

struct RawMesh final {
	DynArray<Vertex> vertices;
	DynArray<uint16_t> materialIndices;
	DynArray<uint32_t> indices;
};

} // namespace phe
