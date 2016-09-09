// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include <cstdint>

#include <sfz/containers/DynArray.hpp>
#include <sfz/geometry/AABB.hpp>
#include <sfz/math/Vector.hpp>

#include "phantasy_engine/level/StaticScene.hpp"

#include "phantasy_engine/ray_tracer_common/BVHNode.hpp"

#pragma once

namespace phe {

using sfz::DynArray;
using sfz::vec3;

// Triangle
// ------------------------------------------------------------------------------------------------

struct TriangleVertices final {
	float v0[3];
	float v1[3];
	float v2[3];
};

static_assert(sizeof(TriangleVertices) == 36, "TrianglePosition is padded");

struct TriangleData final {
	float n0[3];
	float n1[3];
	float n2[3];

	float uv0[2];
	float uv1[2];
	float uv2[2];

	uint32_t materialIndex;
};

static_assert(sizeof(TriangleData) == 64, "TriangleData is padded");

struct TriangleUnused final {
	float po[3];
	float p1[3];
	float p2[3];

	float n0[3];
	float n1[3];
	float n2[3];

	float uv0[2];
	float uv1[2];
	float uv2[2];

	float albedoValue[3];
	uint32_t albedoTexIndex;
	float roughness;
	uint32_t roughnessTexIndex;
	float metallic;
	uint32_t metallicTexIndex;
};

static_assert(sizeof(TriangleUnused) == 128, "TriangleUnused is padded");

// C++ container
// ------------------------------------------------------------------------------------------------

class BVH final {
public:

	// Members
	// --------------------------------------------------------------------------------------------

	DynArray<BVHNode> nodes;

	// These arrays are supposed to be the same size, an index is valid in both lists
	DynArray<TriangleVertices> triangles;
	DynArray<TriangleData> triangleDatas;

	// Methods
	// --------------------------------------------------------------------------------------------

	void buildStaticFrom(const StaticScene& scene) noexcept;
	void buildStaticFrom(const DynArray<TriangleVertices>& triangles) noexcept;

private:

	// Private methods
	// --------------------------------------------------------------------------------------------

	void fillStaticNode(
		uint32_t nodeInd,
		uint32_t depth,
		const DynArray<uint32_t>& triangleInds,
		const DynArray<TriangleVertices>& inTriangles,
		const DynArray<sfz::AABB>& inTriangleAabbs) noexcept;

};

// C++ setters
// ------------------------------------------------------------------------------------------------

inline void setFloat3(float arr[], const vec3& vec) noexcept
{
	arr[0] = vec.x;
	arr[1] = vec.y;
	arr[2] = vec.z;
}

inline void setTriangle(TriangleVertices& triangle, const vec3& p0, const vec3& p1, const vec3& p2) noexcept
{
	setFloat3(triangle.v0, p0);
	setFloat3(triangle.v1, p1);
	setFloat3(triangle.v2, p2);
}

} // namespace phe
