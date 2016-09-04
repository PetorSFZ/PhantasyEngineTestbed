// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include <sfz/containers/DynArray.hpp>
#include <sfz/geometry/AABB.hpp>
#include <sfz/math/Vector.hpp>

#include "phantasy_engine/resources/Renderable.hpp"

namespace sfz {

struct TriangleIntersection {
	bool intersected;
	float t, u, v;
};

struct Triangle {
	vec3 p0, p1, p2;
};

struct RaycastResult {
	const Triangle* triangle;
	TriangleIntersection intersection;
};

struct BvhNode {
	uint32_t left = UINT32_MAX;
	uint32_t right = UINT32_MAX;
	AABB aabb;
	uint32_t triangleInd = UINT32_MAX;

	inline bool isLeaf() const
	{
		return left == UINT32_MAX;
	}

	inline bool isEmpty() const
	{
		return isLeaf() && triangleInd == UINT32_MAX;
	}
};

class AabbTree {

public:
	AabbTree() noexcept = default;

	void constructFrom(const DynArray<Renderable>& renderables) noexcept;

	/// Find the closest triangle that intersects with the ray, making use of the constructed BVH.
	RaycastResult raycast(vec3 origin, vec3 direction) const noexcept;

private:
	AABB aabbFromUseExisting(const DynArray<uint32_t>& triangleInds) noexcept;
	AABB aabbFrom(uint32_t triangleInd) noexcept;
	void fillNode(uint32_t nodeInd, DynArray<BvhNode>& nodes, const DynArray<uint32_t>& triangleInds, uint32_t depth) noexcept;

	DynArray<Triangle> triangles;
	DynArray<AABB> triangleAabbs;
	DynArray<BvhNode> nodes;
};

} // namespace sfz
