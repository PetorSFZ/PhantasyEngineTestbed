#pragma once

#include <sfz/containers/DynArray.hpp>
#include <sfz/geometry/AABB.hpp>
#include <sfz/math/Vector.hpp>

#include "resources/Renderable.hpp"

namespace sfz {

struct TriangleIntersection {
	bool intersected;
	float t, u, v;
};

struct Triangle {
	union {
		struct { vec3 p0, p1, p2; };
		vec3 vertices[3];
	};
};

struct RaycastResult {
	const Triangle* triangle;
	TriangleIntersection intersection;
};

struct BvhNode {
	uint32_t left = ~0;
	uint32_t right = ~0;
	AABB aabb;
	uint32_t triangleInd = ~0;

	inline bool isLeaf() const
	{
		return left == uint32_t(~0);
	}

	inline bool isEmpty() const
	{
		return isLeaf() && triangleInd == uint32_t(~0);
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
