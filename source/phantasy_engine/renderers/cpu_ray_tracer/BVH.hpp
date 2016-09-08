// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#include <cstdint>

#include <sfz/containers/DynArray.hpp>
#include <sfz/math/Vector.hpp>

#include "phantasy_engine/level/StaticScene.hpp"

#pragma once

namespace phe {

using sfz::DynArray;
using sfz::vec3;

// BVHNode
// ------------------------------------------------------------------------------------------------

// Based on the BVH created for https://www.thanassis.space/cudarenderer-BVH.html
// https://github.com/ttsiodras/renderer-cuda/blob/master/src/BVH.h

struct BVHNode final {
	// AABB
	float min[3];
	float max[3];
	
	// The first bit (most significant bit in indices[0]) is set if this node is a leaf.
	// This node is not a leaf: indices[0] contains index of left child, indices[1] of right
	// This node is a leaf: indices[0] (after masking away msb) contains number of triangles,
	// indices[1] is index to first triangle in list
	uint32_t indices[2];
};

static_assert(sizeof(BVHNode) == 32, "BVHNode is padded");

// C++ container
// ------------------------------------------------------------------------------------------------

class BVH final {
public:
	DynArray<BVHNode> nodes;
	//DynArray<Triangles>
};

// TODO: implement
BVH buildBVHFromStaticScene(const StaticScene& scene) noexcept;

// C++ getters
// ------------------------------------------------------------------------------------------------

inline vec3 aabbMin(const BVHNode& node) noexcept
{
	return vec3(node.min);
}

inline vec3 aabbMax(const BVHNode& node) noexcept
{
	return vec3(node.max);
}

inline bool isLeaf(const BVHNode& node) noexcept
{
	uint32_t val = node.indices[0];
	val = val >> 31u;
	val &= 1u;
	return val == 1u;
}

inline uint32_t leftChildIndex(const BVHNode& node) noexcept
{
	return node.indices[0];
}

inline uint32_t rightChildIndex(const BVHNode& node) noexcept
{
	return node.indices[1];
}

inline uint32_t numTriangles(const BVHNode& node) noexcept
{
	uint32_t val = node.indices[0];
	val &= 0x7FFFFFFFu;
	return val;
}

inline uint32_t triangleListIndex(const BVHNode& node) noexcept
{
	return node.indices[1];
}

// C++ setters
// ------------------------------------------------------------------------------------------------

inline void setAABB(BVHNode& node, const vec3& min, const vec3& max) noexcept
{
	node.min[0] = min.x;
	node.min[1] = min.y;
	node.min[2] = min.z;
	node.max[0] = max.x;
	node.max[1] = max.y;
	node.max[2] = max.z;
}

inline void setInner(BVHNode& node, uint32_t leftChildIndex, uint32_t rightChildIndex) noexcept
{
	node.indices[0] = leftChildIndex;
	node.indices[1] = rightChildIndex;
}

inline void setLeaf(BVHNode& node, uint32_t numTriangles, uint32_t triangleListIndex) noexcept
{
	node.indices[0] = numTriangles | 0x80000000u;
	node.indices[1] = triangleListIndex;
}

} // namespace phe
