// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include <sfz/CUDACallable.hpp>
#include <sfz/math/Vector.hpp>

namespace phe {

using sfz::vec3;

// BVHNode
// ------------------------------------------------------------------------------------------------

// Based on the BVH created for https://www.thanassis.space/cudarenderer-BVH.html
// https://github.com/ttsiodras/renderer-cuda/blob/master/src/BVH.h

struct BVHNode final {

	// Members
	// --------------------------------------------------------------------------------------------

	// AABB
	vec3 min;
	vec3 max;
	
	// The first bit (most significant bit in indices[0]) is set if this node is a leaf.
	// This node is not a leaf: indices[0] contains index of left child, indices[1] of right
	// This node is a leaf: indices[0] (after masking away msb) contains number of triangles,
	// indices[1] is index to first triangle in list
	uint32_t indices[2];

	// Getters
	// --------------------------------------------------------------------------------------------

	SFZ_CUDA_CALLABLE uint32_t leftChildIndex() const noexcept
	{
		return this->indices[0];
	}

	SFZ_CUDA_CALLABLE uint32_t rightChildIndex() const noexcept
	{
		return this->indices[1];
	}

	SFZ_CUDA_CALLABLE bool isLeaf() const noexcept
	{
		uint32_t val = this->indices[0];
		val &= 0x80000000u;
		return val != 0u;
	}

	SFZ_CUDA_CALLABLE uint32_t numTriangles() const noexcept
	{
		uint32_t val = this->indices[0];
		val &= 0x7FFFFFFFu;
		return val;
	}

	SFZ_CUDA_CALLABLE uint32_t triangleListIndex() const noexcept
	{
		return this->indices[1];
	}

	// Setters
	// --------------------------------------------------------------------------------------------

	SFZ_CUDA_CALLABLE void setInner(uint32_t leftChildIndex, uint32_t rightChildIndex) noexcept
	{
		this->indices[0] = leftChildIndex;
		this->indices[1] = rightChildIndex;
	}

	SFZ_CUDA_CALLABLE void setLeaf(uint32_t numTriangles, uint32_t triangleListIndex) noexcept
	{
		this->indices[0] = numTriangles | 0x80000000u;
		this->indices[1] = triangleListIndex;
	}
};

static_assert(sizeof(BVHNode) == 32, "BVHNode is padded");

} // namespace phe
