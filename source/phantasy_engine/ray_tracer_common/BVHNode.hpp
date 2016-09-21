// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include <sfz/CUDACallable.hpp>
#include <sfz/math/Vector.hpp>

namespace phe {

using sfz::vec3;
using sfz::vec4;
using sfz::vec4u;

// BVHNode
// ------------------------------------------------------------------------------------------------

struct BVHNode {
	
	// Data (You are not meant to access these directly, use the getters and setters!)
	// --------------------------------------------------------------------------------------------

	// fData contains the following (lc == left child, rc == right child):
	// [lc.aabbMin.x, lc.aabbMin.y, lc.aabbMin.z, lc.aabbMax.x]
	// [lc.aabbMax.y, lc.aabbMax.z, rc.aabbMin.x, rc.aabbMin.y]
	// [rc.aabbMin.z, rc.aabbMax.x, rc.aabbMax.y, rc.aabbMax.z]
	vec4 fData[3];

	// iData contains the following:
	// [lcIndex, rcIndex, lcNumTriangles, rcNumTriangles]
	// In an inner node lcIndex and rcIndex are the indices of the children inner nodes. In a
	// leaf node they are indices to the first triangles.
	// lcNumTriangles and rcNumTriangles are 0 in inner nodes, in leaf nodes they tell the number
	// of triangles pointed to by the leaf node 
	vec4u iData;

	// Getters
	// --------------------------------------------------------------------------------------------

	SFZ_CUDA_CALLABLE vec3 leftChildAABBMin() const noexcept
	{ 
		return fData[0].xyz;
	}

	SFZ_CUDA_CALLABLE vec3 leftChildAABBMax() const noexcept
	{
		return vec3(fData[0].w, fData[1].xy);
	}

	SFZ_CUDA_CALLABLE vec3 rightChildAABBMin() const noexcept
	{
		return vec3(fData[1].zw, fData[2].x);
	}

	SFZ_CUDA_CALLABLE vec3 rightChildAABBMax() const noexcept
	{
		return fData[2].yzw;
	}

	SFZ_CUDA_CALLABLE uint32_t leftChildIndex() const noexcept
	{
		return iData.x;
	}

	SFZ_CUDA_CALLABLE uint32_t rightChildIndex() const noexcept
	{
		return iData.y;
	}

	SFZ_CUDA_CALLABLE uint32_t leftChildNumTriangles() const noexcept
	{
		 return iData.z;
	}

	SFZ_CUDA_CALLABLE uint32_t rightChildNumTriangles() const noexcept
	{
		return iData.w;
	}

	SFZ_CUDA_CALLABLE bool leftChildIsLeaf() const noexcept
	{
		return leftChildNumTriangles() == 0;
	}

	SFZ_CUDA_CALLABLE bool rightChildIsLeaf() const noexcept
	{
		return rightChildNumTriangles() == 0;
	}

	// Setters
	// --------------------------------------------------------------------------------------------

	SFZ_CUDA_CALLABLE void setLeftChildAABB(const vec3& min, const vec3& max) noexcept
	{
		fData[0].xyz = min;
		fData[0].w = max.x;
		fData[1].xy = max.yz;
	}

	SFZ_CUDA_CALLABLE void setRightChildAABB(const vec3& min, const vec3& max) noexcept
	{
		fData[1].zw = min.xy;
		fData[2].x = min.z;
		fData[2].yzw = max;
	}

	SFZ_CUDA_CALLABLE void setLeftChildInner(uint32_t nodeIndex) noexcept
	{
		iData.x = nodeIndex;
		iData.z = 0u;
	}

	SFZ_CUDA_CALLABLE void setLeftChildLeaf(uint32_t triangleIndex, uint32_t numTriangles) noexcept
	{
		iData.x = triangleIndex;
		iData.z = numTriangles;
	}

	SFZ_CUDA_CALLABLE void setRightChildInner(uint32_t nodeIndex) noexcept
	{
		iData.y = nodeIndex;
		iData.w = 0u;
	}

	SFZ_CUDA_CALLABLE void setRightChildLeaf(uint32_t triangleIndex, uint32_t numTriangles) noexcept
	{
		iData.y = triangleIndex;
		iData.w = numTriangles;
	}
};

static_assert(sizeof(BVHNode) == 64, "BVHNode is padded");

} // namespace phe
