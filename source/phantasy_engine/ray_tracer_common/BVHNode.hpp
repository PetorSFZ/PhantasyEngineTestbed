// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include <sfz/CudaCompatibility.hpp>
#include <sfz/math/Vector.hpp>

namespace phe {

using sfz::vec3;
using sfz::vec4;
using sfz::vec4i;

// BVHNode
// ------------------------------------------------------------------------------------------------

/// Each node in the BVH has two children. There are two types of children, inner nodes and leafs.
/// If the child is an inner node its index points to the next BVHNode in the array. If the child
/// is a leaf node its index points to the first triangle in the triangle arrays.
///
/// The index is bitwise negated (~) if it is a leaf. Use the "safe" index getters if you don't
/// care about that detail.
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
	// leaf node they are (bitwise negated) indices to the first triangles.
	// lcNumTriangles and rcNumTriangles are 0 in inner nodes, in leaf nodes they tell the number
	// of triangles pointed to by the leaf node 
	vec4i iData;

	// Getters
	// --------------------------------------------------------------------------------------------

	SFZ_CUDA_CALL vec3 leftChildAABBMin() const noexcept
	{ 
		return fData[0].xyz;
	}

	SFZ_CUDA_CALL vec3 leftChildAABBMax() const noexcept
	{
		return vec3(fData[0].w, fData[1].xy);
	}

	SFZ_CUDA_CALL vec3 rightChildAABBMin() const noexcept
	{
		return vec3(fData[1].zw, fData[2].x);
	}

	SFZ_CUDA_CALL vec3 rightChildAABBMax() const noexcept
	{
		return fData[2].yzw;
	}

	SFZ_CUDA_CALL int32_t leftChildIndexRaw() const noexcept
	{
		return iData.x;
	}

	SFZ_CUDA_CALL int32_t rightChildIndexRaw() const noexcept
	{
		return iData.y;
	}

	inline int32_t leftChildIndexSafe() const noexcept
	{
		int32_t index = leftChildIndexRaw();
		return (index < 0) ? ~index : index;
	}

	inline int32_t rightChildIndexSafe() const noexcept
	{
		int32_t index = rightChildIndexRaw();
		return (index < 0) ? ~index : index;
	}

	SFZ_CUDA_CALL int32_t leftChildNumTriangles() const noexcept
	{
		 return iData.z;
	}

	SFZ_CUDA_CALL int32_t rightChildNumTriangles() const noexcept
	{
		return iData.w;
	}

	SFZ_CUDA_CALL bool leftChildIsLeaf() const noexcept
	{
		return leftChildNumTriangles() != 0;
	}

	SFZ_CUDA_CALL bool rightChildIsLeaf() const noexcept
	{
		return rightChildNumTriangles() != 0;
	}

	// Setters
	// --------------------------------------------------------------------------------------------

	SFZ_CUDA_CALL void setLeftChildAABB(const vec3& min, const vec3& max) noexcept
	{
		fData[0].xyz = min;
		fData[0].w = max.x;
		fData[1].xy = max.yz;
	}

	SFZ_CUDA_CALL void setRightChildAABB(const vec3& min, const vec3& max) noexcept
	{
		fData[1].zw = min.xy;
		fData[2].x = min.z;
		fData[2].yzw = max;
	}

	SFZ_CUDA_CALL void setLeftChildInner(int32_t nodeIndex) noexcept
	{
		iData.x = nodeIndex;
		iData.z = 0;
	}

	SFZ_CUDA_CALL void setLeftChildLeaf(int32_t triangleIndex, int32_t numTriangles) noexcept
	{
		iData.x = triangleIndex;
		iData.z = numTriangles;
	}

	SFZ_CUDA_CALL void setRightChildInner(int32_t nodeIndex) noexcept
	{
		iData.y = nodeIndex;
		iData.w = 0u;
	}

	SFZ_CUDA_CALL void setRightChildLeaf(int32_t triangleIndex, int32_t numTriangles) noexcept
	{
		iData.y = triangleIndex;
		iData.w = numTriangles;
	}
};

static_assert(sizeof(BVHNode) == 64, "BVHNode is padded");

} // namespace phe
