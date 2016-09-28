// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#include "phantasy_engine/ray_tracer_common/DynamicBVH.hpp"

#include "phantasy_engine/ray_tracer_common/BVHNode.hpp"

#include <sfz/geometry/AABB.hpp>
#include <sfz/math/MatrixSupport.hpp>

#include "phantasy_engine/ray_tracer_common/StaticBVHBuilder.hpp"

namespace phe {

using namespace sfz;

void fillDynamicNode(BVH& bvh, AABB& aabb,
	uint32_t& currentTriangleIndex,
	const DynArray<uint32_t>& triangleInds,
	const DynArray<TriangleVertices>& inTriangles,
	const DynArray<AABB>& inTriangleAabbs)
{
	bvh.nodes.add(BVHNode());
	BVHNode& node = bvh.nodes[bvh.nodes.size() - 1];

	// Split AABB in center
	// Determine the longest axis, which will be used to split along
	// x = 0, y = 1, z = 2
	uint8_t splitAxis = 0;
	vec3 extents = aabb.extents();
	float maxAxisLength = extents[0];
	if (extents[1] > maxAxisLength) {
		maxAxisLength = extents[1];
		splitAxis = 1;
	}
	if (extents[2] > maxAxisLength) {
		maxAxisLength = extents[2];
		splitAxis = 2;
	}
	float axisSplitPos = aabb.min[splitAxis] + maxAxisLength / 2.0f;

	DynArray<uint32_t> leftTriangles = DynArray<uint32_t>(0, 0, triangleInds.size());
	DynArray<uint32_t> rightTriangles = DynArray<uint32_t>(0, 0, triangleInds.size());

	for (const uint32_t triangleInd : triangleInds) {
		const auto& triangle = inTriangles[triangleInd];

		float center = triangle.v0[splitAxis] / 3.0f + triangle.v1[splitAxis] / 3.0f + triangle.v2[splitAxis] / 3.0f;
		if (center < axisSplitPos) {
			leftTriangles.add(triangleInd);
		}
		else {
			rightTriangles.add(triangleInd);
		}
	}

	bool leftSmaller = leftTriangles.size() < rightTriangles.size();
	sfz::DynArray<uint32_t>& smallerList = leftSmaller ? leftTriangles : rightTriangles;
	sfz::DynArray<uint32_t>& largerList = leftSmaller ? rightTriangles : leftTriangles;
	float smallerExtremePos = leftSmaller ? aabb.min[splitAxis] : aabb.max[splitAxis];

	// Handle edge case of everything going in one bin by manually copying over one of the triangles
	if (smallerList.size() == 0) {
		for (uint32_t i = 0; i < largerList.size(); i++) {
			uint32_t triangleInd = largerList[i];
			const auto& triangle = inTriangles[triangleInd];
			for (const vec4& vertex : { triangle.v0, triangle.v1, triangle.v2 }) {
				// Intentionally use exact float equality, since no operations should have been
				// done on the stored values
				if (vertex[splitAxis] == smallerExtremePos) {
					largerList.remove(i);
					smallerList.add(triangleInd);
					goto breakNestedFor1;
				}
			}
		}
	}
breakNestedFor1:
	sfz_assert_debug(smallerList.size() != 0);
	sfz_assert_debug(largerList.size() != 0);
	sfz_assert_debug(smallerList.size() + largerList.size() == triangleInds.size());

	// Compute child AABBs
	AABB leftAabb, rightAabb;
	const AABB& leftTri = inTriangleAabbs[smallerList[0]];
	leftAabb.min = leftTri.min;
	leftAabb.max = leftTri.max;

	for (int i = 1; i < smallerList.size(); i++) {
		const AABB& tri = inTriangleAabbs[smallerList[i]];
		leftAabb.min = min(leftAabb.min, leftTri.min);
		leftAabb.max = max(leftAabb.max, leftTri.max);
	}

	const AABB& rightTri = inTriangleAabbs[largerList[0]];
	rightAabb.min = rightTri.min;
	rightAabb.max = rightTri.max;

	for (int i = 1; i < largerList.size(); i++) {
		const AABB& tri = inTriangleAabbs[largerList[i]];
		rightAabb.min = min(rightAabb.min, tri.min);
		rightAabb.max = min(rightAabb.max, tri.max);
	}

	node.setLeftChildAABB(leftAabb.min, leftAabb.max);
	node.setRightChildAABB(rightAabb.min, rightAabb.max);

	// Create right child
	if (largerList.size() <= 3) {
		node.setRightChildLeaf(currentTriangleIndex, largerList.size());
		currentTriangleIndex += largerList.size();
	}
	else {
		node.setRightChildInner(bvh.nodes.size());
		fillDynamicNode(bvh, rightAabb, currentTriangleIndex, largerList, inTriangles, inTriangleAabbs);
	}

	// Create left child
	if (smallerList.size() <= 3) {
		node.setLeftChildLeaf(currentTriangleIndex, smallerList.size());
		currentTriangleIndex += smallerList.size();
	}
	else {
		node.setLeftChildInner(bvh.nodes.size());
		fillDynamicNode(bvh, leftAabb, currentTriangleIndex, smallerList, inTriangles, inTriangleAabbs);
	}
}

BVH createDynamicBvh(const RawMesh& mesh, const mat4& transform)
{
	BVH bvh;
	bvh.nodes = DynArray<BVHNode>(0, 2 * mesh.indices.size() / 3); // Set capacity to twice the number of triangles (a leaf can contain 1 triangle if it is manually moved)

	uint32_t numTriangles = mesh.indices.size() / 3;

	DynArray<TriangleVertices> triangleVertices = DynArray<TriangleVertices>(0, numTriangles);
	DynArray<AABB> aabbs = DynArray<AABB>(0, numTriangles);
	DynArray<TriangleData> triangleDatas = DynArray<TriangleData>(0, numTriangles);
	DynArray<uint32_t> triangleIndices = DynArray<uint32_t>(0, 0, numTriangles);

	for (int i = 0; i < mesh.indices.size(); i += 3) {
		TriangleVertices vertices;
		vec3 v0 = transformPoint(transform, mesh.vertices[mesh.indices[i + 0]].pos);
		vec3 v1 = transformPoint(transform, mesh.vertices[mesh.indices[i + 1]].pos);
		vec3 v2 = transformPoint(transform, mesh.vertices[mesh.indices[i + 2]].pos);
		vertices.v0 = vec4(v0, 0.0f);
		vertices.v1 = vec4(v1, 0.0f);
		vertices.v2 = vec4(v2, 0.0f);
		triangleVertices.add(vertices);

		TriangleData data;
		data.n0 = transformDir(transform, mesh.vertices[mesh.indices[i + 0]].normal);
		data.n1 = transformDir(transform, mesh.vertices[mesh.indices[i + 1]].normal);
		data.n2 = transformDir(transform, mesh.vertices[mesh.indices[i + 2]].normal);

		data.uv0 = mesh.vertices[mesh.indices[i + 0]].uv;
		data.uv1 = mesh.vertices[mesh.indices[i + 1]].uv;
		data.uv2 = mesh.vertices[mesh.indices[i + 2]].uv;

		data.materialIndex = mesh.materialIndices[mesh.indices[i]];
		triangleDatas.add(data);

		AABB aabb;
		aabb.min = min(v0, min(v1, v2));
		aabb.max = max(v0, max(v1, v2));
		aabbs.add(aabb);

		triangleIndices.add(i / 3);
	}

	AABB rootAabb = aabbs[0];
	for (const AABB& aabb : aabbs) {
		rootAabb.min = min(rootAabb.min, aabb.min);
		rootAabb.max = max(rootAabb.max, aabb.max);
	}

	uint32_t triangleStartIndex = 0;
	fillDynamicNode(bvh, rootAabb, triangleStartIndex, triangleIndices, triangleVertices, aabbs);
	bvh.triangleDatas = triangleDatas;
	bvh.triangleVerts = triangleVertices;
	return bvh;
}

void fillOuterNode(OuterBVH& bvh, uint32_t& currentIndex, const DynArray<uint32_t>& bvhIndices, const DynArray<AABB>& aabbs, const AABB& aabb)
{
	bvh.nodes.add(OuterBVHNode());
	OuterBVHNode& node = bvh.nodes[bvh.nodes.size() - 1];

	// Split bvh
	DynArray<uint32_t> leftIndices = DynArray<uint32_t>(0, 0, bvhIndices.size());
	DynArray<uint32_t> rightIndices = DynArray<uint32_t>(0, 0, bvhIndices.size());

	// Split AABB in center
	// Determine the longest axis, which will be used to split along
	// x = 0, y = 1, z = 2
	uint8_t splitAxis = 0;
	vec3 extents = aabb.extents();
	float maxAxisLength = extents[0];
	if (extents[1] > maxAxisLength) {
		maxAxisLength = extents[1];
		splitAxis = 1;
	}
	if (extents[2] > maxAxisLength) {
		maxAxisLength = extents[2];
		splitAxis = 2;
	}
	float axisSplitPos = aabb.min[splitAxis] + maxAxisLength / 2.0f;

	for (const uint32_t index : bvhIndices) {
		const auto& bvhAabb = aabbs[index];

		float center = bvhAabb.min[splitAxis] + bvhAabb.extents()[splitAxis] / 2.0f;
		if (center < axisSplitPos) {
			leftIndices.add(index);
		}
		else {
			rightIndices.add(index);
		}
	}

	bool leftSmaller = leftIndices.size() < rightIndices.size();
	DynArray<uint32_t>& smallerList = leftSmaller ? leftIndices : rightIndices;
	DynArray<uint32_t>& largerList = leftSmaller ? rightIndices : leftIndices;
	float smallerExtremePos = leftSmaller ? aabb.min[splitAxis] : aabb.max[splitAxis];

	// Handle edge case of everything going in one bin by manually copying over one of the triangles
	if (smallerList.size() == 0) {
		for (uint32_t i = 0; i < largerList.size(); i++) {
			uint32_t index = largerList[i];
			const auto& aabb = aabbs[index];
			for (const vec3 point : { aabb.min, aabb.max }) {
				// Intentionally use exact float equality, since no operations should have been
				// done on the stored values
				if (point[splitAxis] == smallerExtremePos) {
					largerList.remove(i);
					smallerList.add(index);
					goto breakNestedFor2;
				}
			}
		}
	}
breakNestedFor2:

	AABB leftAabb, rightAabb;

	leftAabb.min = aabbs[smallerList[0]].min;
	leftAabb.max = aabbs[smallerList[0]].max;
	for (int i = 1; i < smallerList.size(); i++) {
		leftAabb.min = min(leftAabb.min, aabbs[smallerList[i]].min);
		leftAabb.max = max(leftAabb.max, aabbs[smallerList[i]].max);
	}

	rightAabb.min = aabbs[largerList[0]].min;
	rightAabb.max = aabbs[largerList[0]].max;
	for (int i = 1; i < largerList.size(); i++) {
		rightAabb.min = min(rightAabb.min, aabbs[largerList[i]].min);
		rightAabb.max = max(rightAabb.max, aabbs[largerList[i]].max);
	}

	// Right child
	if (largerList.size() == 1) {
		node.rightIsLeaf = true;
		node.rightIndex = currentIndex++;
	}
	else {
		node.rightIsLeaf = false;
		node.rightIndex = bvh.nodes.size();
		fillOuterNode(bvh, currentIndex, largerList, aabbs, rightAabb);
	}

	// Left child
	if (smallerList.size() == 1) {
		node.leftIsLeaf = true;
		node.leftIndex = currentIndex++;
	}
	else {
		node.leftIsLeaf = false;
		node.leftIndex = bvh.nodes.size();
		fillOuterNode(bvh, currentIndex, smallerList, aabbs, leftAabb);
	}
}

OuterBVH createOuterBvh(DynArray<BVH>& bvhs)
{
	uint32_t numBvhs = bvhs.size();

	OuterBVH bvh;
	bvh.nodes = DynArray<OuterBVHNode>(0, bvhs.size() * 2);

	DynArray<AABB> aabbs = DynArray<AABB>(0, numBvhs);
	DynArray<uint32_t> indices = DynArray<uint32_t>(0, 0, numBvhs);

	AABB rootAabb;
	rootAabb.min = bvhs[0].nodes[0].leftChildAABBMin();
	rootAabb.max = bvhs[0].nodes[0].leftChildAABBMax();

	for (int i = 0; i < numBvhs; i++) {
		AABB aabb;
		BVHNode& root = bvhs[i].nodes[0];
		aabb.min = min(root.leftChildAABBMin(), root.rightChildAABBMin());
		aabb.max = max(root.leftChildAABBMax(), root.rightChildAABBMax());
		aabbs.add(aabb);
		indices.add(i);

		rootAabb.min = min(rootAabb.min, aabb.min);
		rootAabb.max = min(rootAabb.max, aabb.max);
	}

	uint32_t startIndex = 0;
	fillOuterNode(bvh, startIndex, indices, aabbs, rootAabb);
	bvh.bvhs = bvhs;

	return bvh;
}

OuterBVH createDynamicBvh(const DynArray<RawMesh>& meshes, const DynArray<mat4>& transforms)
{
	uint32_t numSubBvhs = meshes.size();
	DynArray<BVH> bvhs = DynArray<BVH>(0, numSubBvhs);

	for (int i = 0; i < numSubBvhs; i++) {
		BVH bvh = createDynamicBvh(meshes[i], transforms[i]);
		sanitizeBVH(bvh);
		bvhs.add(bvh);
	}

	return createOuterBvh(bvhs);
}

}
