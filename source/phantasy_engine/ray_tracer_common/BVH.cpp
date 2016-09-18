// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#include <sfz/geometry/AABB.hpp>

#include "phantasy_engine/ray_tracer_common/BVH.hpp"

namespace phe {

struct HeuristicTestNode {
	bool isLeaf;
	uint32_t numTriangles;
	sfz::AABB aabb;
};

// Statics
// ------------------------------------------------------------------------------------------------

static float surfaceArea(const sfz::AABB& aabb)
{
	vec3 extents = aabb.extents();
	return 2.0f * (extents.x * extents.y +
		extents.x * extents.z +
		extents.y * extents.z);
}

static float surfaceArea(const vec3& min, const vec3& max)
{
	vec3 extents = max - min;
	return 2.0f * (extents.x * extents.y +
		extents.x * extents.z +
		extents.y * extents.z);
}

/// Calculate the expected cost of traversing a BVH
static float surfaceAreaHeuristic(const BVHNode& root, DynArray<HeuristicTestNode>& nodes)
{
	const float innerNodeCoefficient = 1.2f;
	const float leafNodeCoefficient = 1.0f;

	const float rootArea = surfaceArea(root.min, root.max);

	float cost = 0.0f;

	for (const HeuristicTestNode& node : nodes) {
		// Corresponds to the intersection probability of a random ray
		float tempCost = surfaceArea(node.aabb) / rootArea;

		if (node.isLeaf) {
			tempCost *= node.numTriangles; // Penalty for many triangles in the same leaf node
			tempCost *= leafNodeCoefficient;
		}
		else {
			tempCost *= innerNodeCoefficient;
		}
		cost += tempCost;
	}
	return cost; // Less is better
}

/// Expand the AABB defined by min and max to also contain point.
static void expandAabb(vec3& min, vec3& max, const vec3& point)
{
	max = sfz::max(max, point);
	min = sfz::min(min, point);
}

/// Given a list of triangles (assumed to have associated AABBs) create an AABB containing all of them.
sfz::AABB createAabbUsingExisting(
	const DynArray<uint32_t>& triangleInds,
	const DynArray<sfz::AABB>& inTriangleAabbs) noexcept
{
	sfz_assert_debug(triangleInds.size() > 0);

	// Initialize to first vertex
	const sfz::AABB& firstAabb = inTriangleAabbs[triangleInds[0]];
	vec3 min = firstAabb.min;
	vec3 max = firstAabb.max;

	for (const uint32_t triangleInd : triangleInds) {
		const sfz::AABB& aabb = inTriangleAabbs[triangleInd];
		min = sfz::min(min, aabb.min);
		max = sfz::max(max, aabb.max);
	}
	return sfz::AABB(min, max);
}

struct SplitNode {
	DynArray<uint32_t> triangleIndices;
	sfz::AABB aabb;
};

struct SplitInfo {
	uint8_t axis;
	float pos;
	float cost;
	uint8_t numChildren;
	SplitNode children[2];
};

SplitInfo split(
	BVHNode& node,
	uint8_t axis,
	float pos,
	BVH& bvh,
	const DynArray<uint32_t>& triangleInds,
	const DynArray<TriangleVertices>& inTriangles,
	const DynArray<TriangleData>& inTriangleDatas,
	const DynArray<sfz::AABB>& inTriangleAabbs)
{
	SplitInfo result;

	DynArray<uint32_t> leftTriangles;
	DynArray<uint32_t> rightTriangles;

	for (const uint32_t triangleInd : triangleInds) {
		const auto& triangle = inTriangles[triangleInd];

		float center = triangle.v0[axis] / 3.0f + triangle.v1[axis] / 3.0f + triangle.v2[axis] / 3.0f;
		if (center < pos) {
			leftTriangles.add(triangleInd);
		}
		else {
			rightTriangles.add(triangleInd);
		}
	}
	DynArray<DynArray<uint32_t>> triangles;
	triangles.add(leftTriangles);
	triangles.add(rightTriangles);

	DynArray<HeuristicTestNode> testNodes;

	if (leftTriangles.size() == 0 || rightTriangles.size() == 0) {
		// This split means all nodes have gone into one bin, so put all in a single leaf test node
		leftTriangles.add(rightTriangles);
		HeuristicTestNode testNode;
		testNode.isLeaf = true;
		testNode.aabb = sfz::AABB{node.min, node.max};
		testNode.numTriangles = leftTriangles.size();
		testNodes.add(testNode);
		result.numChildren = 1;
	}
	else {
		// Get inner test nodes corresponding to split
		for (auto& triangleList : { leftTriangles, rightTriangles }) {
			HeuristicTestNode testNode;
			testNode.numTriangles = triangleList.size();
			testNode.isLeaf = testNode.numTriangles <= 3;
			testNode.aabb = createAabbUsingExisting(triangleList, inTriangleAabbs);
			testNodes.add(testNode);
			result.numChildren = 2;
		}
	}

	float cost = surfaceAreaHeuristic(bvh.nodes[0], testNodes);

	result.cost = cost;

	for (uint8_t i = 0; i < testNodes.size(); i++) {
		HeuristicTestNode testNode = testNodes[i];
		SplitNode node;
		node.aabb = testNode.aabb;
		node.triangleIndices = triangles[i];
		result.children[i] = node;
	}

	return result;
}

static void fillStaticNode(
	BVH& bvh,
	uint32_t nodeInd,
	uint32_t pathDepth,
	const DynArray<uint32_t>& triangleInds,
	const DynArray<TriangleVertices>& inTriangles,
	const DynArray<TriangleData>& inTriangleDatas,
	const DynArray<sfz::AABB>& inTriangleAabbs)
{
	BVHNode& node = bvh.nodes[nodeInd];
	
	// Split node
	vec3 extents = node.max - node.min;
	vec3i numSplits = vec3i(5);
	SplitInfo bestObjectSplit;
	bestObjectSplit.cost = INFINITY;

	for (uint8_t axis = 0; axis < 3; axis++) {
		float extentDiff = extents[axis] / float(numSplits[axis]);
		for (uint32_t splitIndex = 0; splitIndex < numSplits[axis]; splitIndex++) {
			float splitAt = node.min[axis] + splitIndex * extentDiff;
			SplitInfo objectSplit = std::move(split(node, axis, splitAt, bvh, triangleInds, inTriangles, inTriangleDatas, inTriangleAabbs));
			if (objectSplit.cost < bestObjectSplit.cost) {
				bestObjectSplit = std::move(objectSplit);
			}
		}
	}

	// Spatial split
	SplitInfo bestSpatialSplit;
	bestSpatialSplit.cost = INFINITY;

	// Get the best split option
	SplitInfo bestSplit = std::move(bestObjectSplit.cost > bestSpatialSplit.cost ? bestSpatialSplit : bestObjectSplit);

	// Create child nodes and possibly recurse
	if (bestSplit.numChildren == 1) {
		// The best split was to put in one bin
		uint32_t firstTriangleIndex = bvh.triangles.size();
		for (uint32_t triangleInd : triangleInds) {
			bvh.triangles.add(inTriangles[triangleInd]);
			bvh.triangleDatas.add(inTriangleDatas[triangleInd]);
		}
		node.setLeaf(triangleInds.size(), firstTriangleIndex);

		bvh.maxDepth = std::max(bvh.maxDepth, pathDepth + 1);
	}
	else {
		uint32_t indices[2];
		
		for (uint32_t i = 0; i < 2; i++) {
			SplitNode childNode = bestSplit.children[i];

			// If leaf
			if (childNode.triangleIndices.size() <= 3) {
				indices[i] = bvh.nodes.size();
				BVHNode leafNode;
				
				uint32_t firstTriangleIndex = bvh.triangles.size();
				for (uint32_t triangleInd : childNode.triangleIndices) {
					bvh.triangles.add(inTriangles[triangleInd]);
					bvh.triangleDatas.add(inTriangleDatas[triangleInd]);
				}
				leafNode.setLeaf(childNode.triangleIndices.size(), firstTriangleIndex);
				leafNode.min = childNode.aabb.min;
				leafNode.max = childNode.aabb.max;
				bvh.nodes.add(leafNode);
				bvh.maxDepth = std::max(bvh.maxDepth, pathDepth + 1);
			}
			else {
				// Create a new node and initialize its AABB before recursively filling
				indices[i] = bvh.nodes.size();
				BVHNode newNode;
				newNode.min = childNode.aabb.min;
				newNode.max = childNode.aabb.max;
				bvh.nodes.add(newNode);
				fillStaticNode(bvh, indices[i], pathDepth + 1, childNode.triangleIndices, inTriangles, inTriangleDatas, inTriangleAabbs);
			}
		}

		bvh.nodes[nodeInd].setInner(indices[0], indices[1]);
	}
}

// C++ container
// ------------------------------------------------------------------------------------------------

sfz::AABB aabbFrom(uint32_t triangleInd, const DynArray<TriangleVertices>& inTriangles) noexcept
{
	// Initialize to first vertex
	vec3 min = vec3(inTriangles[triangleInd].v0);
	vec3 max = min;

	expandAabb(min, max, vec3(inTriangles[triangleInd].v1));
	expandAabb(min, max, vec3(inTriangles[triangleInd].v2));

	return sfz::AABB(min, max);
}

// Members
// ------------------------------------------------------------------------------------------------

BVH buildStaticFrom(const StaticScene& scene) noexcept
{
	// Extract information from StaticScene to simple arrays, whose indicies map to the same
	// triangle.

	DynArray<TriangleVertices> inTriangles;
	DynArray<TriangleData> inTriangleDatas;

	inTriangles.ensureCapacity(scene.meshes.size() * 32u);
	inTriangleDatas.ensureCapacity(scene.meshes.size() * 32u);

	for (const RawMesh& mesh : scene.meshes) {
		uint32_t newSize = inTriangles.size() + mesh.indices.size() / 3;
		inTriangles.ensureCapacity(newSize);

		for (uint32_t i = 0; i < mesh.indices.size() - 2; i += 3) {
			const Vertex& v0 = mesh.vertices[mesh.indices[i]];
			const Vertex& v1 = mesh.vertices[mesh.indices[i + 1]];
			const Vertex& v2 = mesh.vertices[mesh.indices[i + 2]];

			TriangleVertices triTmp;
			triTmp.v0 = v0.pos;
			triTmp.v1 = v1.pos;
			triTmp.v2 = v2.pos;
			inTriangles.add(triTmp);

			TriangleData dataTmp;
			dataTmp.n0 = v0.normal;
			dataTmp.n1 = v1.normal;
			dataTmp.n2 = v2.normal;
			dataTmp.uv0 = v0.uv;
			dataTmp.uv1 = v1.uv;
			dataTmp.uv2 = v2.uv;

			dataTmp.materialIndex = mesh.materialIndices[mesh.indices[i]]; // Should be same for i, i+1 and i+2

			inTriangleDatas.add(dataTmp);
		}
	}

	return buildStaticFrom(inTriangles, inTriangleDatas); 
}

BVH buildStaticFrom(const DynArray<TriangleVertices>& inTriangles,
                    const DynArray<TriangleData>& inTriangleDatas) noexcept
{
	BVH bvh;
	// Triangle information will be saved internal arrays in an order depending on the BVH structure
	bvh.triangles.ensureCapacity(inTriangles.size());
	bvh.triangleDatas.ensureCapacity(inTriangles.size());

	// Compute AABBs for each Triangle. They are used later in the construction phase.
	DynArray<sfz::AABB> inTriangleAabbs;
	inTriangleAabbs.ensureCapacity(inTriangles.size());

	for (uint32_t i = 0; i < inTriangles.size(); i++) {
		inTriangleAabbs.add(aabbFrom(i, inTriangles));
	}
	sfz_assert_debug(inTriangleAabbs.size() == inTriangles.size());

	// Prepare the initial list of triangles that are to be divided up, which simply is all of them
	DynArray<uint32_t> triangleInds;
	triangleInds.ensureCapacity(inTriangles.size());
	for (uint32_t i = 0; i < inTriangles.size(); i++) {
		triangleInds.add(i);
	}

	// Start the top-down BVH construction by initializing root node at index 0
	BVHNode root;
	sfz::AABB rootAabb = createAabbUsingExisting(triangleInds, inTriangleAabbs);
	root.min = rootAabb.min;
	root.max = rootAabb.max;
	bvh.nodes.add(root);
	bvh.maxDepth = 0;
	fillStaticNode(bvh, 0, 1, triangleInds, inTriangles, inTriangleDatas, inTriangleAabbs);

	return bvh; // Copy will be optimized away
}

} // namespace phe
