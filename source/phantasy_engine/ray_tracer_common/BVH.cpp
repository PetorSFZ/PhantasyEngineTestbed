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

static void fillStaticNode(
	BVH& bvh,
	uint32_t nodeInd,
	uint32_t pathDepth,
	const DynArray<uint32_t>& triangleInds,
	const DynArray<TriangleVertices>& inTriangles,
	const DynArray<TriangleData>& inTriangleDatas,
	const DynArray<sfz::AABB>& inTriangleAabbs) noexcept
{
	BVHNode& node = bvh.nodes[nodeInd];
	sfz::AABB aabb = createAabbUsingExisting(triangleInds, inTriangleAabbs);
	node.min = aabb.min;
	node.max = aabb.max;

	if (triangleInds.size() <= 3) {
		uint32_t firstTriangleIndex = bvh.triangles.size();
		for (uint32_t triangleInd : triangleInds) {
			bvh.triangles.add(inTriangles[triangleInd]);
			bvh.triangleDatas.add(inTriangleDatas[triangleInd]);
		}
		node.setLeaf(triangleInds.size(), firstTriangleIndex);

		bvh.maxDepth = std::max(bvh.maxDepth, pathDepth + 1);
		return;
	}

	uint8_t splitAxis;
	float splitPos;
	float minCost = INFINITY;
	vec3 extentsDiff = aabb.extents() / 5.0f;
	DynArray<HeuristicTestNode> finalTestNodes;

	for (uint8_t axis = 0; axis < 3; axis++) {
		float extentDiff = extentsDiff[axis];
		for (int splitIndex = 0; splitIndex < 5; splitIndex++) {
			float splitAt = aabb.min[axis] + splitIndex * extentDiff;

			DynArray<uint32_t> leftTriangles;
			DynArray<uint32_t> rightTriangles;

			for (const uint32_t triangleInd : triangleInds) {
				const auto& triangle = inTriangles[triangleInd];

				vec3 center = vec3(triangle.v0) / 3.0f + vec3(triangle.v1) / 3.0f + vec3(triangle.v2) / 3.0f;
				if (center[axis] < splitAt) {
					leftTriangles.add(triangleInd);
				}
				else {
					rightTriangles.add(triangleInd);
				}
			}

			DynArray<HeuristicTestNode> testNodes;

			if (leftTriangles.size() == 0 || rightTriangles.size() == 0) {
				HeuristicTestNode testNode;
				testNode.isLeaf = true;
				testNode.aabb = aabb;
				testNode.numTriangles = leftTriangles.size() + rightTriangles.size();
				testNodes.add(testNode);
			}
			else {
				for (auto& triangleList : { leftTriangles, rightTriangles }) {
					HeuristicTestNode testNode;
					testNode.numTriangles = triangleList.size();
					testNode.isLeaf = testNode.numTriangles <= 3;
					testNode.aabb = createAabbUsingExisting(triangleList, inTriangleAabbs);
					testNodes.add(testNode);
				}
			}

			float cost = surfaceAreaHeuristic(bvh.nodes[0], testNodes);

			if (cost < minCost) {
				splitAxis = axis;
				splitPos = splitAt;
				minCost = cost;
				finalTestNodes = testNodes;
			}
		}
	}

	if (finalTestNodes.size() == 1) {
		uint32_t firstTriangleIndex = bvh.triangles.size();
		for (uint32_t triangleInd : triangleInds) {
			bvh.triangles.add(inTriangles[triangleInd]);
			bvh.triangleDatas.add(inTriangleDatas[triangleInd]);
		}
		node.setLeaf(triangleInds.size(), firstTriangleIndex);

		bvh.maxDepth = std::max(bvh.maxDepth, pathDepth + 1);
		return;
	}
	else {
		DynArray<uint32_t> leftTriangles;
		DynArray<uint32_t> rightTriangles;

		for (const uint32_t triangleInd : triangleInds) {
			const auto& triangle = inTriangles[triangleInd];

			vec3 center = vec3(triangle.v0) / 3.0f + vec3(triangle.v1) / 3.0f + vec3(triangle.v2) / 3.0f;
			if (center[splitAxis] < splitPos) {
				leftTriangles.add(triangleInd);
			}
			else {
				rightTriangles.add(triangleInd);
			}
		}

		sfz_assert_debug(leftTriangles.size() != 0);
		sfz_assert_debug(rightTriangles.size() != 0);
		sfz_assert_debug(leftTriangles.size() + rightTriangles.size() == triangleInds.size());

		bvh.nodes.add(BVHNode());
		uint32_t leftIndex = bvh.nodes.size() - 1;
		fillStaticNode(bvh, leftIndex, pathDepth + 1, leftTriangles, inTriangles, inTriangleDatas, inTriangleAabbs);

		bvh.nodes.add(BVHNode());
		uint32_t rightIndex = bvh.nodes.size() - 1;
		fillStaticNode(bvh, rightIndex, pathDepth + 1, rightTriangles, inTriangles, inTriangleDatas, inTriangleAabbs);

		bvh.nodes[nodeInd].setInner(leftIndex, rightIndex);
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

void BVH::buildStaticFrom(const StaticScene& scene) noexcept
{
	DynArray<TriangleVertices> inTriangles;
	DynArray<TriangleData> inTriangleDatas;
	inTriangles.ensureCapacity((scene.opaqueComponents.size() + scene.transparentComponents.size()) * 2u);
	inTriangleDatas.ensureCapacity((scene.opaqueComponents.size() + scene.transparentComponents.size()) * 2u);

	for (const DynArray<RenderableComponent>* renderableComponentList : {&scene.opaqueComponents, &scene.transparentComponents}) {
		for (const RenderableComponent& component : *renderableComponentList) {
			const RawGeometry& rawGeometry = component.geometry;

			uint32_t newSize = inTriangles.size() + rawGeometry.indices.size() / 3;
			inTriangles.ensureCapacity(newSize);

			for (uint32_t i = 0; i < rawGeometry.indices.size() - 2; i += 3) {
				const Vertex& v0 = rawGeometry.vertices[rawGeometry.indices[i]];
				const Vertex& v1 = rawGeometry.vertices[rawGeometry.indices[i + 1]];
				const Vertex& v2 = rawGeometry.vertices[rawGeometry.indices[i + 2]];

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
				dataTmp.materialIndex = ~0u;
				inTriangleDatas.add(dataTmp);
			}
		}
	}
	this->buildStaticFrom(inTriangles, inTriangleDatas);
}

void BVH::buildStaticFrom(const DynArray<TriangleVertices>& inTriangles,
                          const DynArray<TriangleData>& inTriangleDatas) noexcept
{
	triangles.ensureCapacity(inTriangles.size());

	DynArray<sfz::AABB> inTriangleAabbs;
	inTriangleAabbs.ensureCapacity(inTriangles.size());


	for (uint32_t i = 0; i < inTriangles.size(); i++) {
		inTriangleAabbs.add(aabbFrom(i, inTriangles));
	}
	sfz_assert_debug(inTriangleAabbs.size() == inTriangles.size());

	DynArray<uint32_t> triangleInds;
	triangleInds.ensureCapacity(inTriangles.size());
	for (uint32_t i = 0; i < inTriangles.size(); i++) {
		triangleInds.add(i);
	}

	nodes.add(BVHNode());
	this->maxDepth = 0;
	fillStaticNode(*this, 0, 1, triangleInds, inTriangles, inTriangleDatas, inTriangleAabbs);
}

} // namespace phe
