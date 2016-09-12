// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#include <sfz/geometry/AABB.hpp>

#include "phantasy_engine/ray_tracer_common/BVH.hpp"

namespace phe {

// Statics
// ------------------------------------------------------------------------------------------------

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

	// Determine the longest axis, which will be used to split along
	// x = 0, y = 1, z = 2
	uint8_t splitAxis = 0;
	float xLength = aabb.xExtent();
	float yLength = aabb.yExtent();
	float zLength = aabb.zExtent();
	float maxAxisLength = xLength;
	if (yLength > maxAxisLength) {
		maxAxisLength = yLength;
		splitAxis = 1;
	}
	if (zLength > maxAxisLength) {
		maxAxisLength = zLength;
		splitAxis = 2;
	}

	float axisSplitPos = aabb.min[splitAxis] + maxAxisLength / 2.0f;

	DynArray<uint32_t> leftTriangles;
	DynArray<uint32_t> rightTriangles;

	for (const uint32_t triangleInd : triangleInds) {
		const auto& triangle = inTriangles[triangleInd];

		vec3 center = vec3(triangle.v0) / 3.0f + vec3(triangle.v1) / 3.0f + vec3(triangle.v2) / 3.0f;
		if (center[splitAxis] < axisSplitPos) {
			leftTriangles.add(triangleInd);
		} else {
			rightTriangles.add(triangleInd);
		}
	}

	bool leftSmaller = leftTriangles.size() < rightTriangles.size();
	DynArray<uint32_t>& smallerList = leftSmaller ? leftTriangles : rightTriangles;
	DynArray<uint32_t>& largerList = leftSmaller ? rightTriangles : leftTriangles;
	float smallerExtremePos = leftSmaller ? aabb.min[splitAxis] : aabb.max[splitAxis];

	// Handle edge case of everything going in one bin by manually copying over one of the triangles
	if (smallerList.size() == 0) {
		for (uint32_t i = 0; i < largerList.size(); i++) {
			uint32_t triangleInd = largerList[i];
			const auto& triangle = inTriangles[triangleInd];
			for (const vec3& vertex : {vec3(triangle.v0), vec3(triangle.v1), vec3(triangle.v2)}) {
				// Intentionally use exact float equality, since no operations should have been
				// done on the stored values
				if (vertex[splitAxis] == smallerExtremePos) {
					largerList.remove(i);
					smallerList.add(triangleInd);
					goto breakNestedFor;
				}
			}
		}
	}
breakNestedFor:
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
