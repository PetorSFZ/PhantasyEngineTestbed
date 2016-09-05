// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#include "phantasy_engine/renderers/cpu_ray_tracer/AabbTree.hpp"

#include <chrono>

namespace sfz {

bool rayIntersectsAabb(const vec3& origin, const vec3& dir, const AABB& aabb)
{
	vec3 t1 = (aabb.min - origin) / dir;
	vec3 t2 = (aabb.max - origin) / dir;

	float tmin = sfz::maxElement(sfz::min(t1, t2));
	float tmax = sfz::minElement(sfz::max(t1, t2));

	return tmax >= tmin;
}

TriangleIntersection rayTriangleIntersect(const Triangle& triangle, const vec3& origin, const vec3& dir)
{
	const vec3 p0 = triangle.p0;
	const vec3 p1 = triangle.p1;
	const vec3 p2 = triangle.p2;

	const float EPS = 0.00001f;

	vec3 e1 = p1 - p0;
	vec3 e2 = p2 - p0;
	vec3 q = cross(dir, e2);
	float a = dot(e1, q);
	if (-EPS < a && a < EPS) return {false, 0.0f, 0.0f};

	float f = 1.0f / a;
	vec3 s = origin - p0;
	float u = f * dot(s, q);
	if (u < 0.0f) return {false, 0.0f, 0.0f};

	vec3 r = cross(s, e1);
	float v = f * dot(dir, r);
	if (v < 0.0f || (u + v) > 1.0f) return {false, 0.0f, 0.0f};

	float t = f * dot(e2, r);
	if (t < 0.0f) return {false, 0.0f, 0.0f}; // only trace the ray forward
	return {true, t, u, v};
}

/// Expand the AABB defined by min and max to also contain point.
void expandAabb(vec3& min, vec3& max, const vec3& point)
{
	max = sfz::max(max, point);
	min = sfz::min(min, point);
}

/// Given a list of triangles (assumed to have associated AABBs) create an
/// AABB containing all of them.
AABB AabbTree::aabbFromUseExisting(const DynArray<uint32_t>& triangleInds) noexcept
{
	sfz_assert_debug(triangleInds.size() > 0);

	// Initialize to first vertex
	const AABB& firstAabb = triangleAabbs[triangleInds[0]];
	vec3 min = firstAabb.min;
	vec3 max = firstAabb.max;

	for (const uint32_t triangleInd : triangleInds) {
		const AABB& aabb = triangleAabbs[triangleInd];
		min = sfz::min(min, aabb.min);
		max = sfz::max(max, aabb.max);
	}
	return AABB(min, max);
}

AABB AabbTree::aabbFrom(uint32_t triangleInd) noexcept
{
	// Initialize to first vertex
	vec3 min = triangles[triangleInd].p0;
	vec3 max = min;

	expandAabb(min, max, triangles[triangleInd].p1);
	expandAabb(min, max, triangles[triangleInd].p2);

	return AABB(min, max);
}

void AabbTree::fillNode(uint32_t nodeInd, DynArray<BvhNode>& nodes, const DynArray<uint32_t>& triangleInds, uint32_t depth) noexcept
{
	if (triangleInds.size() == 0) {
		return;
	}

	BvhNode& node = nodes[nodeInd];
	if (triangleInds.size() == 1) {
		node.triangleInd = triangleInds[0];
		return;
	}

	node.aabb = aabbFromUseExisting(triangleInds);

	// Determine the longest axis, which will be used to split along
	// x = 0, y = 1, z = 2
	uint8_t splitAxis = 0;
	float xLength = node.aabb.xExtent();
	float yLength = node.aabb.yExtent();
	float zLength = node.aabb.zExtent();
	float maxAxisLength = xLength;
	if (yLength > maxAxisLength) {
		maxAxisLength = yLength;
		splitAxis = 1;
	}
	if (zLength > maxAxisLength) {
		maxAxisLength = zLength;
		splitAxis = 2;
	}

	float axisSplitPos = node.aabb.min[splitAxis] + maxAxisLength / 2.0f;

	DynArray<uint32_t> leftTriangles;
	DynArray<uint32_t> rightTriangles;

	for (const uint32_t triangleInd : triangleInds) {
		const auto& triangle = triangles[triangleInd];

		vec3 center = triangle.p0 / 3.0f + triangle.p1 / 3.0f + triangle.p2 / 3.0f;
		if (center[splitAxis] < axisSplitPos) {
			leftTriangles.add(triangleInd);
		} else {
			rightTriangles.add(triangleInd);
		}
	}

	bool leftSmaller = leftTriangles.size() < rightTriangles.size();
	DynArray<uint32_t>& smallerList = leftSmaller ? leftTriangles : rightTriangles;
	DynArray<uint32_t>& largerList = leftSmaller ? rightTriangles : leftTriangles;
	float smallerExtremePos = leftSmaller ? node.aabb.min[splitAxis] : node.aabb.max[splitAxis];

	// Handle edge case of everything going in one bin by manually copying over one of the triangles
	if (smallerList.size() == 0) {
		for (uint32_t i = 0; i < largerList.size(); i++) {
			uint32_t triangleInd = largerList[i];
			const auto& triangle = triangles[triangleInd];
			for (const vec3& vertex : {triangle.p0, triangle.p1, triangle.p2}) {
				// Intentionally use exact float equality, since no operations should have been
				// done on the stored values
				if (vertex[splitAxis] == smallerExtremePos) {
					smallerList.add(triangleInd);
					largerList.remove(i);
					goto breakNestedFor;
				}
			}
		}
	}
breakNestedFor:
	sfz_assert_debug(leftTriangles.size() != 0);
	sfz_assert_debug(rightTriangles.size() != 0);
	sfz_assert_debug(leftTriangles.size() + rightTriangles.size() ==  triangleInds.size());
	nodes.add(BvhNode());
	nodes.add(BvhNode());

	nodes[nodeInd].left = nodes.size() - 2;
	nodes[nodeInd].right = nodes.size() - 1;

	fillNode(nodes[nodeInd].left, nodes, leftTriangles, depth + 1);
	fillNode(nodes[nodeInd].right, nodes, rightTriangles, depth + 1);
}

void AabbTree::constructFrom(const DynArray<Renderable>& renderables) noexcept
{
	DynArray<Triangle> tmpTriangles;

	for (const Renderable& renderable : renderables) {
		for (const RenderableComponent& component : renderable.components) {
			const RawGeometry& rawGeometry = component.geometry;

			uint32_t newSize = tmpTriangles.size() + rawGeometry.indices.size() / 3;
			tmpTriangles.ensureCapacity(newSize);

			const DynArray<Vertex>& vertices = rawGeometry.vertices;
			for (uint32_t i = 0; i < rawGeometry.indices.size() - 2; i += 3) {
				tmpTriangles.add({vertices[rawGeometry.indices[i]].pos, vertices[rawGeometry.indices[i + 1]].pos, vertices[rawGeometry.indices[i + 2]].pos});
			}
		}
	}

	constructFrom(std::move(tmpTriangles));
}

void AabbTree::constructFrom(DynArray<Triangle> trianglesIn) noexcept
{
	triangles = std::move(trianglesIn);

	triangleAabbs.ensureCapacity(triangles.size());

	for (uint32_t i = 0; i < triangles.size(); i++) {
		triangleAabbs.add(aabbFrom(i));
	}
	sfz_assert_debug(triangleAabbs.size() == triangles.size());

	DynArray<uint32_t> triangleInds;
	triangleInds.ensureCapacity(triangles.size());
	for (uint32_t i = 0; i < triangles.size(); i++) {
		triangleInds.add(i);
	}

	nodes.add(BvhNode());
	fillNode(0, nodes, triangleInds, 0);
}

RaycastResult AabbTree::raycast(vec3 origin, vec3 direction) const noexcept
{
	sfz_assert_debug(nodes.size() > 0);
	DynArray<uint32_t> nodeStack;
	nodeStack.add(0);

	const Triangle* closestTriangle = nullptr;
	TriangleIntersection closestIntersection;
	closestIntersection.intersected = false;

	while (nodeStack.size() > 0) {
		uint32_t nodeInd = nodeStack[nodeStack.size() - 1];
		const BvhNode& node = nodes[nodeInd];
		nodeStack.remove(nodeStack.size() - 1);

		if (node.isEmpty()) {
			continue;
		}

		// AABBs for triangles/leaves are stored in separate array
		const AABB& aabb = node.isLeaf() ? triangleAabbs[node.triangleInd] : node.aabb;

		if (!rayIntersectsAabb(origin, direction, aabb)) {
			continue;
		}

		if (node.isLeaf()) {
			const Triangle& triangle = triangles[node.triangleInd];
			TriangleIntersection intersection = rayTriangleIntersect(triangle, origin, direction);
			if (intersection.intersected && (!closestIntersection.intersected || intersection.t < closestIntersection.t)) {
				// Replace previous best candidate with a new one
				closestTriangle = &triangles[node.triangleInd];
				closestIntersection = intersection;
			}
		} else {
			sfz_assert_debug(node.left != UINT32_MAX);
			sfz_assert_debug(node.right != UINT32_MAX);
			nodeStack.add(nodes[nodeInd].left);
			nodeStack.add(nodes[nodeInd].right);
		}

	}
	RaycastResult res;
	res.intersection = closestIntersection;
	res.triangle = closestTriangle;
	return res;
}

} // namespace sfz
