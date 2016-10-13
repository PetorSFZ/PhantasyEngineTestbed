// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#include "phantasy_engine/ray_tracer_common/StaticBVHBuilder.hpp"

#include <sfz/geometry/AABB.hpp>
#include <sfz/math/Vector.hpp>

#include "phantasy_engine/sbvh-stuff/SplitBVHBuilder.h"
#include "phantasy_engine/sbvh-stuff/BVH.h"
#include "phantasy_engine/sbvh-stuff/Scene.h"

namespace phe {

using sfz::vec3i;

// Statics
// ------------------------------------------------------------------------------------------------

static int32_t sanitizeInternal(BVH& bvh, int32_t oldNodeIndex, DynArray<BVHNode>& newNodes) noexcept
{
	int32_t newNodeIndex = int32_t(newNodes.size());
	newNodes.add(bvh.nodes[oldNodeIndex]);
	BVHNode& newNode = newNodes[newNodeIndex];

	// Process left child
	if (newNode.leftChildIsLeaf()) {

		// Makes sure triangle index is bitwise negated
		if (newNode.leftChildIndexRaw() >= 0) {
			newNode.setLeftChildLeaf(~newNode.leftChildIndexRaw(), newNode.leftChildNumTriangles());
		}
		
		// Sets padding to 0 in all non-last triangles in leaf
		int32_t firstTriIndex = ~newNode.leftChildIndexRaw();
		int32_t lastTriIndex = firstTriIndex + newNode.leftChildNumTriangles() - 1;
		for (int32_t i = firstTriIndex; i < lastTriIndex; i++) {
			bvh.triangleVerts[i].v0.w = 0.0f;
			bvh.triangleVerts[i].v1.w = 0.0f;
			bvh.triangleVerts[i].v2.w = 0.0f;
		}
		
		// Sets padding to -1.0f in last triangle in leaf (end marker)
		bvh.triangleVerts[lastTriIndex].v0.w = -1.0f;
		bvh.triangleVerts[lastTriIndex].v1.w = -1.0f;
		bvh.triangleVerts[lastTriIndex].v2.w = -1.0f;
	}
	else {
		newNode.setLeftChildInner(sanitizeInternal(bvh, newNode.leftChildIndexRaw(), newNodes));
	}

	// Process right child
	if (newNode.rightChildIsLeaf()) {
		
		// Makes sure triangle index is bitwise negated
		if (newNode.rightChildIndexRaw() >= 0) {
			newNode.setRightChildLeaf(~newNode.rightChildIndexRaw(), newNode.rightChildNumTriangles());
		}
		
		// Sets padding to 0 in all non-last triangles in leaf
		int32_t firstTriIndex = ~newNode.rightChildIndexRaw();
		int32_t lastTriIndex = firstTriIndex + newNode.rightChildNumTriangles() - 1;
		for (int32_t i = firstTriIndex; i < lastTriIndex; i++) {
			bvh.triangleVerts[i].v0.w = 0.0f;
			bvh.triangleVerts[i].v1.w = 0.0f;
			bvh.triangleVerts[i].v2.w = 0.0f;
		}
		
		// Sets padding to -1.0f in last triangle in leaf (end marker)
		bvh.triangleVerts[lastTriIndex].v0.w = -1.0f;
		bvh.triangleVerts[lastTriIndex].v1.w = -1.0f;
		bvh.triangleVerts[lastTriIndex].v2.w = -1.0f;
	}
	else {
		newNode.setRightChildInner(sanitizeInternal(bvh, newNode.rightChildIndexRaw(), newNodes));
	}

	return newNodeIndex;
}

static void convertRecursively(phe::BVH& bvh, uint32_t& currentTriangleIndex, const nv::BVHNode* node)
{
	uint32_t nodeIndex = bvh.nodes.size();

	bvh.nodes.add(BVHNode());
	/*
	nv::BVHNode* leftChild = node->getChildNode(0);
	nv::BVHNode* rightChild = node->getChildNode(1);
	
	bvh.nodes[nodeIndex].setLeftChildAABB(leftChild->m_bounds.min(), leftChild->m_bounds.max());
	bvh.nodes[nodeIndex].setRightChildAABB(rightChild->m_bounds.min(), rightChild->m_bounds.max());

	if (rightChild->isLeaf()) {
		bvh.nodes[nodeIndex].setRightChildLeaf(currentTriangleIndex, rightChild->getNumTriangles());
		currentTriangleIndex += rightChild->getNumTriangles();
	}
	else {
		bvh.nodes[nodeIndex].setRightChildInner(bvh.nodes.size());
		convertRecursively(bvh, currentTriangleIndex, rightChild);
	}

	if (leftChild->isLeaf()) {
		bvh.nodes[nodeIndex].setLeftChildLeaf(currentTriangleIndex, leftChild->getNumTriangles());
		currentTriangleIndex += leftChild->getNumTriangles();
	}
	else {
		bvh.nodes[nodeIndex].setLeftChildInner(bvh.nodes.size());
		convertRecursively(bvh, currentTriangleIndex, leftChild);
	}
	*/
	
	void (BVHNode::*setChildAABBFunctions[])(const sfz::vec3& min, const sfz::vec3& max) = { &BVHNode::setLeftChildAABB, &BVHNode::setRightChildAABB };
	void (BVHNode::*setChildLeafFunctions[])(int32_t triangleIndex, int32_t numTriangles) = { &BVHNode::setLeftChildLeaf, &BVHNode::setRightChildLeaf };
	void (BVHNode::*setChildInnerFunctions[])(int32_t nodeIndex) = { &BVHNode::setLeftChildInner, &BVHNode::setRightChildInner };

	// Backwards loop -> Depth-first search in the right child
	for (int64_t i = 1; i >= 0; --i) {
		nv::BVHNode* child = node->getChildNode(i);
		(bvh.nodes[nodeIndex].*setChildAABBFunctions[i])(child->m_bounds.min(), child->m_bounds.max());

		if (child->isLeaf()) {
			(bvh.nodes[nodeIndex].*setChildLeafFunctions[i])(currentTriangleIndex, child->getNumTriangles());
			currentTriangleIndex += child->getNumTriangles();
		}
		else {
			(bvh.nodes[nodeIndex].*setChildInnerFunctions[i])(bvh.nodes.size());
			convertRecursively(bvh, currentTriangleIndex, child);
		}
	}
	
}

// Members
// ------------------------------------------------------------------------------------------------

BVH buildStaticBVH(RawMesh& mesh) noexcept
{
	DynArray<TriangleVertices> inTriangles;
	DynArray<TriangleData> inTriangleDatas;

	inTriangles.setCapacity(mesh.indices.size() / 3);
	inTriangleDatas.setCapacity(mesh.indices.size() / 3);

	for (uint32_t i = 0; i < mesh.indices.size() - 2; i += 3) {
		const Vertex& v0 = mesh.vertices[mesh.indices[i]];
		const Vertex& v1 = mesh.vertices[mesh.indices[i + 1]];
		const Vertex& v2 = mesh.vertices[mesh.indices[i + 2]];

		TriangleVertices triTmp;
		triTmp.v0 = vec4(v0.pos, 0.0f);
		triTmp.v1 = vec4(v1.pos, 0.0f);
		triTmp.v2 = vec4(v2.pos, 0.0f);
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

	return buildStaticFrom(inTriangles, inTriangleDatas);
}

void buildStaticBVH(StaticScene& scene) noexcept
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
			triTmp.v0 = vec4(v0.pos, 0.0f);
			triTmp.v1 = vec4(v1.pos, 0.0f);
			triTmp.v2 = vec4(v2.pos, 0.0f);
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

	scene.bvh = std::move(buildStaticFrom(inTriangles, inTriangleDatas));
}

phe::BVH buildStaticFrom(const DynArray<TriangleVertices>& inTriangles,
                    const DynArray<TriangleData>& inTriangleDatas) noexcept
{
	// Create and convert data from our format to NVIDIA format
	nv::Array<nv::Scene::Triangle> triangles;
	nv::S32 numTriangles = inTriangles.size();
	nv::Array<sfz::vec3> vertices;
	nv::S32 numVertices = 3 * numTriangles;

	for (TriangleVertices triangle : inTriangles) {
		vertices.add(triangle.v0.xyz);
		vertices.add(triangle.v1.xyz);
		vertices.add(triangle.v2.xyz);
	}
	for (uint32_t i = 0; i < numVertices; i += 3) {
		nv::Scene::Triangle triangle;
		triangle.vertices = vec3i{ (nv::S32)i, (nv::S32)i + 1, (nv::S32)i + 2 };
		triangles.add(triangle);
	}

	nv::Scene scene{ numTriangles, numVertices, triangles, vertices };

	nv::Platform platform;
	nv::BVH::BuildParams params;
	nv::BVH::Stats stats;
	nv::BVH sbvh(&scene, platform, params);

	phe::BVH convertedBvh;

	for (uint32_t i = 0; i < sbvh.getTriIndices().getSize(); i++) {
		convertedBvh.triangleVerts.add(inTriangles[sbvh.getTriIndices().get(i)]);
		convertedBvh.triangleDatas.add(inTriangleDatas[sbvh.getTriIndices().get(i)]);
	}

	uint32_t currentTriangleIndex = 0;
	convertRecursively(convertedBvh, currentTriangleIndex, sbvh.getRoot());

	return convertedBvh; // Copy will be optimized away
}

void sanitizeBVH(BVH& bvh) noexcept
{
	DynArray<BVHNode> newNodes;
	newNodes.ensureCapacity(bvh.nodes.capacity() + 32u);

	sanitizeInternal(bvh, 0, newNodes);
	bvh.nodes = std::move(newNodes);
}

} // namespace phe
