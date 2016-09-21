// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#include "phantasy_engine/ray_tracer_common/BVH.hpp"

#include "phantasy_engine/sbvh-stuff/SplitBVHBuilder.h"
#include "phantasy_engine/sbvh-stuff/BVH.h"
#include "phantasy_engine/sbvh-stuff/Scene.h"

namespace phe {

void convertRecursively(phe::BVH& bvh, uint32_t& currentTriangleIndex, const nv::BVHNode* node)
{
	uint32_t nodeIndex = bvh.nodes.size();

	bvh.nodes.add(BVHNode());
	
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
	
	/*
	void (BVHNode::*setChildAABBFunctions[])(const sfz::vec3& min, const sfz::vec3& max) = { &BVHNode::setLeftChildAABB, &BVHNode::setRightChildAABB };
	void (BVHNode::*setChildLeafFunctions[])(uint32_t triangleIndex, uint32_t numTriangles) = { &BVHNode::setLeftChildLeaf, &BVHNode::setRightChildLeaf };
	void (BVHNode::*setChildInnerFunctions[])(uint32_t nodeIndex) = { &BVHNode::setLeftChildInner, &BVHNode::setRightChildInner };

	for (int64_t i = 1; i >= 0; --i) {
		nv::BVHNode* child = node->getChildNode(i);
		(bvh.nodes[nodeIndex].*setChildAABBFunctions[i])(child->m_bounds.min(), child->m_bounds.max());

		if (child->isLeaf()) {
			(bvh.nodes[nodeIndex].*setChildLeafFunctions[i])(child->getNumTriangles(), currentTriangleIndex);
			currentTriangleIndex += child->getNumTriangles();
		}
		else {
			(bvh.nodes[nodeIndex].*setChildInnerFunctions[i])(bvh.nodes.size());
			convertRecursively(bvh, currentTriangleIndex, child);
		}
	}
	*/
}

// Members
// ------------------------------------------------------------------------------------------------

phe::BVH buildStaticFrom(const StaticScene& scene) noexcept
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

phe::BVH buildStaticFrom(const DynArray<TriangleVertices>& inTriangles,
                    const DynArray<TriangleData>& inTriangleDatas) noexcept
{
	// Create and convert data from our format to NVIDIA format
	nv::Array<nv::Scene::Triangle> triangles;
	nv::S32 numTriangles = inTriangles.size();
	nv::Array<sfz::vec3> vertices;
	nv::S32 numVertices = 3 * numTriangles;

	for (TriangleVertices triangle : inTriangles) {
		vertices.add(triangle.v0);
		vertices.add(triangle.v1);
		vertices.add(triangle.v2);
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
		convertedBvh.triangles.add(inTriangles[sbvh.getTriIndices().get(i)]);
		convertedBvh.triangleDatas.add(inTriangleDatas[sbvh.getTriIndices().get(i)]);
	}

	uint32_t currentTriangleIndex = 0;
	convertRecursively(convertedBvh, currentTriangleIndex, sbvh.getRoot());

	return convertedBvh; // Copy will be optimized away
}

} // namespace phe
