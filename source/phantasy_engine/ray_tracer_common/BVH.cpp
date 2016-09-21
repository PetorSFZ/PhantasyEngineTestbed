// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#include "phantasy_engine/ray_tracer_common/BVH.hpp"

#include "phantasy_engine/sbvh-stuff/SplitBVHBuilder.h"
#include "phantasy_engine/sbvh-stuff/BVH.h"
#include "phantasy_engine/sbvh-stuff/Scene.h"

namespace phe {

void convertRecursively(phe::BVH& bvh, uint32_t& currentTriangleIndex, const nv::BVHNode* node)
{
	uint32_t nodeIndex = bvh.nodes.size();

	BVHNode newNode;
	newNode.min = node->m_bounds.min();
	newNode.max = node->m_bounds.max();
	bvh.nodes.add(newNode);

	if (node->isLeaf()) {
		bvh.nodes[nodeIndex].setLeaf(node->getNumTriangles(), currentTriangleIndex);
		currentTriangleIndex += node->getNumTriangles();
	}
	else {
		uint32_t left, right;

		right = bvh.nodes.size();
		convertRecursively(bvh, currentTriangleIndex, node->getChildNode(1));

		left = bvh.nodes.size();
		convertRecursively(bvh, currentTriangleIndex, node->getChildNode(0));

		bvh.nodes[nodeIndex].setInner(left, right);
	}
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
