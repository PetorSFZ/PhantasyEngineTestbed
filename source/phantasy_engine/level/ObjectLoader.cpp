// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#include "ObjectLoader.hpp"

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include <sfz/containers/DynString.hpp>
#include <sfz/math/MatrixSupport.hpp>

namespace phe {

using namespace sfz;

static vec3 toSFZ(const aiVector3D& v)
{
	return vec3(v.x, v.y, v.z);
}

uint32_t loadDynObject(const char* basePath, const char* fileName, Level& level, const mat4& modelMatrix)
{
	size_t basePathLen = std::strlen(basePath);
	size_t fileNameLen = std::strlen(fileName);
	DynString path("", basePathLen + fileNameLen + 2);
	path.printf("%s%s", basePath, fileName);
	if (path.size() < 1) {
		sfz::printErrorMessage("Failed to load model, empty path");
		return -1;
	}

	// Get the real base path from the path
	DynString realBasePath(path.str());
	DynArray<char>& internal = realBasePath.internalDynArray();
	for (uint32_t i = internal.size() - 1; i > 0; i--) {
		const char c = internal[i - 1];
		if (c == '\\' || c == '/') {
			internal[i] = '\0';
			internal.setSize(i + 1);
			break;
		}
	}
	if (realBasePath.size() == path.size()) {
		sfz::printErrorMessage("Failed to find real base path, basePath=\"%s\", fileName=\"%s\"", basePath, fileName);
		return -1;
	}

	// Load model through Assimp
	Assimp::Importer importer;
	const aiScene* scene = importer.ReadFile(path.str(), aiProcessPreset_TargetRealtime_Quality | aiProcess_FlipUVs);
	if (scene == nullptr || scene->mFlags == AI_SCENE_FLAGS_INCOMPLETE || scene->mRootNode == nullptr) {
		sfz::printErrorMessage("Failed to load model \"%s\", error: %s", fileName, importer.GetErrorString());
		return -1;
	}

	uint32_t materialIndex = level.materials.size();

	Material material;
	material.setAlbedoValue(vec4(100 / 255.0f, 149 / 255.0f, 237 / 255.0f, 1.0f));
	material.setMetallicValue(1.0f);
	material.setRoughnessValue(0.5f);
	level.materials.add(material);

	const mat4 normalMatrix = inverse(transpose(modelMatrix));

	RawMesh mesh;

	// Process all meshes in current node
	for (uint32_t meshIndex = 0; meshIndex < scene->mNumMeshes; meshIndex++) {

		const aiMesh* aimesh = scene->mMeshes[meshIndex];

		// Allocate memory for vertices
		uint32_t meshVerticesSize = mesh.vertices.size();
		mesh.vertices.add(DynArray<Vertex>(aimesh->mNumVertices, aimesh->mNumVertices));

		// Fill vertices with positions, normals and uv coordinates
		sfz_assert_debug(aimesh->HasPositions());
		sfz_assert_debug(aimesh->HasNormals());
		for (uint32_t i = 0; i < aimesh->mNumVertices; i++) {
			Vertex& v = mesh.vertices[i + meshVerticesSize];
			v.pos = transformPoint(modelMatrix, toSFZ(aimesh->mVertices[i]));
			v.normal = transformDir(normalMatrix, toSFZ(aimesh->mNormals[i]));
			if (aimesh->mTextureCoords[0] != nullptr) {
				v.uv = toSFZ(aimesh->mTextureCoords[0][i]).xy;
			}
			else {
				v.uv = vec2(0.0f);
			}
			mesh.materialIndices.add(materialIndex);
			mesh.materialIndices.add(materialIndex);
			mesh.materialIndices.add(materialIndex);
		}

		// Fill geometry with indices
		for (uint32_t i = 0; i < aimesh->mNumFaces; i++) {
			const aiFace& face = aimesh->mFaces[i];
			mesh.indices.add(face.mIndices, face.mNumIndices);
		}
	}
	mesh.indices.setCapacity(mesh.indices.size() * 3u);

	level.meshes.add(mesh);
	return level.meshes.size() - 1u;
}

uint32_t loadDynObject(const char* basePath, const char* fileName, Level& level)
{
	return loadDynObject(basePath, fileName, level, identityMatrix4<float>());
}

}