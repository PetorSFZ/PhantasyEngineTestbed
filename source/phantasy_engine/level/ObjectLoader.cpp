// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#include "ObjectLoader.hpp"

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include <sfz/containers/DynString.hpp>
#include <sfz/math/MatrixSupport.hpp>

#include "phantasy_engine/util/IOUtil.hpp"

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

	const mat4 normalMatrix = inverse(transpose(modelMatrix));

	aiString tmpPath;

	RawMesh mesh;

	std::hash<std::string> hashFn;

	// Process all meshes in current node
	for (uint32_t meshIndex = 0; meshIndex < scene->mNumMeshes; meshIndex++) {

		Material materialTmp;

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
		}

		// Fill geometry with indices
		for (uint32_t i = 0; i < aimesh->mNumFaces; i++) {
			const aiFace& face = aimesh->mFaces[i];
			mesh.indices.add(face.mIndices, face.mNumIndices);
		}

		// Retrieve mesh's material
		const aiMaterial* mat = scene->mMaterials[aimesh->mMaterialIndex];
		
		// Albedo (stored in diffuse for sponza pbr)
		if (mat->GetTextureCount(aiTextureType_DIFFUSE) > 0) {
			sfz_assert_debug(mat->GetTextureCount(aiTextureType_DIFFUSE) == 1);

			tmpPath.Clear();
			mat->GetTexture(aiTextureType_DIFFUSE, 0, &tmpPath);

			size_t hashedString = hashFn(std::string(tmpPath.C_Str()));

			const uint32_t* indexPtr = level.texMapping.get(hashedString);
			if (indexPtr == nullptr) {
				//printf("Loaded albedo texture: %s\n", tmpPath.C_Str());

				const uint32_t nextIndex = level.textures.size();
				level.texMapping.put(hashedString, nextIndex);
				indexPtr = level.texMapping.get(hashedString);

				level.textures.add(loadImage(basePath, convertToOSPath(tmpPath.C_Str()).str()));
			}
			materialTmp.setAlbedoTexIndex(*indexPtr);
		}
		else {
			aiColor3D color(0.0f, 0.0f, 0.0f);
			mat->Get(AI_MATKEY_COLOR_DIFFUSE, color);
			materialTmp.setAlbedoValue(vec4(color.r, color.g, color.b, 1.0f));
		}
		
		// Roughness (stored in map_Ns, specular highlight component)
		if (mat->GetTextureCount(aiTextureType_SHININESS) > 0) {
			sfz_assert_debug(mat->GetTextureCount(aiTextureType_SHININESS) == 1);

			tmpPath.Clear();
			mat->GetTexture(aiTextureType_SHININESS, 0, &tmpPath);

			size_t hashedString = hashFn(std::string(tmpPath.C_Str()));

			const uint32_t* indexPtr = level.texMapping.get(hashedString);
			if (indexPtr == nullptr) {
				//printf("Loaded roughness texture: %s\n", tmpPath.C_Str());

				const uint32_t nextIndex = level.textures.size();
				level.texMapping.put(hashedString, nextIndex);
				indexPtr = level.texMapping.get(hashedString);

				level.textures.add(loadImage(basePath, convertToOSPath(tmpPath.C_Str()).str()));
			}
			materialTmp.setRoughnessTexIndex(*indexPtr);
		}
		else {
			aiColor3D color(0.0f, 0.0f, 0.0f);
			mat->Get(AI_MATKEY_COLOR_SPECULAR, color);
			materialTmp.setRoughnessValue(color.r);
		}

		// Metallic (stored in map_Ka, ambient texture map)
		if (mat->GetTextureCount(aiTextureType_AMBIENT) > 0) {
			sfz_assert_debug(mat->GetTextureCount(aiTextureType_AMBIENT) == 1);

			tmpPath.Clear();
			mat->GetTexture(aiTextureType_AMBIENT, 0, &tmpPath);

			size_t hashedString = hashFn(std::string(tmpPath.C_Str()));

			const uint32_t* indexPtr = level.texMapping.get(hashedString);
			if (indexPtr == nullptr) {
				//printf("Loaded metallic texture: %s\n", tmpPath.C_Str());

				const uint32_t nextIndex = level.textures.size();
				level.texMapping.put(hashedString, nextIndex);
				indexPtr = level.texMapping.get(hashedString);

				level.textures.add(loadImage(basePath, convertToOSPath(tmpPath.C_Str()).str()));
			}
			materialTmp.setMetallicTexIndex(*indexPtr);
		}
		else {
			aiColor3D color(0.0f, 0.0f, 0.0f);
			mat->Get(AI_MATKEY_COLOR_AMBIENT, color);
			materialTmp.setMetallicValue(color.r);
		}

		// Add material index
		uint16_t nextMaterialIndex = uint16_t(level.materials.size());
		mesh.materialIndices.add(DynArray<uint16_t>(aimesh->mNumVertices, nextMaterialIndex, 0u));

		// Add the mesh and material
		level.materials.add(materialTmp);
	}
	mesh.indices.setCapacity(mesh.indices.size() * 3u);

	level.meshes.add(mesh);
	return level.meshes.size() - 1u;
}

uint32_t loadDynObject(const char* basePath, const char* fileName, Level& level)
{
	return loadDynObject(basePath, fileName, level, identityMatrix4<float>());
}

uint32_t loadDynObjectCustomMaterial(const char* basePath, const char* fileName, Level& level, const vec3& albedo, float roughness, float metallic, const mat4& modelMatrix)
{
	uint32_t handle = loadDynObject(basePath, fileName, level, modelMatrix);

	const RawMesh& mesh = level.meshes[handle];

	for (uint32_t matIndex : mesh.materialIndices) {
		level.materials[matIndex].setAlbedoTexIndex(-1);
		level.materials[matIndex].setRoughnessTexIndex(-1);
		level.materials[matIndex].setMetallicTexIndex(-1);
		level.materials[matIndex].setAlbedoValue(vec4(albedo, 1.0));
		level.materials[matIndex].setRoughnessValue(roughness);
		level.materials[matIndex].setMetallicValue(metallic);
	}

	return handle;
}

uint32_t loadDynObjectCustomMaterial(const char* basePath, const char* fileName, Level& level, const vec3& albedo, float roughness, float metallic)
{
	return loadDynObjectCustomMaterial(basePath, fileName, level, albedo, roughness, metallic, identityMatrix4<float>());
}

}