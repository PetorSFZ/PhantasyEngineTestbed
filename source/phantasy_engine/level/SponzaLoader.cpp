// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#include "phantasy_engine/level/SponzaLoader.hpp"

#include <string>

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include <sfz/containers/DynString.hpp>
#include <sfz/containers/HashMap.hpp>

#include "phantasy_engine/level/StaticScene.hpp"
#include "phantasy_engine/util/IOUtil.hpp"

namespace phe {

using namespace sfz;

static vec3 toSFZ(const aiVector3D& v)
{
	return vec3(v.x, v.y, v.z);
}

static void processNode(const char* basePath, Level& level,
                        const aiScene* scene, aiNode* node, const mat4& modelMatrix, const mat4& normalMatrix) noexcept
{
	aiString tmpPath;

	std::hash<std::string> hashFn;

	// Process all meshes in current node
	for (uint32_t meshIndex = 0; meshIndex < node->mNumMeshes; meshIndex++) {
		
		const aiMesh* mesh = scene->mMeshes[node->mMeshes[meshIndex]];
		RawMesh meshTmp;
		Material materialTmp;

		// Allocate memory for vertices
		meshTmp.vertices = DynArray<Vertex>(mesh->mNumVertices, mesh->mNumVertices);

		// Fill vertices with positions, normals and uv coordinates
		sfz_assert_debug(mesh->HasPositions());
		sfz_assert_debug(mesh->HasNormals());
		for (uint32_t i = 0; i < mesh->mNumVertices; i++) {
			Vertex& v = meshTmp.vertices[i];
			v.pos = sfz::transformPoint(modelMatrix, toSFZ(mesh->mVertices[i]));
			v.normal = sfz::transformDir(normalMatrix, toSFZ(mesh->mNormals[i]));
			if (mesh->mTextureCoords[0] != nullptr) {
				v.uv = toSFZ(mesh->mTextureCoords[0][i]).xy;
			}
			else {
				v.uv = vec2(0.0f);
			}
		}

		// Fill geometry with indices
		meshTmp.indices.setCapacity(mesh->mNumFaces * 3u);
		for (uint32_t i = 0; i < mesh->mNumFaces; i++) {
			const aiFace& face = mesh->mFaces[i];
			meshTmp.indices.add(face.mIndices, face.mNumIndices);
		}

		// Retrieve mesh's material
		const aiMaterial* mat = scene->mMaterials[mesh->mMaterialIndex];

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

		// Add material index
		uint16_t nextMaterialIndex = uint16_t(level.materials.size());
		meshTmp.materialIndices = DynArray<uint16_t>(meshTmp.vertices.size(), nextMaterialIndex, 0u);

		// Add the mesh and material
		level.staticScene.meshes.add(std::move(meshTmp));
		level.materials.add(materialTmp);
	}

	// Process all children
	for (uint32_t i = 0; i < node->mNumChildren; i++) {
		processNode(basePath, level, scene, node->mChildren[i], modelMatrix, normalMatrix);
	}
}

void loadStaticSceneSponza(const char* basePath, const char* fileName, Level& level,
                           const mat4& modelMatrix) noexcept
{
	// Create full path
	size_t basePathLen = std::strlen(basePath);
	size_t fileNameLen = std::strlen(fileName);
	DynString path("", basePathLen + fileNameLen + 2);
	path.printf("%s%s", basePath, fileName);
	if (path.size() < 1) {
		sfz::printErrorMessage("Failed to load model, empty path");
		return;
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
		return;
	}

	// Load model through Assimp
	Assimp::Importer importer;
	const aiScene* scene = importer.ReadFile(path.str(), aiProcessPreset_TargetRealtime_Quality | aiProcess_FlipUVs);
	if (scene == nullptr || scene->mFlags == AI_SCENE_FLAGS_INCOMPLETE || scene->mRootNode == nullptr) {
		sfz::printErrorMessage("Failed to load model \"%s\", error: %s", fileName, importer.GetErrorString());
		return;
	}

	// Clear existing static scene
	level.staticScene.meshes.clear();
	level.staticScene.sphereLights.clear();

	// Process tree, filling up the list of renderable components along the way
	const mat4 normalMatrix = inverse(transpose(modelMatrix));
	processNode(realBasePath.str(), level, scene, scene->mRootNode, modelMatrix, normalMatrix);
}

} // namespace phe
