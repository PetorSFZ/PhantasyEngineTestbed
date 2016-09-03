// Copyright (c) Peter Hillerström (skipifzero.com, peter@hstroem.se)

#include "resources/Renderable.hpp"

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include <sfz/containers/DynString.hpp>
#include <sfz/containers/HashMap.hpp>

#include "util/IOUtil.hpp"

namespace sfz {

// Renderable creation functions
// ------------------------------------------------------------------------------------------------

static vec3 toSFZ(const aiVector3D& v)
{
	return vec3(v.x, v.y, v.z);
}

static void processNode(const char* basePath, Renderable& renderable,
                        HashMap<std::string,uint32_t>& texMapping,
                        const aiScene* scene, aiNode* node) noexcept
{
	aiString tmpPath;

	// Process all meshes in current node
	for (uint32_t meshIndex = 0; meshIndex < node->mNumMeshes; meshIndex++) {
		const aiMesh* mesh = scene->mMeshes[node->mMeshes[meshIndex]];
		RenderableComponent tmp;

		// Allocate memory for vertices
		tmp.geometry.vertices = DynArray<Vertex>(mesh->mNumVertices, mesh->mNumVertices);// .setCapacity(mesh->mNumVertices);

		// Fill vertices with positions, normals and uv coordinates
		sfz_assert_debug(mesh->HasPositions());
		sfz_assert_debug(mesh->HasNormals());
		for (uint32_t i = 0; i < mesh->mNumVertices; i++) {
			Vertex& v = tmp.geometry.vertices[i];
			v.pos = toSFZ(mesh->mVertices[i]);
			v.normal = toSFZ(mesh->mNormals[i]);
			if (mesh->mTextureCoords[0] != nullptr) {
				v.uv = toSFZ(mesh->mTextureCoords[0][i]).xy;
			} else {
				v.uv = vec2(0.0f);
			}
		}

		// Fill geometry with indices
		tmp.geometry.indices.setCapacity(mesh->mNumFaces * 3);
		for (uint32_t i = 0; i < mesh->mNumFaces; i++) {
			const aiFace& face = mesh->mFaces[i];
			tmp.geometry.indices.add(face.mIndices, face.mNumIndices);
		}

		// Load geometry into OpenGL
		tmp.glModel.load(tmp.geometry);

		// Retrieve mesh's material
		const aiMaterial* mat = scene->mMaterials[mesh->mMaterialIndex];

		// Albedo (stored in diffuse for sponza pbr)
		if (mat->GetTextureCount(aiTextureType_DIFFUSE) > 0) {
			sfz_assert_debug(mat->GetTextureCount(aiTextureType_DIFFUSE) == 1);
			
			tmpPath.Clear();
			mat->GetTexture(aiTextureType_DIFFUSE, 0, &tmpPath);
			
			const uint32_t* indexPtr = texMapping.get(tmpPath.C_Str());
			if (indexPtr == nullptr) {
				//printf("Loaded albedo texture: %s\n", tmpPath.C_Str());

				const uint32_t nextIndex = renderable.textures.size();
				texMapping.put(tmpPath.C_Str(), nextIndex);
				indexPtr = texMapping.get(tmpPath.C_Str());

				renderable.images.add(loadImage(basePath, convertToOSPath(tmpPath.C_Str()).str()));
				renderable.images.last().convertToLinear();
				renderable.textures.add(GLTexture(renderable.images[nextIndex]));
				sfz_assert_debug(renderable.textures.last().isValid());
			}
			tmp.material.albedoIndex = *indexPtr;
		}

		// Roughness (stored in map_Ns, specular highlight component)
		if (mat->GetTextureCount(aiTextureType_SHININESS) > 0) {
			sfz_assert_debug(mat->GetTextureCount(aiTextureType_SHININESS) == 1);

			tmpPath.Clear();
			mat->GetTexture(aiTextureType_SHININESS, 0, &tmpPath);

			const uint32_t* indexPtr = texMapping.get(tmpPath.C_Str());
			if (indexPtr == nullptr) {
				//printf("Loaded roughness texture: %s\n", tmpPath.C_Str());

				const uint32_t nextIndex = renderable.textures.size();
				texMapping.put(tmpPath.C_Str(), nextIndex);
				indexPtr = texMapping.get(tmpPath.C_Str());

				renderable.images.add(loadImage(basePath, convertToOSPath(tmpPath.C_Str()).str()));
				//renderable.images.last().convertToLinear();
				renderable.textures.add(GLTexture(renderable.images[nextIndex]));
				sfz_assert_debug(renderable.textures.last().isValid());
			}
			tmp.material.roughnessIndex = *indexPtr;
		}

		// Metallic (stored in map_Ka, ambient texture map)
		if (mat->GetTextureCount(aiTextureType_AMBIENT) > 0) {
			sfz_assert_debug(mat->GetTextureCount(aiTextureType_AMBIENT) == 1);

			tmpPath.Clear();
			mat->GetTexture(aiTextureType_AMBIENT, 0, &tmpPath);

			const uint32_t* indexPtr = texMapping.get(tmpPath.C_Str());
			if (indexPtr == nullptr) {
				//printf("Loaded metallic texture: %s\n", tmpPath.C_Str());

				const uint32_t nextIndex = renderable.textures.size();
				texMapping.put(tmpPath.C_Str(), nextIndex);
				indexPtr = texMapping.get(tmpPath.C_Str());

				renderable.images.add(loadImage(basePath, convertToOSPath(tmpPath.C_Str()).str()));
				//renderable.images.last().convertToLinear();
				renderable.textures.add(GLTexture(renderable.images[nextIndex]));
				sfz_assert_debug(renderable.textures.last().isValid());
			}
			tmp.material.metallicIndex = *indexPtr;
		}

		// Add component to Renderable
		renderable.components.add(std::move(tmp));
	}

	// Process all children
	for (uint32_t i = 0; i < node->mNumChildren; i++) {
		processNode(basePath, renderable, texMapping, scene, node->mChildren[i]);
	}
}

Renderable assimpLoadSponza(const char* basePath, const char* fileName) noexcept
{
	// Create full path
	size_t basePathLen = std::strlen(basePath);
	size_t fileNameLen = std::strlen(fileName);
	DynString path("", basePathLen + fileNameLen + 2);
	path.printf("%s%s", basePath, fileName);
	if (path.size() < 1) {
		printErrorMessage("Failed to load model, empty path");
		return Renderable();
	}

	// Get the real base path from the path
	DynString realBasePath(path.str());
	DynArray<char>& internal = realBasePath.internalDynArray();
	for (uint32_t i = internal.size() - 1; i > 0; i--) {
		const char c = internal[i-1];
		if (c == '\\' || c == '/') {
			internal[i] = '\0';
			internal.setSize(i + 1);
			break;
		}
	}
	if (realBasePath.size() == path.size()) {
		printErrorMessage("Failed to find real base path, basePath=\"%s\", fileName=\"%s\"", basePath, fileName);
		return Renderable();
	}

	// Load model through Assimp
	Assimp::Importer importer;
	const aiScene* scene = importer.ReadFile(path.str(), aiProcessPreset_TargetRealtime_Quality | aiProcess_FlipUVs);
	if (scene == nullptr || scene->mFlags == AI_SCENE_FLAGS_INCOMPLETE || scene->mRootNode == nullptr) {
		printErrorMessage("Failed to load model \"%s\", error: %s", fileName, importer.GetErrorString());
		return Renderable();
	}
	
	// Process tree, creating renderables along the way
	Renderable renderable;
	HashMap<std::string,uint32_t> texMapping(uint32_t(scene->mNumTextures));
	processNode(realBasePath.str(), renderable, texMapping, scene, scene->mRootNode);
	return std::move(renderable);
}

} // namespace sfz
