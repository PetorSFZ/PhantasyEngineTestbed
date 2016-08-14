// Copyright (c) Peter Hillerström (skipifzero.com, peter@hstroem.se)

#include "resources/Renderable.hpp"

#include <tiny_obj_loader.h>

namespace sfz {

// Renderable creation functions
// ------------------------------------------------------------------------------------------------

Renderable tinyObjLoadRenderable(const char* basePath, const char* fileName) noexcept
{
	using tinyobj::shape_t;
	using tinyobj::material_t;

	std::string path = std::string(basePath) + fileName;
	std::vector<shape_t> shapes;
	std::vector<material_t> materials;
	std::string error;
	bool success = tinyobj::LoadObj(shapes, materials, error, path.c_str(), basePath);

	if (!success) {
		printErrorMessage("Failed loading model %s, error: %s", fileName, error.c_str());
		return Renderable();
	}
	if (shapes.size() == 0) {
		printErrorMessage("Model %s has no shapes", fileName);
		return Renderable();
	}

	shape_t shape = shapes[0];
	
	// Calculate number of vertices and create default vertices (all 0)
	uint32_t numVertices = (uint32_t)std::max(shape.mesh.positions.size() / 3,
	                                 std::max(shape.mesh.normals.size() / 3,
	                                 shape.mesh.texcoords.size() / 2));
	Renderable tmp;
	tmp.geometry.vertices = DynArray<Vertex>(numVertices, numVertices);

	// Fill vertices with positions
	for (size_t i = 0; i < shape.mesh.positions.size() / 3; i++) {
		tmp.geometry.vertices[uint32_t(i)].pos = vec3(&shape.mesh.positions[i * 3]);
	}

	// Fill vertices with normals
	for (size_t i = 0; i < shape.mesh.normals.size() / 3; i++) {
		tmp.geometry.vertices[uint32_t(i)].normal = vec3(&shape.mesh.normals[i * 3]);
	}

	// Fill vertices with uv coordinates
	for (size_t i = 0; i < shape.mesh.texcoords.size() / 2; i++) {
		tmp.geometry.vertices[uint32_t(i)].uv = vec2(&shape.mesh.texcoords[i * 2]);
	}

	// Create indices
	tmp.geometry.indices.add(&shape.mesh.indices[0], (uint32_t)shape.mesh.indices.size());

	// Load geometry into OpenGL
	tmp.glModel.load(tmp.geometry);

	return std::move(tmp);
}

DynArray<Renderable> tinyObjLoadSponza(const char* basePath, const char* fileName) noexcept
{
	using tinyobj::shape_t;
	using tinyobj::material_t;

	// Extract path and modelBasePath from the given parameters
	std::string path = std::string(basePath) + fileName;
	size_t lastDirOffs1 = path.rfind('/');
	size_t lastDirOffs2 = path.rfind('\\');
	size_t lastDirOffs = ~size_t(0);
	if (lastDirOffs1 == std::string::npos && lastDirOffs2 == std::string::npos) {
		sfz::error("tinObjLoader: Bad path");
	}
	else if (lastDirOffs1 == std::string::npos) {
		lastDirOffs = lastDirOffs2;
	}
	else if (lastDirOffs2 == std::string::npos) {
		lastDirOffs = lastDirOffs1;
	}
	else {
		lastDirOffs = std::max(lastDirOffs1, lastDirOffs2);
	}
	if (lastDirOffs == ~size_t(0)) {
		sfz::error("tinyObjLoader: Bad path");
	}
	std::string modelBasePath = path;
	modelBasePath.erase(lastDirOffs + 1, modelBasePath.size() - lastDirOffs - 1);
	
	std::vector<shape_t> shapes;
	std::vector<material_t> materials;
	std::string error;
	bool success = tinyobj::LoadObj(shapes, materials, error, path.c_str(), modelBasePath.c_str());

	if (!success) {
		printErrorMessage("Failed loading model %s, error: %s", fileName, error.c_str());
		return DynArray<Renderable>();
	}
	if (shapes.size() == 0) {
		printErrorMessage("Model %s has no shapes", fileName);
		return DynArray<Renderable>();
	}
	
	// Debug print material properties
	for (size_t i = 0; i < materials.size(); i++) {
		printf("material[%ld].name = %s\n", i, materials[i].name.c_str());
		printf("  material.Ka = (%f, %f, %f)\n", materials[i].ambient[0], materials[i].ambient[1], materials[i].ambient[2]);
		printf("  material.Kd = (%f, %f, %f)\n", materials[i].diffuse[0], materials[i].diffuse[1], materials[i].diffuse[2]);
		printf("  material.Ks = (%f, %f, %f)\n", materials[i].specular[0], materials[i].specular[1], materials[i].specular[2]);
		printf("  material.Tr = (%f, %f, %f)\n", materials[i].transmittance[0], materials[i].transmittance[1], materials[i].transmittance[2]);
		printf("  material.Ke = (%f, %f, %f)\n", materials[i].emission[0], materials[i].emission[1], materials[i].emission[2]);
		printf("  material.Ns = %f\n", materials[i].shininess);
		printf("  material.Ni = %f\n", materials[i].ior);
		printf("  material.dissolve = %f\n", materials[i].dissolve);
		printf("  material.illum = %d\n", materials[i].illum);
		printf("  material.map_Ka = %s\n", materials[i].ambient_texname.c_str());
		printf("  material.map_Kd = %s\n", materials[i].diffuse_texname.c_str());
		printf("  material.map_Ks = %s\n", materials[i].specular_texname.c_str());
		printf("  material.map_Ns = %s\n", materials[i].bump_texname.c_str()); // normale_texname???
		auto it = materials[i].unknown_parameter.begin();
		auto itEnd = materials[i].unknown_parameter.end();
		for (; it != itEnd; it++) {
			printf("  material.%s = %s\n", it->first.c_str(), it->second.c_str());
		}
		printf("\n");
	}

	DynArray<Renderable> renderables = DynArray<Renderable>(0, uint32_t(shapes.size()));
	//for (size_t i = 0; i < shapes.size(); i++) {
	for (size_t i = 0; i < 2; i++) {
		shape_t& shape = shapes[i];
		Renderable tmp;

		sfz_assert_debug(shape.mesh.material_ids.size() > 0);
		int materialId = shape.mesh.material_ids[0];
		for (size_t j = 0; j < shape.mesh.material_ids.size(); j++) {
		//for (size_t j = 0; j < 2; j++) {
			printf("Material %u, id = %i\n", uint32_t(j), shape.mesh.material_ids[j]);
			int matId2 = shape.mesh.material_ids[j];
			if (materialId != matId2) {
				sfz::error("tinyObjLoader: Invalid shape, has multiple materials");
			}
		}
		printf("\n");
		if (materialId == -1) {
			sfz::error("tinyObjLoader: Invalid shape, has no material");
		}


		/*printf("Shape: %u\n", uint32_t(i));
		for (size_t j = 0; j < shape.mesh.material_ids.size(); j++) {
			printf("Material %u, id = %i\n", uint32_t(j), shape.mesh.material_ids[j]);
		}
		printf("\n");*/

		//materials[0].
		//shape.mesh.

		// Calculate number of vertices and create default vertices (all 0)
		uint32_t numVertices = (uint32_t)std::max(shape.mesh.positions.size() / 3,
		                                 std::max(shape.mesh.normals.size() / 3,
		                                 shape.mesh.texcoords.size() / 2));

		tmp.geometry.vertices = DynArray<Vertex>(numVertices, numVertices);

		// Fill vertices with positions
		for (size_t i = 0; i < shape.mesh.positions.size() / 3; i++) {
			tmp.geometry.vertices[uint32_t(i)].pos = vec3(&shape.mesh.positions[i * 3]);
		}

		// Fill vertices with normals
		for (size_t i = 0; i < shape.mesh.normals.size() / 3; i++) {
			tmp.geometry.vertices[uint32_t(i)].normal = vec3(&shape.mesh.normals[i * 3]);
		}

		// Fill vertices with uv coordinates
		for (size_t i = 0; i < shape.mesh.texcoords.size() / 2; i++) {
			tmp.geometry.vertices[uint32_t(i)].uv = vec2(&shape.mesh.texcoords[i * 2]);
		}

		// Create indices
		tmp.geometry.indices.add(&shape.mesh.indices[0], (uint32_t)shape.mesh.indices.size());

		// Load geometry into OpenGL
		tmp.glModel.load(tmp.geometry);

		renderables.add(std::move(tmp));
	}

	return std::move(renderables);
}

} // namespace sfz
