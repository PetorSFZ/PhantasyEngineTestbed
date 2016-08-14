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

	std::string path = std::string(basePath) + fileName;
	std::vector<shape_t> shapes;
	std::vector<material_t> materials;
	std::string error;
	bool success = tinyobj::LoadObj(shapes, materials, error, path.c_str(), basePath);

	if (!success) {
		printErrorMessage("Failed loading model %s, error: %s", fileName, error.c_str());
		return DynArray<Renderable>();
	}
	if (shapes.size() == 0) {
		printErrorMessage("Model %s has no shapes", fileName);
		return DynArray<Renderable>();
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

	DynArray<Renderable> tmpDyn;
	tmpDyn.add(std::move(tmp));
	return std::move(tmpDyn);
}

} // namespace sfz
