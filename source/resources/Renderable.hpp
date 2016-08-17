// Copyright (c) Peter Hillerström (skipifzero.com, peter@hstroem.se)

#pragma once

#include <sfz/containers/DynArray.hpp>

#include "resources/GLMaterial.hpp"
#include "resources/GLModel.hpp"
#include "resources/RawGeometry.hpp"
#include "resources/RawMaterial.hpp"

namespace sfz {

// Renderable class
// ------------------------------------------------------------------------------------------------

class Renderable final {
public:
	// Members
	// --------------------------------------------------------------------------------------------

	RawGeometry geometry;
	RawMaterial material;
	GLModel glModel;
	GLMaterial glMaterial;

	// Constructors & destructors
	// --------------------------------------------------------------------------------------------

	Renderable() noexcept = default;
	Renderable(const Renderable&) = delete;
	Renderable& operator= (const Renderable&) = delete;
	Renderable(Renderable&&) noexcept = default;
	Renderable& operator= (Renderable&&) noexcept = default;
	~Renderable() noexcept = default;
};

// Renderable creation functions
// ------------------------------------------------------------------------------------------------

Renderable tinyObjLoadRenderable(const char* basePath, const char* fileName) noexcept;

/// Specialized function to load sponza (with pbr textures) using tinyObjLoader
DynArray<Renderable> tinyObjLoadSponza(const char* basePath, const char* fileName) noexcept;

DynArray<Renderable> assimpLoadSponza(const char* basePath, const char* fileName) noexcept;

} // namespace sfz
