#pragma once

#include <sfz/math/Matrix.hpp>

#include "phantasy_engine/level/Level.hpp"


namespace phe {

using namespace sfz;

uint32_t loadDynObject(const char* basePath, const char* fileName, Level& level, const mat4& modelMatrix);
uint32_t loadDynObject(const char* basePath, const char* fileName, Level& level);

uint32_t loadDynObjectCustomMaterial(const char* basePath, const char* fileName, Level& level, vec3& albedo, float roughness, float metallic, const mat4& modelMatrix);
uint32_t loadDynObjectCustomMaterial(const char* basePath, const char* fileName, Level& level, vec3& albedo, float roughness, float metallic);

}
