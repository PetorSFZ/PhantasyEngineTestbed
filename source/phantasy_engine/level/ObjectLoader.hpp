#pragma once

#include <sfz/math/Matrix.hpp>

#include "phantasy_engine/level/Level.hpp"


namespace phe {

using namespace sfz;

uint32_t loadDynObject(const char* basePath, const char* fileName, Level& level, const mat4& modelMatrix);
uint32_t loadDynObject(const char* basePath, const char* fileName, Level& level);


}
