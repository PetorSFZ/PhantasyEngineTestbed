cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project("Phantasy Engine - Testbed")

# Sets build type to release if no build type is specified in a single-configuration generator.
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
if(MSVC)
	# MSVC flags
	set(CMAKE_CXX_FLAGS "/Wall /Zi /EHsc /wd4201 /wd4350 /wd4514 /wd4571 /wd4647 /wd4668 /wd4710 /wd4711 /wd4820 /D_CRT_SECURE_NO_WARNINGS")
	set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "/O2 /fp:fast /DEBUG /DSFZ_NO_DEBUG")
	set(CMAKE_CXX_FLAGS_RELEASE "/O2 /fp:fast /DSFZ_NO_DEBUG")
	set(CMAKE_CXX_FLAGS_DEBUG "/Od /DEBUG")
else()
	# GCC / Clang flags
	set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wno-c++11-extensions -std=c++11 -pthread")
	set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -ffast-math -g -DSFZ_NO_DEBUG")
	set(CMAKE_CXX_FLAGS_RELEASE "-O3 -ffast-math -DSFZ_NO_DEBUG")
	set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
endif()

# Directories
set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/source)
set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/source)
set(EXTERNALS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/externals)
set(RUNTIME_DIR ${CMAKE_CURRENT_SOURCE_DIR}/runtime)

# sfzCore
set(SFZ_CORE_OPENGL TRUE)
add_subdirectory(${EXTERNALS_DIR}/sfzCore)

# tinyobjloader
add_subdirectory(${EXTERNALS_DIR}/tinyobjloader)

# Assimp
add_subdirectory(${EXTERNALS_DIR}/assimp)

# stb_image
add_subdirectory(${EXTERNALS_DIR}/stb_image)

# Include directories
include_directories(
	${INCLUDE_DIR}
	${SFZ_CORE_INCLUDE_DIRS}
	${TINYOBJLOADER_INCLUDE_DIRS}
	${ASSIMP_INCLUDE_DIRS}
	${STB_IMAGE_INCLUDE_DIRS}
)

set(SOURCE_ROOT_FILES
	${SOURCE_DIR}/Main.cpp)
source_group(root FILES ${SOURCE_ROOT_FILES})

set(SOURCE_RENDERERS_FILES
	${INCLUDE_DIR}/Renderers.hpp
	${INCLUDE_DIR}/renderers/BaseRenderer.hpp
	${INCLUDE_DIR}/renderers/DeferredRenderer.hpp
	 ${SOURCE_DIR}/renderers/DeferredRenderer.cpp)
source_group(renderers FILES ${SOURCE_RENDERERS_FILES})

set(SOURCE_RESOURCES_FILES
	${INCLUDE_DIR}/Resources.hpp
	${INCLUDE_DIR}/resources/GLModel.hpp
	 ${SOURCE_DIR}/resources/GLModel.cpp
	${INCLUDE_DIR}/resources/GLTexture.hpp
	 ${SOURCE_DIR}/resources/GLTexture.cpp
	${INCLUDE_DIR}/resources/RawGeometry.hpp
	${INCLUDE_DIR}/resources/RawImage.hpp
	 ${SOURCE_DIR}/resources/RawImage.cpp
	${INCLUDE_DIR}/resources/Renderable.hpp
	 ${SOURCE_DIR}/resources/Renderable.cpp)
source_group(resources FILES ${SOURCE_RESOURCES_FILES})

set(SOURCE_SCREENS_FILES
	${INCLUDE_DIR}/Screens.hpp
	${INCLUDE_DIR}/screens/GameScreen.hpp
	 ${SOURCE_DIR}/screens/GameScreen.cpp
	${INCLUDE_DIR}/screens/LevelSelectScreen.hpp
	 ${SOURCE_DIR}/screens/LevelSelectScreen.cpp)
source_group(screens FILES ${SOURCE_SCREENS_FILES})

set(SOURCE_ALL_FILES
	${SOURCE_ROOT_FILES}
	${SOURCE_RENDERERS_FILES}
	${SOURCE_RESOURCES_FILES}
	${SOURCE_SCREENS_FILES})

# Main executable
add_executable(PhantasyEngineTestbed ${SOURCE_ALL_FILES})

# Linking libraries to main executable
target_link_libraries(
	PhantasyEngineTestbed

	${SFZ_CORE_LIBRARIES}
	${TINYOBJLOADER_LIBRARIES}
	${ASSIMP_LIBRARIES}
	${STB_IMAGE_LIBRARIES}
)

# Windows specific file copying
if(MSVC)
	# Create assets symlinks batch file
	file(COPY ${RUNTIME_DIR}/windows/CREATE_RESOURCES_SYMLINKS.BAT DESTINATION ${CMAKE_BINARY_DIR})

	# Runtime DLLs
	file(COPY ${RUNTIME_DIR}/windows/SDL2.dll DESTINATION ${CMAKE_BINARY_DIR}/Debug)
	file(COPY ${RUNTIME_DIR}/windows/SDL2.dll DESTINATION ${CMAKE_BINARY_DIR}/RelWithDebInfo)
	file(COPY ${RUNTIME_DIR}/windows/SDL2.dll DESTINATION ${CMAKE_BINARY_DIR}/Release)

	file(COPY ${RUNTIME_DIR}/windows/SDL2_mixer.dll DESTINATION ${CMAKE_BINARY_DIR}/Debug)
	file(COPY ${RUNTIME_DIR}/windows/SDL2_mixer.dll DESTINATION ${CMAKE_BINARY_DIR}/RelWithDebInfo)
	file(COPY ${RUNTIME_DIR}/windows/SDL2_mixer.dll DESTINATION ${CMAKE_BINARY_DIR}/Release)
endif()
