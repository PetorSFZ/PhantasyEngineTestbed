cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project("Phantasy Engine - Testbed")

# Sets build type to release if no build type is specified in a single-configuration generator.
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
if(MSVC)
	# MSVC flags
	set(CMAKE_CXX_FLAGS "/Wall /Zi /EHsc /wd4201 /wd4350 /wd4514 /wd4571 /wd4647 /wd4668 /wd4710 /wd4711 /wd4820 /D_CRT_SECURE_NO_WARNINGS /MD")
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

# Assimp
add_subdirectory(${EXTERNALS_DIR}/assimp)

# stb_image
add_subdirectory(${EXTERNALS_DIR}/stb_image)

# Cuda
if(CUDA_TRACER)
	find_package(CUDA REQUIRED)
	add_definitions(-DCUDA_TRACER_AVAILABLE)

	set(CUDA_NVCC_FLAGS
		${CUDA_NVCC_FLAGS}
		-O3)
endif()

# Include directories
include_directories(
	${INCLUDE_DIR}
	${SFZ_CORE_INCLUDE_DIRS}
	${ASSIMP_INCLUDE_DIRS}
	${STB_IMAGE_INCLUDE_DIRS}
)

set(SOURCE_ALL_FILES
	${SOURCE_DIR}/Main.cpp

	${INCLUDE_DIR}/Config.hpp
	${INCLUDE_DIR}/config/GlobalConfig.hpp
	 ${SOURCE_DIR}/config/GlobalConfig.cpp
	${INCLUDE_DIR}/config/Setting.hpp
	 ${SOURCE_DIR}/config/Setting.cpp

	${INCLUDE_DIR}/Renderers.hpp
	${INCLUDE_DIR}/renderers/BaseRenderer.hpp
	${INCLUDE_DIR}/renderers/DeferredRenderer.hpp
	 ${SOURCE_DIR}/renderers/DeferredRenderer.cpp
	${INCLUDE_DIR}/renderers/ViewFrustum.hpp
	 ${SOURCE_DIR}/renderers/ViewFrustum.cpp
	${INCLUDE_DIR}/renderers/FullscreenTriangle.hpp
	 ${SOURCE_DIR}/renderers/FullscreenTriangle.cpp

	${INCLUDE_DIR}/renderers/cpu_ray_tracer/CPURayTracerRenderer.hpp
	 ${SOURCE_DIR}/renderers/cpu_ray_tracer/CPURayTracerRenderer.cpp

	${INCLUDE_DIR}/Resources.hpp
	${INCLUDE_DIR}/resources/GLModel.hpp
	 ${SOURCE_DIR}/resources/GLModel.cpp
	${INCLUDE_DIR}/resources/GLTexture.hpp
	 ${SOURCE_DIR}/resources/GLTexture.cpp
	${INCLUDE_DIR}/resources/RawGeometry.hpp
	${INCLUDE_DIR}/resources/RawImage.hpp
	 ${SOURCE_DIR}/resources/RawImage.cpp
	${INCLUDE_DIR}/resources/Renderable.hpp
	 ${SOURCE_DIR}/resources/Renderable.cpp

	${INCLUDE_DIR}/Screens.hpp
	${INCLUDE_DIR}/screens/GameScreen.hpp
	 ${SOURCE_DIR}/screens/GameScreen.cpp
	${INCLUDE_DIR}/screens/LevelSelectScreen.hpp
	 ${SOURCE_DIR}/screens/LevelSelectScreen.cpp

	${INCLUDE_DIR}/level/PointLight.hpp
	${INCLUDE_DIR}/level/Scene.hpp

	${INCLUDE_DIR}/util/IOUtil.hpp
	 ${SOURCE_DIR}/util/IOUtil.cpp)

if(CUDA_TRACER)
set(SOURCE_ALL_FILES ${SOURCE_ALL_FILES}
	${INCLUDE_DIR}/renderers/cuda_ray_tracer/CUDAGLTexture.hpp
	 ${SOURCE_DIR}/renderers/cuda_ray_tracer/CUDAGLTexture.cpp
	${INCLUDE_DIR}/renderers/cuda_ray_tracer/CUDARayTracerRenderer.hpp
	 ${SOURCE_DIR}/renderers/cuda_ray_tracer/CUDARayTracerRenderer.cpp
	${INCLUDE_DIR}/renderers/cuda_ray_tracer/CUDATest.cuh
	 ${SOURCE_DIR}/renderers/cuda_ray_tracer/CUDATest.cu
	${INCLUDE_DIR}/renderers/cuda_ray_tracer/errorCheck.cuh)
endif()

# Automatically set source group to correspond to directory structure.
# From Stack Overflow post by Florian:
# http://stackoverflow.com/a/31423421
function(set_directory_source_group)
	foreach(_source IN ITEMS ${ARGN})
		if (IS_ABSOLUTE "${_source}")
			file(RELATIVE_PATH _source_rel "${CMAKE_CURRENT_SOURCE_DIR}" "${_source}")
		else()
			set(source_rel "${_source}")
		endif()
		get_filename_component(_source_path "${_source_rel}" PATH)
		string(REPLACE "/" "\\" _source_path_msvc "${_source_path}")
		source_group("${_source_path_msvc}" FILES "${_source}")
	endforeach()
endfunction(set_directory_source_group)

set_directory_source_group(${SOURCE_ALL_FILES})

# Main executable
if(CUDA_TRACER)
	cuda_add_executable(PhantasyEngineTestbed ${SOURCE_ALL_FILES})
else()
	add_executable(PhantasyEngineTestbed ${SOURCE_ALL_FILES})
endif()

# Linking libraries to main executable
target_link_libraries(
	PhantasyEngineTestbed

	${SFZ_CORE_LIBRARIES}
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

if(UNIX)
	add_custom_command(TARGET PhantasyEngineTestbed PRE_BUILD
			COMMAND ${CMAKE_COMMAND} -E create_symlink
			\"${PROJECT_SOURCE_DIR}/resources\"
			\"$<TARGET_FILE_DIR:PhantasyEngineTestbed>/resources\")
endif()
