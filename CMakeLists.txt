# Functions
# ------------------------------------------------------------------------------------------------

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

# Compiler flags and directories
# ------------------------------------------------------------------------------------------------

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
set(TESTS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tests)
set(EXTERNALS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/externals)
set(RUNTIME_DIR ${CMAKE_CURRENT_SOURCE_DIR}/runtime)

# Dependencies
# ------------------------------------------------------------------------------------------------

# sfzCore
set(SFZ_CORE_OPENGL TRUE)
add_subdirectory(${EXTERNALS_DIR}/sfzCore)

# Assimp
add_subdirectory(${EXTERNALS_DIR}/assimp)

# stb_image
add_subdirectory(${EXTERNALS_DIR}/stb_image)

# imgui
add_subdirectory(${EXTERNALS_DIR}/imgui)

# Include directories
include_directories(
	${INCLUDE_DIR}
	${SFZ_CORE_INCLUDE_DIRS}
	${ASSIMP_INCLUDE_DIRS}
	${STB_IMAGE_INCLUDE_DIRS}
	${IMGUI_INCLUDE_DIRS}
)

# Phantasy Engine Library
# ------------------------------------------------------------------------------------------------

set(PH_SOURCE_DIR ${SOURCE_DIR}/phantasy_engine)
set(PH_INCLUDE_DIR ${INCLUDE_DIR}/phantasy_engine)

set(PHANTASY_ENGINE_LIB_FILES
	${PH_INCLUDE_DIR}/PhantasyEngine.hpp
	 ${PH_SOURCE_DIR}/PhantasyEngine.cpp

	${PH_INCLUDE_DIR}/Config.hpp
	${PH_INCLUDE_DIR}/config/GlobalConfig.hpp
	 ${PH_SOURCE_DIR}/config/GlobalConfig.cpp
	${PH_INCLUDE_DIR}/config/Setting.hpp
	 ${PH_SOURCE_DIR}/config/Setting.cpp

	${PH_INCLUDE_DIR}/ray_tracer_common/Ray.hpp

	${PH_INCLUDE_DIR}/Renderers.hpp
	${PH_INCLUDE_DIR}/renderers/BaseRenderer.hpp
	${PH_INCLUDE_DIR}/renderers/DeferredRenderer.hpp
	 ${PH_SOURCE_DIR}/renderers/DeferredRenderer.cpp
	${PH_INCLUDE_DIR}/renderers/ViewFrustum.hpp
	 ${PH_SOURCE_DIR}/renderers/ViewFrustum.cpp
	${PH_INCLUDE_DIR}/renderers/FullscreenTriangle.hpp
	 ${PH_SOURCE_DIR}/renderers/FullscreenTriangle.cpp

	${PH_INCLUDE_DIR}/renderers/cpu_ray_tracer/AabbTree.hpp
	 ${PH_SOURCE_DIR}/renderers/cpu_ray_tracer/AabbTree.cpp
	${PH_INCLUDE_DIR}/renderers/cpu_ray_tracer/BVH.hpp
	 ${PH_SOURCE_DIR}/renderers/cpu_ray_tracer/BVH.cpp
	${PH_INCLUDE_DIR}/renderers/cpu_ray_tracer/CPURayTracerRenderer.hpp
	 ${PH_SOURCE_DIR}/renderers/cpu_ray_tracer/CPURayTracerRenderer.cpp
	${PH_INCLUDE_DIR}/renderers/cpu_ray_tracer/RayTracerCommon.hpp
	 ${PH_SOURCE_DIR}/renderers/cpu_ray_tracer/RayTracerCommon.cpp

	${PH_INCLUDE_DIR}/Resources.hpp
	${PH_INCLUDE_DIR}/resources/GLModel.hpp
	 ${PH_SOURCE_DIR}/resources/GLModel.cpp
	${PH_INCLUDE_DIR}/resources/GLTexture.hpp
	 ${PH_SOURCE_DIR}/resources/GLTexture.cpp
	${PH_INCLUDE_DIR}/resources/RawGeometry.hpp
	${PH_INCLUDE_DIR}/resources/RawImage.hpp
	 ${PH_SOURCE_DIR}/resources/RawImage.cpp
	${PH_INCLUDE_DIR}/resources/Renderable.hpp
	 ${PH_SOURCE_DIR}/resources/Renderable.cpp

	${PH_INCLUDE_DIR}/Screens.hpp
	${PH_INCLUDE_DIR}/screens/GameScreen.hpp
	 ${PH_SOURCE_DIR}/screens/GameScreen.cpp

	${PH_INCLUDE_DIR}/level/Level.hpp
	${PH_INCLUDE_DIR}/level/PointLight.hpp
	${PH_INCLUDE_DIR}/level/StaticScene.hpp

	${PH_INCLUDE_DIR}/util/IOUtil.hpp
	 ${PH_SOURCE_DIR}/util/IOUtil.cpp)
set_directory_source_group(${PHANTASY_ENGINE_LIB_FILES})

add_library(PhantasyEngineLib ${PHANTASY_ENGINE_LIB_FILES})
target_link_libraries(
	PhantasyEngineLib

	${SFZ_CORE_LIBRARIES}
	${ASSIMP_LIBRARIES}
	${STB_IMAGE_LIBRARIES}
	${IMGUI_LIBRARIES}
)

# CUDA Ray Tracer Renderer Library
# ------------------------------------------------------------------------------------------------

if(CUDA_TRACER)

	# CUDA dependency
	find_package(CUDA REQUIRED)
	add_definitions(-DCUDA_TRACER_AVAILABLE)

	# CUDA compiler flags
	set(CUDA_NVCC_FLAGS
		${CUDA_NVCC_FLAGS}
		-O3)

	set(CT_SOURCE_DIR ${SOURCE_DIR}/cuda_ray_tracer)
	set(CT_INCLUDE_DIR ${INCLUDE_DIR}/cuda_ray_tracer)
	include_directories(${CT_INCLUDE_DIR})

	# Source files
	set(CUDA_RAY_TRACER_FILES
		${CT_INCLUDE_DIR}/CudaRayTracerRenderer.hpp
		 ${CT_SOURCE_DIR}/CudaRayTracerRenderer.cpp
		${CT_INCLUDE_DIR}/CudaTracerEntry.cuh
		 ${CT_SOURCE_DIR}/CudaTracerEntry.cu
		${CT_INCLUDE_DIR}/CudaVectorHelpers.cuh)
	set_directory_source_group(${CUDA_RAY_TRACER_FILES})

	# CUDA Ray Tracer Library
	cuda_add_library(CudaRayTracerLib ${CUDA_RAY_TRACER_FILES})
	target_link_libraries(
		CudaRayTracerLib

		PhantasyEngineLib
		${SFZ_CORE_LIBRARIES}
	)

endif()

# Testbed application
# ------------------------------------------------------------------------------------------------

set(TB_SOURCE_DIR ${SOURCE_DIR}/testbed_application)
set(TB_INCLUDE_DIR ${INCLUDE_DIR}/testbed_application)
include_directories(${TB_INCLUDE_DIR})

set(PHANTASY_ENGINE_TESTBED_FILES
	 ${TB_SOURCE_DIR}/Main.cpp

	${TB_INCLUDE_DIR}/Helpers.hpp
	 ${TB_SOURCE_DIR}/Helpers.cpp
	${TB_INCLUDE_DIR}/TestbedLogic.hpp
	 ${TB_SOURCE_DIR}/TestbedLogic.cpp)
set_directory_source_group(${PHANTASY_ENGINE_TESTBED_FILES})

add_executable(PhantasyEngineTestbed ${PHANTASY_ENGINE_TESTBED_FILES})
target_link_libraries(
	PhantasyEngineTestbed

	PhantasyEngineLib
	${SFZ_CORE_LIBRARIES}
)

if(CUDA_TRACER)
	target_link_libraries(
		PhantasyEngineTestbed

		CudaRayTracerLib
	)
endif()

# Phantasy Engine Tests
# ------------------------------------------------------------------------------------------------

if(PHANTASY_ENGINE_BUILD_TESTS)

	# Catch
	add_subdirectory(${EXTERNALS_DIR}/catch)
	include_directories(${CATCH_INCLUDE_DIRS})

	# Test files
	set(PHANTASY_ENGINE_TEST_FILES
		${TESTS_DIR}/Main_Tests.cpp

		${TESTS_DIR}/renderers/ViewFrustum_Tests.cpp
		${TESTS_DIR}/renderers/cpu_ray_tracer/BVH_Tests.cpp
	)
	set_directory_source_group(${PHANTASY_ENGINE_TEST_FILES})

	enable_testing(true)
	add_executable(PhantasyEngineTests ${PHANTASY_ENGINE_TEST_FILES})
	target_link_libraries(
		PhantasyEngineTests

		PhantasyEngineLib
		${SFZ_CORE_LIBRARIES}
	)
	add_test(PhantasyEngineTestsName PhantasyEngineTests)

endif()

# File copying
# ------------------------------------------------------------------------------------------------

# Windows specific file copying
if(MSVC)
	# Copy assets symlinks batch file
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
