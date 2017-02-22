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

# Require 64bit compiler
if(NOT (CMAKE_SIZEOF_VOID_P EQUAL 8))
	message(FATAL_ERROR "Can only compile in 64bit mode")
endif()

# Compiler flags
if(MSVC)
	# MSVC flags
	# /W4 = Warning level 4 (/Wall is too picky and has annoying warnings in standard headers)
	# /wd4201 = Disable warning 4201 (nonstandard extension used : nameless struct/union)
	# /Zi = Produce .pdb debug information. Does not affect optimizations, but does imply /debug.
	# /arch:AVX = Enable (require) Intel AVX instructions for code generation
	# /D_CRT_SECURE_NO_WARNINGS = Removes annyoing warning when using c standard library
	# /MD = "Defines _MT and _DLL so that both multithread- and DLL-specific versions of the
	#        run-time routines are selected from the standard .h files."
	set(CMAKE_CXX_FLAGS "/W4 /wd4201 /Zi /EHsc /arch:AVX /D_CRT_SECURE_NO_WARNINGS /MD")
	# /O2 = Optimize code for fastest speed
	# /fp:fast = "optimize floating-point code for speed at the expense of accuracy and correctness"
	# /DSFZ_NO_DEBUG = defines the "SFZ_NO_DEBUG" macro, which disables sfz_assert_debug()
	set(CMAKE_CXX_FLAGS_RELEASE "/O2 /fp:fast /DSFZ_NO_DEBUG")
	# /DEBUG = "creates debugging information for the .exe file or DLL"
	set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "/O2 /fp:fast /DEBUG /DSFZ_NO_DEBUG")
	# /Od = "disables optimization, speeding compilation and simplifying debugging"
	set(CMAKE_CXX_FLAGS_DEBUG "/Od /DEBUG")
else()
	# GCC / Clang flags
	set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wno-c++11-extensions -std=c++11 -pthread -fno-strict-aliasing")
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
if(CUDA_TRACER)
set(SFZ_CORE_CUDA TRUE)
endif()
if(NOT WIN32)
add_definitions(-DGLEW_STATIC)
endif()
add_subdirectory(${EXTERNALS_DIR}/sfzCore)

# Assimp
add_subdirectory(${EXTERNALS_DIR}/assimp)

# stb_image
add_subdirectory(${EXTERNALS_DIR}/stb_image)

# imgui
add_subdirectory(${EXTERNALS_DIR}/imgui)

# SDL2_mixer, sets ${SDL2_MIXER_INCLUDE_DIRS}, ${SDL2_MIXER_LIBRARIES}, ${SDL2_MIXER_DLLS}
add_subdirectory(${EXTERNALS_DIR}/sdl2_mixer)

# Include directories
include_directories(
	${INCLUDE_DIR}
	${SFZ_CORE_INCLUDE_DIRS}
	${ASSIMP_INCLUDE_DIRS}
	${STB_IMAGE_INCLUDE_DIRS}
	${IMGUI_INCLUDE_DIRS}
	${SDL2_MIXER_INCLUDE_DIRS}
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

	${PH_INCLUDE_DIR}/DeferredRenderer.hpp
	${PH_INCLUDE_DIR}/deferred_renderer/DeferredRenderer.hpp
	 ${PH_SOURCE_DIR}/deferred_renderer/DeferredRenderer.cpp
	${PH_INCLUDE_DIR}/deferred_renderer/GLModel.hpp
	 ${PH_SOURCE_DIR}/deferred_renderer/GLModel.cpp
	${PH_INCLUDE_DIR}/deferred_renderer/GLTexture.hpp
	 ${PH_SOURCE_DIR}/deferred_renderer/GLTexture.cpp
	${PH_INCLUDE_DIR}/deferred_renderer/ShadowCubeMap.hpp
	 ${PH_SOURCE_DIR}/deferred_renderer/ShadowCubeMap.cpp
	${PH_INCLUDE_DIR}/deferred_renderer/SSBO.hpp
	 ${PH_SOURCE_DIR}/deferred_renderer/SSBO.cpp

	${PH_INCLUDE_DIR}/ECS.hpp
	${PH_INCLUDE_DIR}/ecs/EcsComponentAccessor.hpp
	${PH_INCLUDE_DIR}/ecs/EcsWrapper.hpp
	 ${PH_SOURCE_DIR}/ecs/EcsWrapper.cpp
	${PH_INCLUDE_DIR}/ecs/EntityComponentSystem.hpp
	 ${PH_SOURCE_DIR}/ecs/EntityComponentSystem.cpp
	${PH_INCLUDE_DIR}/ecs/SceneGraph.hpp
	 ${PH_SOURCE_DIR}/ecs/SceneGraph.cpp

	${PH_INCLUDE_DIR}/level/Level.hpp
	${PH_INCLUDE_DIR}/level/ObjectLoader.hpp
	 ${PH_SOURCE_DIR}/level/ObjectLoader.cpp
	${PH_INCLUDE_DIR}/level/SphereLight.hpp
	${PH_INCLUDE_DIR}/level/SponzaLoader.hpp
	 ${PH_SOURCE_DIR}/level/SponzaLoader.cpp
	${PH_INCLUDE_DIR}/level/StaticScene.hpp

	${PH_INCLUDE_DIR}/RayTracerCommon.hpp
	${PH_INCLUDE_DIR}/ray_tracer_common/BVH.hpp
	${PH_INCLUDE_DIR}/ray_tracer_common/BVHMetrics.hpp
	 ${PH_SOURCE_DIR}/ray_tracer_common/BVHMetrics.cpp
	${PH_INCLUDE_DIR}/ray_tracer_common/BVHNode.hpp
	${PH_INCLUDE_DIR}/ray_tracer_common/BVHTraversal.hpp
	${PH_INCLUDE_DIR}/ray_tracer_common/DynamicBVH.hpp
	 ${PH_SOURCE_DIR}/ray_tracer_common/DynamicBVH.cpp
	${PH_INCLUDE_DIR}/ray_tracer_common/GenerateRays.hpp
	 ${PH_SOURCE_DIR}/ray_tracer_common/GenerateRays.cpp
	${PH_INCLUDE_DIR}/ray_tracer_common/Intersection.hpp
	${PH_INCLUDE_DIR}/ray_tracer_common/Ray.hpp
	${PH_INCLUDE_DIR}/ray_tracer_common/Shading.hpp
	${PH_INCLUDE_DIR}/ray_tracer_common/StaticBVHBuilder.hpp
	 ${PH_SOURCE_DIR}/ray_tracer_common/StaticBVHBuilder.cpp
	${PH_INCLUDE_DIR}/ray_tracer_common/Triangle.hpp

	${PH_INCLUDE_DIR}/rendering/BaseRenderer.hpp
	${PH_INCLUDE_DIR}/rendering/FullscreenTriangle.hpp
	 ${PH_SOURCE_DIR}/rendering/FullscreenTriangle.cpp
	${PH_INCLUDE_DIR}/rendering/Material.hpp
	${PH_INCLUDE_DIR}/rendering/RawImage.hpp
	 ${PH_SOURCE_DIR}/rendering/RawImage.cpp
	${PH_INCLUDE_DIR}/rendering/RawMesh.hpp
	${PH_INCLUDE_DIR}/rendering/RenderComponent.hpp
	${PH_INCLUDE_DIR}/rendering/ViewFrustum.hpp
	 ${PH_SOURCE_DIR}/rendering/ViewFrustum.cpp

	${PH_INCLUDE_DIR}/sbvh-stuff/Array.h
	${PH_INCLUDE_DIR}/sbvh-stuff/BVH.h
	 ${PH_SOURCE_DIR}/sbvh-stuff/BVH.cpp
	${PH_INCLUDE_DIR}/sbvh-stuff/BVHNode.h
	 ${PH_SOURCE_DIR}/sbvh-stuff/BVHNode.cpp
	${PH_INCLUDE_DIR}/sbvh-stuff/Scene.h
	${PH_INCLUDE_DIR}/sbvh-stuff/Sort.h
	 ${PH_SOURCE_DIR}/sbvh-stuff/Sort.cpp
	${PH_INCLUDE_DIR}/sbvh-stuff/SplitBVHBuilder.h
	 ${PH_SOURCE_DIR}/sbvh-stuff/SplitBVHBuilder.cpp
	${PH_INCLUDE_DIR}/sbvh-stuff/Util.h
	 ${PH_SOURCE_DIR}/sbvh-stuff/Util.cpp

	${PH_INCLUDE_DIR}/Screens.hpp
	${PH_INCLUDE_DIR}/screens/GameScreen.hpp
	 ${PH_SOURCE_DIR}/screens/GameScreen.cpp

	${PH_INCLUDE_DIR}/Sound.hpp
	${PH_INCLUDE_DIR}/sound/Music.hpp
	 ${PH_SOURCE_DIR}/sound/Music.cpp
	${PH_INCLUDE_DIR}/sound/SoundEffect.hpp
	 ${PH_SOURCE_DIR}/sound/SoundEffect.cpp
	${PH_INCLUDE_DIR}/sound/SoundSession.hpp
	 ${PH_SOURCE_DIR}/sound/SoundSession.cpp

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
	${SDL2_MIXER_LIBRARIES}
)

# CUDA Tracer Renderer Library
# ------------------------------------------------------------------------------------------------

if(CUDA_TRACER)

	# CUDA dependency
	find_package(CUDA REQUIRED)
	add_definitions(-DCUDA_TRACER_AVAILABLE)

	# CUDA compiler flags
	set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -expt-relaxed-constexpr --gpu-architecture=compute_30 --gpu-code=compute_30)
	set(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG} -g -lineinfo) # Add -G for CUDA device debugging
	set(CUDA_NVCC_FLAGS_RELWITHDEBINFO ${CUDA_NVCC_FLAGS_RELWITHDEBINFO} -O3 -use_fast_math -g -lineinfo)
	set(CUDA_NVCC_FLAGS_RELEASE ${CUDA_NVCC_FLAGS_RELEASE} -O3 -use_fast_math)

	set(CT_SOURCE_DIR ${SOURCE_DIR}/cuda_ray_tracer)
	set(CT_INCLUDE_DIR ${INCLUDE_DIR}/cuda_ray_tracer)
	include_directories(${CT_INCLUDE_DIR})

	# Source files
	set(CUDA_TRACER_FILES
		
		${CT_INCLUDE_DIR}/deprecated/BVHTraversal.cuh
		${CT_INCLUDE_DIR}/deprecated/CudaTracer.cuh
		 ${CT_SOURCE_DIR}/deprecated/CudaTracer.cu

		${CT_INCLUDE_DIR}/kernels/GenSecondaryRaysKernel.hpp
		 ${CT_SOURCE_DIR}/kernels/GenSecondaryRaysKernel.cu
		${CT_INCLUDE_DIR}/kernels/GenSecondaryShadowRaysKernel.hpp
		 ${CT_SOURCE_DIR}/kernels/GenSecondaryShadowRaysKernel.cu
		${CT_INCLUDE_DIR}/kernels/InitCurandKernel.hpp
		 ${CT_SOURCE_DIR}/kernels/InitCurandKernel.cu
		${CT_INCLUDE_DIR}/kernels/InterpretRayHitKernel.hpp
		 ${CT_SOURCE_DIR}/kernels/InterpretRayHitKernel.cu
		${CT_INCLUDE_DIR}/kernels/MaterialKernel.hpp
		 ${CT_SOURCE_DIR}/kernels/MaterialKernel.cu
		${CT_INCLUDE_DIR}/kernels/PetorShading.hpp
		 ${CT_SOURCE_DIR}/kernels/PetorShading.cu
		${CT_INCLUDE_DIR}/kernels/ProcessGBufferKernel.hpp
		 ${CT_SOURCE_DIR}/kernels/ProcessGBufferKernel.cu
		${CT_INCLUDE_DIR}/kernels/RayCastKernel.hpp
		 ${CT_SOURCE_DIR}/kernels/RayCastKernel.cu
		${CT_INCLUDE_DIR}/kernels/ShadeSecondaryHit.hpp
		 ${CT_SOURCE_DIR}/kernels/ShadeSecondaryHit.cu

		${CT_INCLUDE_DIR}/CudaBindlessTexture.hpp
		 ${CT_SOURCE_DIR}/CudaBindlessTexture.cpp
		${CT_INCLUDE_DIR}/CudaDeviceHelpers.cuh
		${CT_INCLUDE_DIR}/CudaGLInterop.hpp
		 ${CT_SOURCE_DIR}/CudaGLInterop.cpp
		${CT_INCLUDE_DIR}/CudaPbr.cuh
		${CT_INCLUDE_DIR}/CudaSfzVectorCompatibility.cuh
		${CT_INCLUDE_DIR}/CudaTextureBuffer.hpp
		${CT_INCLUDE_DIR}/CudaTracerRenderer.hpp
		 ${CT_SOURCE_DIR}/CudaTracerRenderer.cpp
		${CT_INCLUDE_DIR}/GBufferRead.cuh
		${CT_INCLUDE_DIR}/Morton.cuh
		${CT_INCLUDE_DIR}/ShadowCubeMapCudaGL.hpp
		 ${CT_SOURCE_DIR}/ShadowCubeMapCudaGL.cpp
	)
	set_directory_source_group(${CUDA_TRACER_FILES})

	# CUDA Tracer Library
	cuda_add_library(CudaTracerLib ${CUDA_TRACER_FILES})
	target_link_libraries(
		CudaTracerLib

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

		CudaTracerLib
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

		${TESTS_DIR}/level/EntityComponentSystem_Tests.cpp

		${TESTS_DIR}/renderers/ViewFrustum_Tests.cpp
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
	# Runtime DLLs
	file(COPY ${SFZ_CORE_DLLS} DESTINATION ${CMAKE_BINARY_DIR}/Debug)
	file(COPY ${SFZ_CORE_DLLS} DESTINATION ${CMAKE_BINARY_DIR}/RelWithDebInfo)
	file(COPY ${SFZ_CORE_DLLS} DESTINATION ${CMAKE_BINARY_DIR}/Release)

	file(COPY ${ASSIMP_DLLS} DESTINATION ${CMAKE_BINARY_DIR}/Debug)
	file(COPY ${ASSIMP_DLLS} DESTINATION ${CMAKE_BINARY_DIR}/RelWithDebInfo)
	file(COPY ${ASSIMP_DLLS} DESTINATION ${CMAKE_BINARY_DIR}/Release)

	file(COPY ${SDL2_MIXER_DLLS} DESTINATION ${CMAKE_BINARY_DIR}/Debug)
	file(COPY ${SDL2_MIXER_DLLS} DESTINATION ${CMAKE_BINARY_DIR}/RelWithDebInfo)
	file(COPY ${SDL2_MIXER_DLLS} DESTINATION ${CMAKE_BINARY_DIR}/Release)
endif()

if(UNIX)
	add_custom_command(TARGET PhantasyEngineTestbed PRE_BUILD
			COMMAND ${CMAKE_COMMAND} -E create_symlink
			\"${PROJECT_SOURCE_DIR}/resources\"
			\"$<TARGET_FILE_DIR:PhantasyEngineTestbed>/resources\")
endif()
