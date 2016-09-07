add_library(imgui
	STATIC
	src/imgui.cpp
	src/imgui_demo.cpp
	src/imgui_draw.cpp
	src/imgui_impl_sdl_gl3.cpp
)

include_directories(
	${CMAKE_CURRENT_SOURCE_DIR}/include
	${SFZ_CORE_INCLUDE_DIRS}
)

# Variables for linking
set(IMGUI_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include PARENT_SCOPE)
set(IMGUI_LIBRARIES imgui PARENT_SCOPE)
