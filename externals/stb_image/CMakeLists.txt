# Disable warnings when building stb_image
if(MSVC)
	set(CMAKE_CXX_FLAGS "/W1 /EHsc")
else()
	set(CMAKE_CXX_FLAGS "-w")
endif()

# Compile stb_image
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
add_library(stb_image
	${CMAKE_CURRENT_SOURCE_DIR}/include/stb_image.h
	${CMAKE_CURRENT_SOURCE_DIR}/src/STBImage.cpp)

# Variables for linking
set(STB_IMAGE_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include PARENT_SCOPE)
set(STB_IMAGE_LIBRARIES stb_image PARENT_SCOPE)
