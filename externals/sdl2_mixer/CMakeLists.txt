# Author: Peter Hillerström
# License: Public Domain (do whatever you want with this file,
#                         but it would be nice if you kept this header)
# SDL2_mixer License: zlib

# Check if 32 or 64bit compiler
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
	set(BITS_64 TRUE)
	set(BITS_32 FALSE)
else()
	set(BITS_64 FALSE)
	set(BITS_32 TRUE)
endif()

if(WIN32)
	# Use bundled SDL2 on Windows

	set(SDL2_MIXER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/SDL2_mixer-devel-2.0.1-VC)
	set(SDL2_MIXER_INCLUDE_DIR ${SDL2_MIXER_DIR}/include)

	if (BITS_64)
		find_library(SDL2_MIXER_LIB
			NAMES
			SDL2_mixer
			HINTS
			${SDL2_MIXER_DIR}/lib/x64)
		set(SDL2_MIXER_LIBRARY ${SDL2_MIXER_LIB})
		set(DLLPath ${SDL2_MIXER_DIR}/lib/x64/libFLAC-8.dll
		            ${SDL2_MIXER_DIR}/lib/x64/libmikmod-2.dll
		            ${SDL2_MIXER_DIR}/lib/x64/libmodplug-1.dll
		            ${SDL2_MIXER_DIR}/lib/x64/libogg-0.dll
		            ${SDL2_MIXER_DIR}/lib/x64/libvorbis-0.dll
		            ${SDL2_MIXER_DIR}/lib/x64/libvorbisfile-3.dll
		            ${SDL2_MIXER_DIR}/lib/x64/LICENSE.FLAC.txt
		            ${SDL2_MIXER_DIR}/lib/x64/LICENSE.mikmod.txt
		            ${SDL2_MIXER_DIR}/lib/x64/LICENSE.modplug.txt
		            ${SDL2_MIXER_DIR}/lib/x64/LICENSE.ogg-vorbis.txt
		            ${SDL2_MIXER_DIR}/lib/x64/LICENSE.smpeg.txt
		            ${SDL2_MIXER_DIR}/lib/x64/SDL2_mixer.dll
		            ${SDL2_MIXER_DIR}/lib/x64/smpeg2.dll
		            ${SDL2_MIXER_DIR}/README-SDL_mixer.txt)
	elseif(BITS32)
		find_library(SDL2_MIXER_LIB
			NAMES
			SDL2_mixer
			HINTS
			${SDL2_MIXER_DIR}/lib/x86)
		set(SDL2_MIXER_LIBRARY ${SDL2_MIXER_LIB})
		set(DLLPath ${SDL2_MIXER_DIR}/lib/x86/libFLAC-8.dll
		            ${SDL2_MIXER_DIR}/lib/x86/libmodplug-1.dll
		            ${SDL2_MIXER_DIR}/lib/x86/libogg-0.dll
		            ${SDL2_MIXER_DIR}/lib/x86/libvorbis-0.dll
		            ${SDL2_MIXER_DIR}/lib/x86/libvorbisfile-3.dll
		            ${SDL2_MIXER_DIR}/lib/x86/LICENSE.FLAC.txt
		            ${SDL2_MIXER_DIR}/lib/x86/LICENSE.modplug.txt
		            ${SDL2_MIXER_DIR}/lib/x86/LICENSE.ogg-vorbis.txt
		            ${SDL2_MIXER_DIR}/lib/x86/LICENSE.smpeg.txt
		            ${SDL2_MIXER_DIR}/lib/x86/SDL2_mixer.dll
		            ${SDL2_MIXER_DIR}/lib/x86/smpeg2.dll
		            ${SDL2_MIXER_DIR}/README-SDL_mixer.txt)
	endif()

else()
	# Attempt to use bundled FindSDL2_mixer.cmake module to find SDL2_mixer installed on users system
	list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/find_sdl2_mixer_module)
	find_package(SDL2_mixer REQUIRED) # ${SDL2_MIXER_INCLUDE_DIRS}, ${SDL2_MIXER_LIBRARIES}
	set(SDL2_MIXER_INCLUDE_DIR ${SDL2_MIXER_INCLUDE_DIRS})
	set(SDL2_MIXER_LIBRARY ${SDL2_MIXER_LIBRARIES})
endif()

# Return variables
set(SDL2_MIXER_INCLUDE_DIRS ${SDL2_MIXER_INCLUDE_DIR} PARENT_SCOPE)
set(SDL2_MIXER_LIBRARIES ${SDL2_MIXER_LIBRARY} PARENT_SCOPE)
set(SDL2_MIXER_DLLS ${DLLPath} PARENT_SCOPE)
