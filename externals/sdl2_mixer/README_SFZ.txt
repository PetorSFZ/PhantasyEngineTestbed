# SDL2_mixer CMake wrapper

Wrapper used to link and include SDL2_mixer in a project. Bundles official pre-built SDL2_mixer binaries (https://www.libsdl.org/projects/SDL_mixer/) for Windows in order to make it easier to build. Uses `FindSDL2_mixer.cmake` module (originally found in flare-engine-next: `https://github.com/dorkster/flare-engine-next/blob/master/cmake/FindSDL2_mixer.cmake`) on other platforms.


# Usage

Place this entire directory in the subdirectory of your CMake project, then add it with `add_subdirectory()`. Three CMake variables are returned:

`${SDL2_MIXER_INCLUDE_DIRS}`: The headers you need to include with `include_directories()` or similar.

`${SDL2_MIXER_LIBRARIES}`: The libraries you need to link your to with `target_link_libraries()`.

`${SDL2_MIXER_DLLS}`: The path to the bundled `dll`. Useful so you can set CMake up to automatically copy it to your build directory on Windows.


# License

Made by `Peter Hillerström` (`https://github.com/PetorSFZ`). I license this whole build wrapper as `public domain` (`SDL2_mixer` is obviously still `zlib`). The sole exception being the `FindSDL2_mixer.cmake` module. Feel free to do whatever you want with this wrapper, but it would be nice if you kept this readme and the header in the `CMakeLists.txt` file.
