# Phantasy Engine Testbed

This is an early work in progress game engine. Of note is an experimental real-time ray tracing backend using CUDA.

## Building

### Windows

You need to have the following things installed:

* CMake
* Visual Studio 2015 (or newer)
* (Optional) CUDA 8 (or newer) for building CUDA backends

The easiest way to generate the visual studio solution is to run the `GenBuildFiles.bat` or `GenBuildFilesWithCUDA.bat` batch files. These files automatically perform the operations described in the manual method below. The first file will generate a solution without CUDA support in a directory called `build_gl`, the latter file will generate with CUDA support in a directory called `build_cuda`.

The manual way:

* Create a directory called build
* Run the following command if using CUDA: `cmake .. -G "Visual Studio 14 2015 Win64" -DCUDA_TRACER=TRUE -DPHANTASY_ENGINE_BUILD_TESTS=TRUE`
* Run the following command if not using CUDA: `cmake .. -G "Visual Studio 14 2015 Win64" -DPHANTASY_ENGINE_BUILD_TESTS=TRUE`

The `CUDA_TRACER` variable controls whether the CUDA ray tracer backend is built or not. If you don't have a CUDA compatible card this should not be enabled. The `PHANTASY_ENGINE_BUILD_TESTS` variable controls whether the tests are built or not.