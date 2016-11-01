# Phantasy Engine Testbed

This is an early work in progress game engine. Of note is an experimental real-time ray tracing backend using CUDA.

## Building

Create a directory called `build`, then run the following command:

	cmake .. -G "Visual Studio 14 2015 Win64" -DCUDA_TRACER=TRUE -DPHANTASY_ENGINE_BUILD_TESTS=TRUE

The `CUDA_TRACER` variable controls whether the CUDA ray tracer backend is built or not. If you don't have a CUDA compatible card this should not be enabled. The `PHANTASY_ENGINE_BUILD_TESTS` variable controls whether the tests are built or not.