// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#include "CudaRayTracerRenderer.hpp"

#include <chrono>

#include <sfz/gl/Program.hpp>
#include <sfz/math/MathHelpers.hpp>
#include <sfz/memory/New.hpp>
#include <sfz/util/IO.hpp>

#include <sfz/gl/IncludeOpenGL.hpp>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include <phantasy_engine/RayTracerCommon.hpp>
#include <phantasy_engine/renderers/FullscreenTriangle.hpp>

#include "CudaHelpers.hpp"
#include "CudaTracer.cuh"

namespace phe {

using namespace sfz;

// CUDARayTracerRendererImpl
// ------------------------------------------------------------------------------------------------

class CUDARayTracerRendererImpl final {
public:
	gl::Program transferShader;
	FullscreenTriangle fullscreenTriangle;

	// Temp
	GLuint glTex = 0;
	cudaGraphicsResource_t cudaResource = 0;
	cudaArray_t cudaArray = 0; // Probably no need to free, since memory is owned by OpenGL
	cudaSurfaceObject_t cudaSurface = 0;

	BVH bvh;
	StaticSceneCuda staticSceneCuda;

	CUDARayTracerRendererImpl() noexcept
	{
		
	}

	~CUDARayTracerRendererImpl() noexcept
	{
		CHECK_CUDA_ERROR(cudaDestroySurfaceObject(cudaSurface));
		CHECK_CUDA_ERROR(cudaGraphicsUnregisterResource(cudaResource));
		glDeleteTextures(1, &glTex);

		CHECK_CUDA_ERROR(cudaFree(staticSceneCuda.pointLights));
		CHECK_CUDA_ERROR(cudaFree(staticSceneCuda.bvhNodes));
		CHECK_CUDA_ERROR(cudaFree(staticSceneCuda.triangleVertices));
		CHECK_CUDA_ERROR(cudaFree(staticSceneCuda.triangleDatas));
	}
};


// CUDARayTracerRenderer: Constructors & destructors
// ------------------------------------------------------------------------------------------------

CUDARayTracerRenderer::CUDARayTracerRenderer() noexcept
{
	mImpl = sfz_new<CUDARayTracerRendererImpl>();

	StackString128 shadersPath;
	shadersPath.printf("%sresources/shaders/", basePath());
	mImpl->transferShader = gl::Program::postProcessFromFile(shadersPath.str, "transfer.frag");
	glUseProgram(mImpl->transferShader.handle());
	gl::setUniform(mImpl->transferShader, "uSrcTexture", 0);
}

CUDARayTracerRenderer::~CUDARayTracerRenderer() noexcept
{
	sfz_delete(mImpl);
}

// CUDARayTracerRenderer: Virtual methods from BaseRenderer interface
// ------------------------------------------------------------------------------------------------

RenderResult CUDARayTracerRenderer::render(Framebuffer& resultFB) noexcept
{
	// Calculate camera def in order to generate first rays
	vec2 resultRes = vec2(mTargetResolution);
	CameraDef cam = generateCameraDef(mMatrices.position, mMatrices.forward, mMatrices.up,
	                                  mMatrices.vertFovRad, resultRes);

	// Run CUDA ray tracer
	runCudaRayTracer(mImpl->cudaSurface, mTargetResolution, cam, mImpl->staticSceneCuda);
	
	// Transfer result from Cuda texture to result framebuffer
	glUseProgram(mImpl->transferShader.handle());
	resultFB.bindViewportClearColorDepth(vec4(0.0f, 0.0f, 0.0f, 0.0f), 0.0f);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, mImpl->glTex);

	mImpl->fullscreenTriangle.render();

	// Return result
	RenderResult tmp;
	tmp.renderedRes = mTargetResolution;
	return tmp;
}

// CUDARayTracerRenderer: Protected virtual methods from BaseRenderer interface
// ------------------------------------------------------------------------------------------------

void CUDARayTracerRenderer::staticSceneChanged() noexcept
{
	// Build the BVH
	BVH& bvh = mImpl->bvh;
	{
		using time_point = std::chrono::high_resolution_clock::time_point;
		time_point before = std::chrono::high_resolution_clock::now();

		bvh.buildStaticFrom(*mStaticScene.get());
		optimizeBVHCacheLocality(bvh);

		time_point after = std::chrono::high_resolution_clock::now();
		using FloatSecond = std::chrono::duration<float>;
		float delta = std::chrono::duration_cast<FloatSecond>(after - before).count();
		printf("CUDA Ray Tracer: Time spent building BVH: %.3f seconds\n", delta);
	}

	// Copy pointlights to GPU
	PointLight*& gpuPointLights = mImpl->staticSceneCuda.pointLights;
	CHECK_CUDA_ERROR(cudaFree(gpuPointLights));
	size_t numPointLightBytes = mStaticScene->pointLights.size() * sizeof(PointLight);
	CHECK_CUDA_ERROR(cudaMalloc(&gpuPointLights, numPointLightBytes));
	CHECK_CUDA_ERROR(cudaMemcpy(gpuPointLights, mStaticScene->pointLights.data(), numPointLightBytes, cudaMemcpyHostToDevice));

	// Copy BVHNodes to GPU
	BVHNode*& gpuBVHNodes = mImpl->staticSceneCuda.bvhNodes;
	CHECK_CUDA_ERROR(cudaFree(gpuBVHNodes));
	size_t numBVHNodesBytes = bvh.nodes.size() * sizeof(BVHNode);
	CHECK_CUDA_ERROR(cudaMalloc(&gpuBVHNodes, numBVHNodesBytes));
	CHECK_CUDA_ERROR(cudaMemcpy(gpuBVHNodes, bvh.nodes.data(), numBVHNodesBytes, cudaMemcpyHostToDevice));

	// Copy triangle positions to GPU
	TriangleVertices*& gpuTriangleVertices = mImpl->staticSceneCuda.triangleVertices;
	CHECK_CUDA_ERROR(cudaFree(gpuTriangleVertices));
	size_t numTrianglePosBytes = bvh.triangles.size() * sizeof(TriangleVertices);
	CHECK_CUDA_ERROR(cudaMalloc(&gpuTriangleVertices, numTrianglePosBytes));
	CHECK_CUDA_ERROR(cudaMemcpy(gpuTriangleVertices, mImpl->bvh.triangles.data(), numTrianglePosBytes, cudaMemcpyHostToDevice));

	// Copy Triangle datas to GPU
	TriangleData*& gpuTriangleDatas = mImpl->staticSceneCuda.triangleDatas;
	CHECK_CUDA_ERROR(cudaFree(gpuTriangleDatas));
	size_t numTriangleDatasBytes = bvh.triangleDatas.size() * sizeof(TriangleData);
	CHECK_CUDA_ERROR(cudaMalloc(&gpuTriangleDatas, numTriangleDatasBytes));
	CHECK_CUDA_ERROR(cudaMemcpy(gpuTriangleDatas, mImpl->bvh.triangleDatas.data(), numTriangleDatasBytes, cudaMemcpyHostToDevice));
}

void CUDARayTracerRenderer::targetResolutionUpdated() noexcept
{
	glActiveTexture(GL_TEXTURE0);

	// Cleanup eventual previous texture and bindings
	if (mImpl->cudaSurface != 0) {
		CHECK_CUDA_ERROR(cudaDestroySurfaceObject(mImpl->cudaSurface));
		mImpl->cudaSurface = 0;
	}
	if (mImpl->cudaResource != 0) {
		CHECK_CUDA_ERROR(cudaGraphicsUnregisterResource(mImpl->cudaResource));
		mImpl->cudaResource = 0;
	}
	glDeleteTextures(1, &mImpl->glTex);

	// Create OpenGL texture and allocate memory
	glGenTextures(1, &mImpl->glTex);
	glBindTexture(GL_TEXTURE_2D, mImpl->glTex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, mTargetResolution.x, mTargetResolution.y, 0, GL_RGBA, GL_FLOAT, nullptr);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);

	// https://github.com/nvpro-samples/gl_cuda_interop_pingpong_st
	cudaGraphicsResource_t& resource = mImpl->cudaResource;
	cudaArray_t& array = mImpl->cudaArray;
	CHECK_CUDA_ERROR(cudaGraphicsGLRegisterImage(&resource, mImpl->glTex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore));
	CHECK_CUDA_ERROR(cudaGraphicsMapResources(1, &resource, 0));
	CHECK_CUDA_ERROR(cudaGraphicsSubResourceGetMappedArray(&array, resource, 0, 0));
	CHECK_CUDA_ERROR(cudaGraphicsUnmapResources(1, &resource, 0));

	// Create cuda surface object from binding
	cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(cudaResourceDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = mImpl->cudaArray;
	CHECK_CUDA_ERROR(cudaCreateSurfaceObject(&mImpl->cudaSurface, &resDesc));
}

} // namespace phe
