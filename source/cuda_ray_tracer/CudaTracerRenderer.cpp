// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#include "CudaTracerRenderer.hpp"

#include <chrono>

#include <sfz/gl/Program.hpp>
#include <sfz/math/MathHelpers.hpp>
#include <sfz/memory/New.hpp>
#include <sfz/util/IO.hpp>

#include <sfz/gl/IncludeOpenGL.hpp>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include <phantasy_engine/config/GlobalConfig.hpp>
#include <phantasy_engine/RayTracerCommon.hpp>
#include <phantasy_engine/rendering/FullscreenTriangle.hpp>

#include "CudaBindlessTexture.hpp"
#include "CudaHelpers.hpp"
#include "CudaTracer.cuh"

namespace phe {

using namespace sfz;

// CudaTracerRendererImpl
// ------------------------------------------------------------------------------------------------

class CudaTracerRendererImpl final {
public:
	gl::Program transferShader;
	FullscreenTriangle fullscreenTriangle;

	// Holding the OpenGL Cuda surface data, surface object is in CudaTracerParams.
	GLuint glTex = 0;
	cudaGraphicsResource_t cudaResource = 0;
	cudaArray_t cudaArray = 0; // Probably no need to free, since memory is owned by OpenGL

	// Parameters for tracer
	BVH staticBvh; // TODO: Move out to static scene
	DynArray<CudaBindlessTexture> textureWrappers;
	DynArray<cudaTextureObject_t> textureObjectHandles;
	CudaTracerParams tracerParams;

	Setting* mCudaDebugRender = nullptr;

	CameraDef lastCamera;

	CudaTracerRendererImpl() noexcept
	{
		
	}

	~CudaTracerRendererImpl() noexcept
	{
		// Target surface and OpenGL data
		CHECK_CUDA_ERROR(cudaDestroySurfaceObject(tracerParams.targetSurface));
		CHECK_CUDA_ERROR(cudaGraphicsUnregisterResource(cudaResource));
		glDeleteTextures(1, &glTex);

		// Cuda RNG states
		CHECK_CUDA_ERROR(cudaFree(tracerParams.curandStates));

		// Materials & textures
		CHECK_CUDA_ERROR(cudaFree(tracerParams.materials));
		CHECK_CUDA_ERROR(cudaFree(tracerParams.textures));

		// Static Geometry
		CHECK_CUDA_ERROR(cudaFree(tracerParams.staticBvhNodes));
		CHECK_CUDA_ERROR(cudaFree(tracerParams.staticTriangleVertices));
		CHECK_CUDA_ERROR(cudaFree(tracerParams.staticTriangleDatas));

		// Static light sources
		CHECK_CUDA_ERROR(cudaFree(tracerParams.staticPointLights));
	}
};

// CudaTracerRenderer: Constructors & destructors
// ------------------------------------------------------------------------------------------------

CudaTracerRenderer::CudaTracerRenderer() noexcept
{
	mImpl = sfz_new<CudaTracerRendererImpl>();

	StackString128 shadersPath;
	shadersPath.printf("%sresources/shaders/", basePath());
	mImpl->transferShader = gl::Program::postProcessFromFile(shadersPath.str, "cuda_transfer.frag");
	glUseProgram(mImpl->transferShader.handle());
	gl::setUniform(mImpl->transferShader, "uSrcTexture", 0);

	GlobalConfig& cfg = GlobalConfig::instance();
	mImpl->mCudaDebugRender = cfg.sanitizeBool("CudaTracer", "cudaDebugRender", false);
}

CudaTracerRenderer::~CudaTracerRenderer() noexcept
{
	sfz_delete(mImpl);
}

// CudaTracerRenderer: Virtual methods from BaseRenderer interface
// ------------------------------------------------------------------------------------------------

void CudaTracerRenderer::bakeMaterials(const DynArray<RawImage>& textures,
                                       const DynArray<Material>& materials) noexcept
{
	// Copy materials to CUDA
	Material*& gpuMaterials = mImpl->tracerParams.materials;
	CHECK_CUDA_ERROR(cudaFree(gpuMaterials));
	size_t numGpuMaterialBytes = materials.size() * sizeof(Material);
	CHECK_CUDA_ERROR(cudaMalloc(&gpuMaterials, numGpuMaterialBytes));
	CHECK_CUDA_ERROR(cudaMemcpy(gpuMaterials, materials.data(), numGpuMaterialBytes, cudaMemcpyHostToDevice));
	mImpl->tracerParams.numMaterials = materials.size();

	// Create CUDA bindless textures from textures
	mImpl->textureWrappers.clear();
	mImpl->textureObjectHandles.clear();
	for (const RawImage& texture : textures) {
		CudaBindlessTexture tmp;
		tmp.load(texture);
		mImpl->textureWrappers.add(std::move(tmp));
		mImpl->textureObjectHandles.add(mImpl->textureWrappers.last().textureObject());
	}

	// Copy texture objects into CUDA
	cudaSurfaceObject_t*& gpuTextures = mImpl->tracerParams.textures;
	CHECK_CUDA_ERROR(cudaFree(gpuTextures));
	size_t numGpuTexturesBytes = mImpl->textureObjectHandles.size() * sizeof(cudaSurfaceObject_t);
	CHECK_CUDA_ERROR(cudaMalloc(&gpuTextures, numGpuTexturesBytes));
	CHECK_CUDA_ERROR(cudaMemcpy(gpuTextures, mImpl->textureObjectHandles.data(), numGpuTexturesBytes, cudaMemcpyHostToDevice));
	mImpl->tracerParams.numTextures = textures.size();
}

void CudaTracerRenderer::addMaterial(RawImage& texture, Material& material) noexcept
{
	sfz::error("CudaTracerRenderer: addMaterial() not implemented");
}

void CudaTracerRenderer::bakeStaticScene(const StaticScene& staticScene) noexcept
{
	// Build the BVH
	// TODO: Remove and place in static scene
	BVH& staticBvh = mImpl->staticBvh;
	{
		using time_point = std::chrono::high_resolution_clock::time_point;
		time_point before = std::chrono::high_resolution_clock::now();

		staticBvh = std::move(buildStaticFrom(staticScene));
		optimizeBVHCacheLocality(staticBvh);

		time_point after = std::chrono::high_resolution_clock::now();
		using FloatSecond = std::chrono::duration<float>;
		float delta = std::chrono::duration_cast<FloatSecond>(after - before).count();
		printf("CudaTracerRenderer: Time spent building BVH: %.3f seconds\n", delta);
	}

	// Copy static BVH to GPU
	BVHNode*& gpuBVHNodes = mImpl->tracerParams.staticBvhNodes;
	CHECK_CUDA_ERROR(cudaFree(gpuBVHNodes));
	size_t numBVHNodesBytes = staticBvh.nodes.size() * sizeof(BVHNode);
	CHECK_CUDA_ERROR(cudaMalloc(&gpuBVHNodes, numBVHNodesBytes));
	CHECK_CUDA_ERROR(cudaMemcpy(gpuBVHNodes, staticBvh.nodes.data(), numBVHNodesBytes, cudaMemcpyHostToDevice));

	// Copy static triangle vertices to GPU
	TriangleVertices*& gpuTriangleVertices = mImpl->tracerParams.staticTriangleVertices;
	CHECK_CUDA_ERROR(cudaFree(gpuTriangleVertices));
	size_t numTriangleVertBytes = staticBvh.triangles.size() * sizeof(TriangleVertices);
	CHECK_CUDA_ERROR(cudaMalloc(&gpuTriangleVertices, numTriangleVertBytes));
	CHECK_CUDA_ERROR(cudaMemcpy(gpuTriangleVertices, staticBvh.triangles.data(), numTriangleVertBytes, cudaMemcpyHostToDevice));

	// Copy static triangle datas to GPU
	TriangleData*& gpuTriangleDatas = mImpl->tracerParams.staticTriangleDatas;
	CHECK_CUDA_ERROR(cudaFree(gpuTriangleDatas));
	size_t numTriangleDatasBytes = staticBvh.triangleDatas.size() * sizeof(TriangleData);
	CHECK_CUDA_ERROR(cudaMalloc(&gpuTriangleDatas, numTriangleDatasBytes));
	CHECK_CUDA_ERROR(cudaMemcpy(gpuTriangleDatas, staticBvh.triangleDatas.data(), numTriangleDatasBytes, cudaMemcpyHostToDevice));

	// Copy static point lights to GPU
	PointLight*& gpuPointLights = mImpl->tracerParams.staticPointLights;
	CHECK_CUDA_ERROR(cudaFree(gpuPointLights));
	size_t numPointLightBytes = staticScene.pointLights.size() * sizeof(PointLight);
	CHECK_CUDA_ERROR(cudaMalloc(&gpuPointLights, numPointLightBytes));
	CHECK_CUDA_ERROR(cudaMemcpy(gpuPointLights, staticScene.pointLights.data(), numPointLightBytes, cudaMemcpyHostToDevice));
	mImpl->tracerParams.numStaticPointLights = staticScene.pointLights.size();
}

RenderResult CudaTracerRenderer::render(Framebuffer& resultFB) noexcept
{
	// Calculate camera def in order to generate first rays
	vec2 resultRes = vec2(mTargetResolution);
	mImpl->tracerParams.cam = generateCameraDef(mMatrices.position, mMatrices.forward, mMatrices.up,
	                                            mMatrices.vertFovRad, resultRes);

	CudaTracerParams& params = mImpl->tracerParams;

	// Check if camera has moved. If so, forget accumulated color.
	if (mImpl->lastCamera.origin != params.cam.origin ||
	    mImpl->lastCamera.dir != params.cam.dir) {
		clearSurface(params.targetSurface, params.targetRes, vec4(0.0f));
		params.frameCount = 0;

		mImpl->lastCamera = params.cam;
	}
	params.frameCount++;

	// Run CUDA ray tracer
	bool cudaDebugRender = mImpl->mCudaDebugRender->boolValue();
	if (!cudaDebugRender) {
		runCudaRayTracer(params);
	} else {
		runCudaDebugRayTracer(params);
	}
	
	// Transfer result from Cuda texture to result framebuffer
	glUseProgram(mImpl->transferShader.handle());
	gl::setUniform(mImpl->transferShader, "uAccumulationPasses", !cudaDebugRender ? float(params.frameCount) : 1.0f);

	resultFB.bindViewportClearColorDepth(vec4(0.0f, 0.0f, 0.0f, 0.0f), 0.0f);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, mImpl->glTex);

	mImpl->fullscreenTriangle.render();

	// Return result
	RenderResult tmp;
	tmp.renderedRes = mTargetResolution;
	return tmp;
}

// CudaTracerRenderer: Protected virtual methods from BaseRenderer interface
// ------------------------------------------------------------------------------------------------

void CudaTracerRenderer::targetResolutionUpdated() noexcept
{
	mImpl->tracerParams.targetRes = mTargetResolution;

	glActiveTexture(GL_TEXTURE0);

	// Cleanup eventual previous texture and bindings
	if (mImpl->tracerParams.targetSurface != 0) {
		CHECK_CUDA_ERROR(cudaDestroySurfaceObject(mImpl->tracerParams.targetSurface));
		mImpl->tracerParams.targetSurface = 0;
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
	CHECK_CUDA_ERROR(cudaCreateSurfaceObject(&mImpl->tracerParams.targetSurface, &resDesc));

	if (mImpl->tracerParams.curandStates != nullptr) {
		cudaFree(mImpl->tracerParams.curandStates);
	}

	mImpl->tracerParams.numCurandStates = mTargetResolution.x * mTargetResolution.y;
	size_t curandStateBytes = mImpl->tracerParams.numCurandStates * sizeof(curandState);
	CHECK_CUDA_ERROR(cudaMalloc(&mImpl->tracerParams.curandStates, curandStateBytes));
	initCurand(mImpl->tracerParams);
}

} // namespace phe
