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

	Setting* cudaRenderMode = nullptr;
	CameraDef lastCamera;
	int32_t lastRenderMode = 0;
	uint32_t accumulationPasses = 0;

	// Holding the OpenGL Cuda surface data, surface object is in CudaTracerParams.
	GLuint glTex = 0;
	cudaGraphicsResource_t cudaResource = 0;
	cudaArray_t cudaArray = 0; // Probably no need to free, since memory is owned by OpenGL

	// Parameters for tracer
	DynArray<CudaBindlessTexture> textureWrappers;
	DynArray<cudaTextureObject_t> textureObjectHandles;
	CudaTracerParams tracerParams;

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
		CHECK_CUDA_ERROR(cudaDestroyTextureObject(tracerParams.staticBvhNodesTex));
		CHECK_CUDA_ERROR(cudaDestroyTextureObject(tracerParams.staticTriangleVerticesTex));
		CHECK_CUDA_ERROR(cudaFree(tracerParams.staticBvhNodes));
		CHECK_CUDA_ERROR(cudaFree(tracerParams.staticTriangleVertices));
		CHECK_CUDA_ERROR(cudaFree(tracerParams.staticTriangleDatas));

		// Static light sources
		CHECK_CUDA_ERROR(cudaFree(tracerParams.staticSphereLights));
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
	mImpl->cudaRenderMode = cfg.sanitizeInt("CudaTracer", "cudaRenderMode", 0, 0, 2);
	mImpl->lastRenderMode = mImpl->cudaRenderMode->intValue();
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
	const BVH& staticBvh = staticScene.bvh;

	// Copy static BVH to GPU
	BVHNode*& gpuBVHNodes = mImpl->tracerParams.staticBvhNodes;
	CHECK_CUDA_ERROR(cudaFree(gpuBVHNodes));
	size_t numBVHNodesBytes = staticBvh.nodes.size() * sizeof(BVHNode);
	CHECK_CUDA_ERROR(cudaMalloc(&gpuBVHNodes, numBVHNodesBytes));
	CHECK_CUDA_ERROR(cudaMemcpy(gpuBVHNodes, staticBvh.nodes.data(), numBVHNodesBytes, cudaMemcpyHostToDevice));

	// Create texture object for BVH
	CHECK_CUDA_ERROR(cudaDestroyTextureObject(mImpl->tracerParams.staticBvhNodesTex));

	cudaResourceDesc bvhNodesResDesc;
	memset(&bvhNodesResDesc, 0, sizeof(cudaResourceDesc));
	bvhNodesResDesc.resType = cudaResourceTypeLinear;
	bvhNodesResDesc.res.linear.devPtr = gpuBVHNodes;
	bvhNodesResDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
	bvhNodesResDesc.res.linear.desc.x = 32; // bits per channel
	bvhNodesResDesc.res.linear.desc.y = 32; // bits per channel
	bvhNodesResDesc.res.linear.desc.z = 32; // bits per channel
	bvhNodesResDesc.res.linear.desc.w = 32; // bits per channel
	bvhNodesResDesc.res.linear.sizeInBytes = numBVHNodesBytes;

	cudaTextureDesc bvhNodesTexDesc;
	memset(&bvhNodesTexDesc, 0, sizeof(cudaTextureDesc));
	bvhNodesTexDesc.readMode = cudaReadModeElementType;

	CHECK_CUDA_ERROR(cudaCreateTextureObject(&mImpl->tracerParams.staticBvhNodesTex, &bvhNodesResDesc,
	                                         &bvhNodesTexDesc, NULL));

	// Copy static triangle vertices to GPU
	TriangleVertices*& gpuTriangleVertices = mImpl->tracerParams.staticTriangleVertices;
	CHECK_CUDA_ERROR(cudaFree(gpuTriangleVertices));
	size_t numTriangleVertBytes = staticBvh.triangles.size() * sizeof(TriangleVertices);
	CHECK_CUDA_ERROR(cudaMalloc(&gpuTriangleVertices, numTriangleVertBytes));
	CHECK_CUDA_ERROR(cudaMemcpy(gpuTriangleVertices, staticBvh.triangles.data(), numTriangleVertBytes, cudaMemcpyHostToDevice));

	// Create texture object for triangle vertices
	CHECK_CUDA_ERROR(cudaDestroyTextureObject(mImpl->tracerParams.staticTriangleVerticesTex));

	cudaResourceDesc triVertsResDesc;
	memset(&triVertsResDesc, 0, sizeof(cudaResourceDesc));
	triVertsResDesc.resType = cudaResourceTypeLinear;
	triVertsResDesc.res.linear.devPtr = gpuTriangleVertices;
	triVertsResDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
	triVertsResDesc.res.linear.desc.x = 32; // bits per channel
	triVertsResDesc.res.linear.desc.y = 32; // bits per channel
	triVertsResDesc.res.linear.desc.z = 32; // bits per channel
	triVertsResDesc.res.linear.desc.w = 32; // bits per channel
	triVertsResDesc.res.linear.sizeInBytes = numTriangleVertBytes;

	cudaTextureDesc triVertsTexDesc;
	memset(&triVertsTexDesc, 0, sizeof(cudaTextureDesc));
	triVertsTexDesc.readMode = cudaReadModeElementType;

	CHECK_CUDA_ERROR(cudaCreateTextureObject(&mImpl->tracerParams.staticTriangleVerticesTex, &triVertsResDesc,
	                                         &triVertsTexDesc, NULL));

	// Copy static triangle datas to GPU
	TriangleData*& gpuTriangleDatas = mImpl->tracerParams.staticTriangleDatas;
	CHECK_CUDA_ERROR(cudaFree(gpuTriangleDatas));
	size_t numTriangleDatasBytes = staticBvh.triangleDatas.size() * sizeof(TriangleData);
	CHECK_CUDA_ERROR(cudaMalloc(&gpuTriangleDatas, numTriangleDatasBytes));
	CHECK_CUDA_ERROR(cudaMemcpy(gpuTriangleDatas, staticBvh.triangleDatas.data(), numTriangleDatasBytes, cudaMemcpyHostToDevice));

	// Copy static sphere lights to GPU
	SphereLight*& gpuSphereLights = mImpl->tracerParams.staticSphereLights;
	CHECK_CUDA_ERROR(cudaFree(gpuSphereLights));
	size_t numSphereLightBytes = staticScene.sphereLights.size() * sizeof(SphereLight);
	CHECK_CUDA_ERROR(cudaMalloc(&gpuSphereLights, numSphereLightBytes));
	CHECK_CUDA_ERROR(cudaMemcpy(gpuSphereLights, staticScene.sphereLights.data(), numSphereLightBytes, cudaMemcpyHostToDevice));
	mImpl->tracerParams.numStaticSphereLights = staticScene.sphereLights.size();
}

void CudaTracerRenderer::setDynObjectsForRendering(const DynArray<RawMesh>& meshes, const DynArray<mat4>& transforms) noexcept
{

}

RenderResult CudaTracerRenderer::render(Framebuffer& resultFB) noexcept
{
	// Calculate camera def in order to generate first rays
	vec2 resultRes = vec2(mTargetResolution);
	mImpl->tracerParams.cam = generateCameraDef(mMatrices.position, mMatrices.forward, mMatrices.up,
	                                            mMatrices.vertFovRad, resultRes);

	CudaTracerParams& params = mImpl->tracerParams;
	int32_t renderMode = mImpl->cudaRenderMode->intValue();

	// Check if accumulated color should be reset
	if (!approxEqual(mImpl->lastCamera.origin, params.cam.origin) ||
	    !approxEqual(mImpl->lastCamera.dir, params.cam.dir) ||
	    renderMode == 0 && mImpl->lastRenderMode != 0) {
		// Reset color buffer through GL command
		static const float BLACK[]{0.0f, 0.0f, 0.0f, 0.0f};
		glClearTexImage(mImpl->glTex, 0, GL_RGBA, GL_FLOAT, BLACK);

		glFinish(); // Potentially stalls GPU more than necessary

		mImpl->accumulationPasses = 0;
		mImpl->lastCamera = params.cam;
	}

	// Run CUDA ray tracer
	switch (renderMode) {
	case 0:
		cudaRayTrace(params);
		mImpl->accumulationPasses++;
		break;
	case 1:
		cudaHeatmapTrace(params);
		break;
	case 2:
		cudaCastRayTest(params);
		break;
	default:
		sfz_assert_debug(false);
		break;
	}

	if (renderMode != 0) {
		mImpl->accumulationPasses = 1;
	}

	mImpl->lastRenderMode = renderMode;

	// Transfer result from Cuda texture to result framebuffer
	glUseProgram(mImpl->transferShader.handle());
	gl::setUniform(mImpl->transferShader, "uAccumulationPasses", float(mImpl->accumulationPasses));

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

	// Clear texture
	static const float BLACK[]{ 0.0f, 0.0f, 0.0f, 0.0f };
	glClearTexImage(mImpl->glTex, 0, GL_RGBA, GL_FLOAT, BLACK);
	glFinish();
	mImpl->accumulationPasses = 0;

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

	// Clear allocated curandStates
	if (mImpl->tracerParams.curandStates != nullptr) {
		cudaFree(mImpl->tracerParams.curandStates);
	}

	// Allocate curandState for each pixel
	mImpl->tracerParams.numCurandStates = mTargetResolution.x * mTargetResolution.y;
	size_t curandStateBytes = mImpl->tracerParams.numCurandStates * sizeof(curandState);
	CHECK_CUDA_ERROR(cudaMalloc(&mImpl->tracerParams.curandStates, curandStateBytes));

	auto timeSeed = static_cast<unsigned long long>(time(nullptr));
	initCurand(mImpl->tracerParams, timeSeed);
}

} // namespace phe
