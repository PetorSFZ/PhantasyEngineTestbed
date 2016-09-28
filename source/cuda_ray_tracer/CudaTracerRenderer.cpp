// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#include "CudaTracerRenderer.hpp"

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <sfz/containers/DynArray.hpp>
#include <sfz/containers/StackString.hpp>
#include <sfz/gl/IncludeOpenGL.hpp>
#include <sfz/gl/Program.hpp>
#include <sfz/memory/New.hpp>
#include <sfz/util/IO.hpp>

#include <phantasy_engine/config/GlobalConfig.hpp>
#include <phantasy_engine/deferred_renderer/GLModel.hpp>
#include <phantasy_engine/rendering/FullscreenTriangle.hpp>

#include "CudaHelpers.hpp"

/*#include <chrono>


#include <sfz/math/MathHelpers.hpp>
#include <sfz/math/MatrixSupport.hpp>
#include <sfz/util/IO.hpp>
#include <sfz/geometry/AABB.hpp>


#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include <phantasy_engine/RayTracerCommon.hpp>


#include "CudaArray.hpp"
#include "CudaBindlessTexture.hpp"
#include "CudaHelpers.hpp"
#include "CudaTracer.cuh"
#include "RayCastKernel.cuh"*/

namespace phe {

using namespace sfz;
using sfz::gl::Program;

// Statics
// ------------------------------------------------------------------------------------------------

static const uint32_t GBUFFER_POSITION = 0u; // uv "u" coordinate stored in w
static const uint32_t GBUFFER_NORMAL = 1u; // uv "v" coordinate stored in w
static const uint32_t GBUFFER_MATERIAL_ID = 2u;

// CudaTracerRendererImpl
// ------------------------------------------------------------------------------------------------

class CudaTracerRendererImpl final {
public:

	// Settings
	Setting* cudaDeviceIndex = nullptr;

	// The device properties of the used CUDA device
	cudaDeviceProp deviceProperties;

	// OpenGL FullscreenTriangle
	FullscreenTriangle fullscreenTriangle;

	// OpenGL shaders and framebuffers
	Program gbufferGenShader, transferShader;
	Framebuffer gbuffer;

	// OpenGL models for static and dynamic scene
	DynArray<GLModel> staticGLModels;
	DynArray<GLModel> dynamicGLModels;


	CudaTracerRendererImpl() noexcept
	{
		// Initialize settings
		GlobalConfig& cfg = GlobalConfig::instance();
		cudaDeviceIndex = cfg.sanitizeInt("CudaTracer", "deviceIndex", 0, 0, 16);

		// Initialize cuda and get device properties
		CHECK_CUDA_ERROR(cudaSetDevice(cudaDeviceIndex->intValue()));
		CHECK_CUDA_ERROR(cudaGetDeviceProperties(&deviceProperties, cudaDeviceIndex->intValue()));
		
		// Load OpenGL shaders
		{
			StackString256 shadersPath;
			shadersPath.printf("%sresources/shaders/cuda_tracer_renderer/", basePath());

			gbufferGenShader = Program::fromFile(shadersPath.str, "gbuffer_gen.vert", "gbuffer_gen.frag",
				[](uint32_t shaderProgram) {
				glBindAttribLocation(shaderProgram, 0, "inPosition");
				glBindAttribLocation(shaderProgram, 1, "inNormal");
				glBindAttribLocation(shaderProgram, 2, "inUV");
				glBindAttribLocation(shaderProgram, 3, "inMaterialId");
			});

			transferShader = Program::postProcessFromFile(shadersPath.str, "transfer.frag");
		}
	}
	
	~CudaTracerRendererImpl() noexcept
	{

	}

	/*gl::Program transferShader;
	FullscreenTriangle fullscreenTriangle;

	Setting* cudaRenderMode = nullptr;
	Setting* cudaDeviceIndex = nullptr;
	CameraDef lastCamera;
	int32_t lastRenderMode = 0;
	uint32_t accumulationPasses = 0;

	// The device properties of the used CUDA device
	cudaDeviceProp deviceProperties;

	// Holding the OpenGL Cuda surface data, surface object is in CudaTracerParams.
	GLuint glTex = 0;
	cudaGraphicsResource_t cudaResource = 0;
	cudaArray_t cudaArray = 0; // Probably no need to free, since memory is owned by OpenGL

	BVH staticBvh; // TODO: Move out to static scene
	OuterBVH dynamicBvh;
	DynArray<RawMesh> dynMeshes;
	
	// Parameters for tracer
	DynArray<CudaBindlessTexture> textureWrappers;
	DynArray<cudaTextureObject_t> textureObjectHandles;
	CudaTracerParams tracerParams;

	// Temp
	RayIn* gpuRaysBuffer = nullptr;
	RayHit* gpuRayHitsBuffer = nullptr;

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

		// Temp
		CHECK_CUDA_ERROR(cudaFree(gpuRaysBuffer));
		CHECK_CUDA_ERROR(cudaFree(gpuRayHitsBuffer));
	}*/
};

// CudaTracerRenderer: Constructors & destructors
// ------------------------------------------------------------------------------------------------

CudaTracerRenderer::CudaTracerRenderer() noexcept
{
	mImpl = sfz_new<CudaTracerRendererImpl>();

	/*StackString128 shadersPath;
	shadersPath.printf("%sresources/shaders/", basePath());
	mImpl->transferShader = gl::Program::postProcessFromFile(shadersPath.str, "cuda_transfer.frag");
	glUseProgram(mImpl->transferShader.handle());
	gl::setUniform(mImpl->transferShader, "uSrcTexture", 0);

	GlobalConfig& cfg = GlobalConfig::instance();
	mImpl->cudaRenderMode = cfg.sanitizeInt("CudaTracer", "cudaRenderMode", 0, 0, 3);
	mImpl->lastRenderMode = mImpl->cudaRenderMode->intValue();
	mImpl->cudaDeviceIndex = cfg.sanitizeInt("CudaTracer", "deviceIndex", 0, 0, 32);

	// Initialize cuda and get device properties
	CHECK_CUDA_ERROR(cudaSetDevice(mImpl->cudaDeviceIndex->intValue()));
	CHECK_CUDA_ERROR(cudaGetDeviceProperties(&mImpl->deviceProperties, mImpl->cudaDeviceIndex->intValue()));

	printf("multiProcessorCount: %i\n", mImpl->deviceProperties.multiProcessorCount);
	printf("maxThreadsPerMultiProcessor: %i\n", mImpl->deviceProperties.maxThreadsPerMultiProcessor);
	printf("maxThreadsPerBlock: %i\n", mImpl->deviceProperties.maxThreadsPerBlock);*/
}

CudaTracerRenderer::~CudaTracerRenderer() noexcept
{
	sfz_delete(mImpl);
}

// CudaTracerRenderer: Virtual methods from BaseRenderer interface
// ------------------------------------------------------------------------------------------------

void CudaTracerRenderer::setMaterialsAndTextures(const DynArray<Material>& materials,
                                                 const DynArray<RawImage>& textures) noexcept
{
	/*// Copy materials to CUDA
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
	mImpl->tracerParams.numTextures = textures.size();*/
}

void CudaTracerRenderer::addTexture(const RawImage& texture) noexcept
{
	sfz::error("CudaTracerRenderer: addTexture() not implemented");
}

void CudaTracerRenderer::addMaterial(const Material& material) noexcept
{
	sfz::error("CudaTracerRenderer: addMaterial() not implemented");
}

void CudaTracerRenderer::setStaticScene(const StaticScene& staticScene) noexcept
{
	// Create static OpenGL models
	DynArray<GLModel>& glModels = mImpl->staticGLModels;
	glModels.clear();
	for (const RawMesh& mesh : staticScene.meshes) {
		glModels.add(GLModel(mesh));
	}

	/*const BVH& staticBvh = staticScene.bvh;

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
	size_t numTriangleVertBytes = staticBvh.triangleVerts.size() * sizeof(TriangleVertices);
	CHECK_CUDA_ERROR(cudaMalloc(&gpuTriangleVertices, numTriangleVertBytes));
	CHECK_CUDA_ERROR(cudaMemcpy(gpuTriangleVertices, staticBvh.triangleVerts.data(), numTriangleVertBytes, cudaMemcpyHostToDevice));

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
	mImpl->tracerParams.numStaticSphereLights = staticScene.sphereLights.size();*/
}
	
void CudaTracerRenderer::setDynamicMeshes(const DynArray<RawMesh>& meshes) noexcept
{
	// Create dynamic OpenGL models
	DynArray<GLModel>& glModels = mImpl->dynamicGLModels;
	glModels.clear();
	for (const RawMesh& mesh : meshes) {
		glModels.add(GLModel(mesh));
	}

	/*mImpl->dynMeshes = meshes;*/
}

template<class T>
void stupidGpuSend(T*& gpuPtr, const DynArray<T>& cpuData) noexcept
{
	/*CHECK_CUDA_ERROR(cudaFree(gpuPtr));
	size_t numBytes = cpuData.size() * sizeof(T);
	CHECK_CUDA_ERROR(cudaMalloc(&gpuPtr, numBytes));
	CHECK_CUDA_ERROR(cudaMemcpy(gpuPtr, cpuData.data(), numBytes, cudaMemcpyHostToDevice));*/
}


static void sendDynamicBvhToCuda()
{
	/*//const OuterBVH
	OuterBVH& dynamicOuterBvh = mImpl->dynamicBvh;

	// Copy dynamic BVH to GPU
	stupidGpuSend(mImpl->tracerParams.dynamicOuterBvhNodes, dynamicOuterBvh.nodes);

	uint32_t numInnerBvhs = dynamicOuterBvh.bvhs.size();

	DynArray<BVHNode*> bvhNodePointers = DynArray<BVHNode*>(numInnerBvhs, nullptr, numInnerBvhs);
	DynArray<cudaTextureObject_t> bvhNodeTex = DynArray<cudaTextureObject_t>(numInnerBvhs, 0, numInnerBvhs);
	DynArray<TriangleData*> bvhTriangleDataPointers = DynArray<TriangleData*>(numInnerBvhs, nullptr, numInnerBvhs);
	DynArray<TriangleVertices*> bvhTriangleVerticesPointers = DynArray<TriangleVertices*>(numInnerBvhs, nullptr, numInnerBvhs);
	DynArray<cudaTextureObject_t> bvhTriangleVerticesTex = DynArray<cudaTextureObject_t>(numInnerBvhs, 0, numInnerBvhs);

	for (int i = 0; i < numInnerBvhs; i++) {
		BVH& bvh = dynamicOuterBvh.bvhs[i];
		stupidGpuSend(bvhNodePointers[i], bvh.nodes);
		stupidGpuSend(bvhTriangleDataPointers[i], bvh.triangleDatas);

		// Create texture object for BVH
		CHECK_CUDA_ERROR(cudaDestroyTextureObject(bvhNodeTex[i]));

		cudaResourceDesc bvhNodesResDesc;
		memset(&bvhNodesResDesc, 0, sizeof(cudaResourceDesc));
		bvhNodesResDesc.resType = cudaResourceTypeLinear;
		bvhNodesResDesc.res.linear.devPtr = bvhNodePointers[i];
		bvhNodesResDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
		bvhNodesResDesc.res.linear.desc.x = 32; // bits per channel
		bvhNodesResDesc.res.linear.desc.y = 32; // bits per channel
		bvhNodesResDesc.res.linear.desc.z = 32; // bits per channel
		bvhNodesResDesc.res.linear.desc.w = 32; // bits per channel
		bvhNodesResDesc.res.linear.sizeInBytes = bvh.nodes.size() * sizeof(BVHNode);

		cudaTextureDesc bvhNodesTexDesc;
		memset(&bvhNodesTexDesc, 0, sizeof(cudaTextureDesc));
		bvhNodesTexDesc.readMode = cudaReadModeElementType;

		CHECK_CUDA_ERROR(cudaCreateTextureObject(&bvhNodeTex[i], &bvhNodesResDesc, &bvhNodesTexDesc, NULL));

		stupidGpuSend(bvhTriangleVerticesPointers[i], bvh.triangleVerts);

		// Create texture object for triangle vertices
		CHECK_CUDA_ERROR(cudaDestroyTextureObject(bvhTriangleVerticesTex[i]));

		cudaResourceDesc triVertsResDesc;
		memset(&triVertsResDesc, 0, sizeof(cudaResourceDesc));
		triVertsResDesc.resType = cudaResourceTypeLinear;
		triVertsResDesc.res.linear.devPtr = bvhTriangleVerticesPointers[i];
		triVertsResDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
		triVertsResDesc.res.linear.desc.x = 32; // bits per channel
		triVertsResDesc.res.linear.desc.y = 32; // bits per channel
		triVertsResDesc.res.linear.desc.z = 32; // bits per channel
		triVertsResDesc.res.linear.desc.w = 32; // bits per channel
		triVertsResDesc.res.linear.sizeInBytes = bvh.triangleVerts.size() * sizeof(TriangleVertices);

		cudaTextureDesc triVertsTexDesc;
		memset(&triVertsTexDesc, 0, sizeof(cudaTextureDesc));
		triVertsTexDesc.readMode = cudaReadModeElementType;

		CHECK_CUDA_ERROR(cudaCreateTextureObject(&bvhTriangleVerticesTex[i], &triVertsResDesc, &triVertsTexDesc, NULL));
	}

	stupidGpuSend(mImpl->tracerParams.dynamicBvhNodes, bvhNodePointers);
	stupidGpuSend(mImpl->tracerParams.dynamicBvhNodesTex, bvhNodeTex);
	stupidGpuSend(mImpl->tracerParams.dynamicTriangleDatas, bvhTriangleDataPointers);
	stupidGpuSend(mImpl->tracerParams.dynamicTriangleVertices, bvhTriangleVerticesPointers);
	stupidGpuSend(mImpl->tracerParams.dynamicTriangleVerticesTex, bvhTriangleVerticesTex);
	mImpl->tracerParams.numDynBvhs = numInnerBvhs;
//	stupidGpuSend(mImpl->tracerParams.dynamic, bvhTriangleDataTex);*/
}

void CudaTracerRenderer::addDynamicMesh(const RawMesh& mesh) noexcept
{
	sfz::error("CudaTracerRenderer: addDynamicMesh() not implemented");
}

RenderResult CudaTracerRenderer::render(Framebuffer& resultFB,
                                        const DynArray<DynObject>& objects,
                                        const DynArray<SphereLight>& lights) noexcept
{
	const mat4 viewMatrix = mMatrices.headMatrix * mMatrices.originMatrix;
	const mat4 projMatrix = mMatrices.projMatrix;

	// GBuffer generation
	// --------------------------------------------------------------------------------------------

	auto& gbuffer = mImpl->gbuffer;
	auto& gbufferGenShader = mImpl->gbufferGenShader;

	glDisable(GL_BLEND);
	glEnable(GL_CULL_FACE);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_GREATER); // reversed-z

	gbuffer.bindViewportClearColorDepth(vec2i(0), mTargetResolution, vec4(0.0f), 0.0f);
	gbufferGenShader.useProgram();

	const mat4 modelMatrix = identityMatrix4<float>();
	gl::setUniform(gbufferGenShader, "uProjMatrix", projMatrix);
	gl::setUniform(gbufferGenShader, "uViewMatrix", viewMatrix);

	const int modelMatrixLoc = glGetUniformLocation(gbufferGenShader.handle(), "uModelMatrix");
	const int normalMatrixLoc = glGetUniformLocation(gbufferGenShader.handle(), "uNormalMatrix");

	// For the static scene the model matrix should be identity
	// normalMatrix = inverse(transpose(modelMatrix)) since we want worldspace normals
	gl::setUniform(modelMatrixLoc, identityMatrix4<float>());
	gl::setUniform(normalMatrixLoc, identityMatrix4<float>());

	for (const GLModel& model : mImpl->staticGLModels) {
		model.draw();
	}

	for (const DynObject& obj : objects) {
		gl::setUniform(modelMatrixLoc, obj.transform);
		gl::setUniform(normalMatrixLoc, inverse(transpose(obj.transform)));
		const GLModel& model = mImpl->dynamicGLModels[obj.meshIndex];
		model.draw();
	}

	// Transfer result to resultFB
	// --------------------------------------------------------------------------------------------

	glDisable(GL_BLEND);
	glEnable(GL_CULL_FACE);
	glDisable(GL_DEPTH_TEST);

	// Copy color result to resultFB
	resultFB.bindViewport();
	mImpl->transferShader.useProgram();
	gl::setUniform(mImpl->transferShader, "uSrcTexture", 0);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, gbuffer.texture(GBUFFER_NORMAL));
	mImpl->fullscreenTriangle.render();

	// Copy depth from GBuffer to resultFB
	glBindFramebuffer(GL_READ_FRAMEBUFFER, gbuffer.fbo());
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, resultFB.fbo());

	glBlitFramebuffer(0, 0, mTargetResolution.x, mTargetResolution.y,
	                  0, 0, mTargetResolution.x, mTargetResolution.y,
	                  GL_DEPTH_BUFFER_BIT, GL_NEAREST);

	// Return result
	RenderResult tmp;
	tmp.renderedRes = mTargetResolution;
	return tmp;

	/*// Calculate camera def in order to generate first rays
	vec2 resultRes = vec2(mTargetResolution);
	mImpl->tracerParams.cam = generateCameraDef(mMatrices.position, mMatrices.forward, mMatrices.up,
	                                            mMatrices.vertFovRad, resultRes);

	if (objects.size() > 0) {
		uint32_t numSubBvhs = objects.size();
		DynArray<BVH> bvhs = DynArray<BVH>(0, numSubBvhs);

		for (const DynObject& object : objects) {
			BVH bvh = createDynamicBvh(mImpl->dynMeshes[object.meshIndex], object.transform);
			sanitizeBVH(bvh);
			bvhs.add(bvh);
		}

		mImpl->dynamicBvh = createOuterBvh(bvhs);
		sendDynamicBvhToCuda();
	}

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
	case 3:
		launchGenPrimaryRaysKernel(mImpl->gpuRaysBuffer, params.cam, mTargetResolution);
		launchRayCastKernel(mImpl->tracerParams.staticBvhNodesTex, mImpl->tracerParams.staticTriangleVerticesTex,
		                    mImpl->gpuRaysBuffer, mImpl->gpuRayHitsBuffer, mTargetResolution.x * mTargetResolution.y, mImpl->deviceProperties);
		launchWriteRayHitsToScreenKernel(mImpl->tracerParams.targetSurface, mImpl->tracerParams.targetRes, mImpl->gpuRayHitsBuffer);
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

	mImpl->fullscreenTriangle.render();*/

	
}

// CudaTracerRenderer: Protected virtual methods from BaseRenderer interface
// ------------------------------------------------------------------------------------------------

void CudaTracerRenderer::targetResolutionUpdated() noexcept
{
	// Update GBuffer resolution
	// TODO: CHANGE R_INT_U8 TO R_INT_U16
	mImpl->gbuffer = gl::FramebufferBuilder(mTargetResolution)
	    .addDepthTexture(gl::FBDepthFormat::F32, gl::FBTextureFiltering::NEAREST)
	    .addTexture(GBUFFER_POSITION, gl::FBTextureFormat::RGBA_F32, gl::FBTextureFiltering::NEAREST)
	    .addTexture(GBUFFER_NORMAL, gl::FBTextureFormat::RGBA_F32, gl::FBTextureFiltering::LINEAR)
	    .addTexture(GBUFFER_MATERIAL_ID, gl::FBTextureFormat::R_INT_U8, gl::FBTextureFiltering::NEAREST)
	    .build();


	/*mImpl->tracerParams.targetRes = mTargetResolution;

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

	// Allocate ray infos for each pixel
	CHECK_CUDA_ERROR(cudaFree(mImpl->gpuRaysBuffer));
	size_t numRayBytes = mTargetResolution.x * mTargetResolution.y * sizeof(RayIn);
	CHECK_CUDA_ERROR(cudaMalloc(&mImpl->gpuRaysBuffer, numRayBytes));

	CHECK_CUDA_ERROR(cudaFree(mImpl->gpuRayHitsBuffer));
	size_t numRayHitBytes = mTargetResolution.x * mTargetResolution.y * sizeof(RayHit);
	CHECK_CUDA_ERROR(cudaMalloc(&mImpl->gpuRayHitsBuffer, numRayHitBytes));*/
}

} // namespace phe
