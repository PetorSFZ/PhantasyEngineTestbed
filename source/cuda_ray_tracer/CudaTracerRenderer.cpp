// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#include "CudaTracerRenderer.hpp"

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>

#include <sfz/gl/IncludeOpenGL.hpp>
#include <cuda_gl_interop.h>

#include <sfz/containers/DynArray.hpp>
#include <sfz/containers/StackString.hpp>
#include <sfz/gl/Program.hpp>
#include <sfz/memory/New.hpp>
#include <sfz/util/IO.hpp>

#include <phantasy_engine/deferred_renderer/GLTexture.hpp>
#include <phantasy_engine/deferred_renderer/SSBO.hpp>
#include <phantasy_engine/config/GlobalConfig.hpp>
#include <phantasy_engine/deferred_renderer/GLModel.hpp>
#include <phantasy_engine/rendering/FullscreenTriangle.hpp>

#include "CudaBindlessTexture.hpp"
#include "CudaBuffer.hpp"
#include "CudaGLInterop.hpp"
#include "CudaHelpers.hpp"
#include "CudaTextureBuffer.hpp"
#include "kernels/GenSecondaryShadowRaysKernel.hpp"
#include "kernels/InitCurandKernel.hpp"
#include "kernels/MaterialKernel.hpp"
#include "kernels/InterpretRayHitKernel.hpp"
#include "kernels/PetorShading.hpp"
#include "kernels/ProcessGBufferKernel.hpp"
#include "kernels/RayCastKernel.hpp"
#include "kernels/ShadeSecondaryHit.hpp"

namespace phe {

using namespace sfz;
using sfz::gl::Program;

// CudaTracerRendererImpl
// ------------------------------------------------------------------------------------------------

class CudaTracerRendererImpl final {
public:
	// Settings
	Setting* rayCastPerfTest;
	Setting* rayCastPerfTestPrimaryRays;
	Setting* rayCastPerfTestPersistentThreads;
	Setting* petorShading;

	// The device properties of the used CUDA device
	int glDeviceIndex;
	cudaDeviceProp glDeviceProperties;

	// OpenGL FullscreenTriangle
	FullscreenTriangle fullscreenTriangle;

	// OpenGL shaders
	Program gbufferGenShader, transferShader;

	// Cuda OpenGL interop textures and framebuffers
	CudaGLGBuffer gbuffer;
	CudaGLTexture cudaResultTex;

	// OpenGL models for static and dynamic scene
	DynArray<GLModel> staticGLModels;
	DynArray<GLModel> dynamicGLModels;

	// OpenGL materials & textures
	DynArray<GLTexture> glTextures;
	SSBO texturesSSBO;
	SSBO materialsSSBO;

	// Cuda materials & textures
	CudaTextureBuffer<Material> materials;
	DynArray<CudaBindlessTexture> textureWrappers;
	CudaBuffer<cudaTextureObject_t> cudaTextures;

	// Cuda static geometry
	CudaTextureBuffer<BVHNode> staticBvhNodes;
	CudaTextureBuffer<TriangleVertices> staticTriangleVertices;
	CudaBuffer<TriangleData> staticTriangleDatas;

	// Cuda static light sources
	CudaBuffer<SphereLight> staticSphereLights;

	// Shading buffers
	CudaBuffer<PathState> pathStates;
	CudaBuffer<vec3> lightContributions;

	// Raycast input and output buffers
	const uint32_t NUM_RAYS_PER_PIXEL_PER_BATCH = 1;
	CudaBuffer<RayIn> rayBuffer;
	CudaBuffer<RayIn> shadowRayBuffer;
	CudaBuffer<RayHit> rayResultBuffer;
	CudaBuffer<RayHitInfo> rayHitInfoBuffer;
	CudaBuffer<bool> shadowRayResultBuffer;

	// CUDA RNG state
	CudaBuffer<curandState> randStates;

	// PetorShading stuff
	CudaBuffer<RayIn> petorShadingSecondaryRayBuffer;
	CudaBuffer<RayHit> petorShadingRayHitBuffer;
	CudaBuffer<RayHitInfo> petorShadingRayHitInfoBuffer;
	CudaBuffer<RayIn> petorShadingSecondaryShadowRayBuffer;
	CudaBuffer<bool> petorShadingShadowRayInLightBuffer;
	CudaBuffer<IncomingLight> petorShadingIncomingLightBuffer;

	CudaTracerRendererImpl() noexcept
	{
		// Initialize settings
		GlobalConfig& cfg = GlobalConfig::instance();
		rayCastPerfTest = cfg.sanitizeBool("CudaTracer", "rayCastPerfTest", false);
		rayCastPerfTestPrimaryRays = cfg.sanitizeBool("CudaTracer", "rayCastPerfTestPrimaryRays", false);
		rayCastPerfTestPersistentThreads = cfg.sanitizeBool("CudaTracer", "rayCastPerfTestPersistentThreads", true);
		petorShading = cfg.sanitizeBool("CudaTracer", "petorShading", false);

		// Initialize cuda with the same device that is bound to the OpenGL context
		unsigned int deviceCount = 0;
		CHECK_CUDA_ERROR(cudaGLGetDevices(&deviceCount, &glDeviceIndex, 1, cudaGLDeviceListCurrentFrame));
		CHECK_CUDA_ERROR(cudaSetDevice(glDeviceIndex));

		// Get device properties 
		CHECK_CUDA_ERROR(cudaGetDeviceProperties(&glDeviceProperties, glDeviceIndex));

		// Print device properties
		printf("CUDA device index %i, properties:\n", glDeviceIndex);
		printf("multiProcessorCount: %i\n", glDeviceProperties.multiProcessorCount);
		printf("maxThreadsPerMultiProcessor: %i\n", glDeviceProperties.maxThreadsPerMultiProcessor);
		printf("maxThreadsPerBlock: %i\n\n", glDeviceProperties.maxThreadsPerBlock);

		// Load OpenGL shaders
		{
			StackString256 shadersPath;
			shadersPath.printf("%sresources/shaders/cuda_tracer_renderer/", basePath());

			gbufferGenShader = Program::fromFile(shadersPath.str, "gbuffer_gen.vert",
			                                     "gbuffer_gen.frag", [](uint32_t shaderProgram) {
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
};

// CudaTracerRenderer: Constructors & destructors
// ------------------------------------------------------------------------------------------------

CudaTracerRenderer::CudaTracerRenderer() noexcept
{
	mImpl = sfz_new<CudaTracerRendererImpl>();
}

CudaTracerRenderer::~CudaTracerRenderer() noexcept
{
	sfz_delete(mImpl);
	cudaDeviceReset();
}

// CudaTracerRenderer: Virtual methods from BaseRenderer interface
// ------------------------------------------------------------------------------------------------

void CudaTracerRenderer::setMaterialsAndTextures(const DynArray<Material>& materials,
                                                 const DynArray<RawImage>& textures) noexcept
{
	// Destroy old values
	mImpl->glTextures.clear();
	mImpl->texturesSSBO.destroy();
	mImpl->materialsSSBO.destroy();

	// Allocate SSBO memory and upload compact materials
	uint32_t numMaterialBytes = materials.size() * sizeof(Material);
	mImpl->materialsSSBO.create(numMaterialBytes);
	mImpl->materialsSSBO.uploadData(materials.data(), numMaterialBytes);

	// Create GLTextures
	DynArray<uint64_t> tmpBindlessTextureHandles;
	tmpBindlessTextureHandles.setCapacity(textures.size());
	mImpl->glTextures.setCapacity(textures.size());
	for (const RawImage& img : textures) {
		mImpl->glTextures.add(GLTexture(img));
		tmpBindlessTextureHandles.add(mImpl->glTextures.last().bindlessHandle());
	}

	// Allocate SSBO memory and upload bindless texture handles
	uint32_t numBindlessTextureHandleBytes = tmpBindlessTextureHandles.size() * sizeof(uint64_t);
	mImpl->texturesSSBO.create(numBindlessTextureHandleBytes);
	mImpl->texturesSSBO.uploadData(tmpBindlessTextureHandles.data(), numBindlessTextureHandleBytes);


	// Copy materials to Cuda
	mImpl->materials = CudaTextureBuffer<Material>(materials.data(), materials.size());

	// Create Cuda bindless textures
	mImpl->textureWrappers.clear();
	mImpl->textureWrappers.setCapacity(textures.size());
	DynArray<cudaTextureObject_t> tmpTextureHandles;
	tmpTextureHandles.setCapacity(textures.size());
	
	for (const RawImage& texture : textures) {
		CudaBindlessTexture tmp;
		tmp.load(texture);
		mImpl->textureWrappers.add(std::move(tmp));
		tmpTextureHandles.add(mImpl->textureWrappers.last().textureObject());
	}

	// Upload texture handles to Cuda
	mImpl->cudaTextures = CudaBuffer<cudaTextureObject_t>(tmpTextureHandles.data(),
	                                                      tmpTextureHandles.size());
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

	// Upload static bvh to Cuda
	const BVH& bvh = staticScene.bvh;
	mImpl->staticBvhNodes = CudaTextureBuffer<BVHNode>(bvh.nodes.data(), bvh.nodes.size());
	mImpl->staticTriangleVertices = CudaTextureBuffer<TriangleVertices>(bvh.triangleVerts.data(),
	                                                                    bvh.triangleVerts.size());
	mImpl->staticTriangleDatas = CudaBuffer<TriangleData>(bvh.triangleDatas.data(),
	                                                      bvh.triangleVerts.size());

	// Upload static lights to Cuda
	mImpl->staticSphereLights = CudaBuffer<SphereLight>(staticScene.sphereLights.data(),
	                                                    staticScene.sphereLights.size());
}
	
void CudaTracerRenderer::setDynamicMeshes(const DynArray<RawMesh>& meshes) noexcept
{
	// Create dynamic OpenGL models
	DynArray<GLModel>& glModels = mImpl->dynamicGLModels;
	glModels.clear();
	for (const RawMesh& mesh : meshes) {
		glModels.add(GLModel(mesh));
	}

	// TODO: Dynamic bvh?
}

/*template<class T>
void stupidGpuSend(T*& gpuPtr, const DynArray<T>& cpuData) noexcept
{
	CHECK_CUDA_ERROR(cudaFree(gpuPtr));
	size_t numBytes = cpuData.size() * sizeof(T);
	CHECK_CUDA_ERROR(cudaMalloc(&gpuPtr, numBytes));
	CHECK_CUDA_ERROR(cudaMemcpy(gpuPtr, cpuData.data(), numBytes, cudaMemcpyHostToDevice));
}


static void sendDynamicBvhToCuda()
{
	//const OuterBVH
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
//	stupidGpuSend(mImpl->tracerParams.dynamic, bvhTriangleDataTex);
}*/

void CudaTracerRenderer::addDynamicMesh(const RawMesh& mesh) noexcept
{
	sfz::error("CudaTracerRenderer: addDynamicMesh() not implemented");
}

RenderResult CudaTracerRenderer::render(Framebuffer& resultFB,
                                        const DynArray<DynObject>& objects,
                                        const DynArray<SphereLight>& lights) noexcept
{
	const mat4 viewMatrix = mCamera.viewMatrix();
	const mat4 projMatrix = mCamera.projMatrix(mTargetResolution);

	// GBuffer generation
	// --------------------------------------------------------------------------------------------

	auto& gbuffer = mImpl->gbuffer;
	auto& gbufferGenShader = mImpl->gbufferGenShader;

	glDisable(GL_BLEND);
	glEnable(GL_CULL_FACE);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_GREATER); // reversed-z

	gbuffer.bindViewportClearColorDepth();
	gbufferGenShader.useProgram();

	const mat4 modelMatrix = identityMatrix4<float>();
	gl::setUniform(gbufferGenShader, "uProjMatrix", projMatrix);
	gl::setUniform(gbufferGenShader, "uViewMatrix", viewMatrix);

	// Bind SSBOs
	mImpl->materialsSSBO.bind(0);
	mImpl->texturesSSBO.bind(1);

	const int modelMatrixLoc = glGetUniformLocation(gbufferGenShader.handle(), "uModelMatrix");
	const int normalMatrixLoc = glGetUniformLocation(gbufferGenShader.handle(), "uNormalMatrix");

	// For the static scene the model matrix should be identity
	// normalMatrix = inverse(transpose(modelMatrix)) since we want worldspace normals
	gl::setUniform(modelMatrixLoc, identityMatrix4<float>());
	gl::setUniform(normalMatrixLoc, identityMatrix4<float>());

	// Simply skip the draw calls if we are performance testing the ray cast kernel with primary rays
	if (!(mImpl->rayCastPerfTest->boolValue() && mImpl->rayCastPerfTestPrimaryRays->boolValue())) {
		
		for (const GLModel& model : mImpl->staticGLModels) {
			model.draw();
		}

		for (const DynObject& obj : objects) {
			gl::setUniform(modelMatrixLoc, obj.transform);
			gl::setUniform(normalMatrixLoc, inverse(transpose(obj.transform)));
			const GLModel& model = mImpl->dynamicGLModels[obj.meshIndex];
			model.draw();
		}
	}

	// Wait for OpenGL to finish rendering GBuffer before we start using it in Cuda
	glFinish();

	sfz_assert_release((mTargetResolution.x % 8) == 0);
	sfz_assert_release((mTargetResolution.y % 8) == 0);

	// Ray cast performance test (don't touch)
	if (mImpl->rayCastPerfTest->boolValue()) {
		
		// Whether to generate primary or secondary rays for the test
		if (mImpl->rayCastPerfTestPrimaryRays->boolValue()) {
			CameraDef camDef = generateCameraDef(mCamera.pos(), mCamera.dir(), mCamera.up(),
			                                     mCamera.verticalFov() * DEG_TO_RAD(),
			                                     vec2(mTargetResolution));
			launchGenPrimaryRaysKernel(mImpl->rayBuffer.cudaPtr(), camDef, mTargetResolution);
		}
		else {
			launchGenSecondaryRaysKernel(mImpl->rayBuffer.cudaPtr(), mCamera.pos(),
			                             mTargetResolution, mImpl->gbuffer.positionSurfaceCuda(),
			                             mImpl->gbuffer.normalSurfaceCuda());
		}

		RayCastKernelInput rayCastInput;
		rayCastInput.bvhNodes = mImpl->staticBvhNodes.cudaTexture();
		rayCastInput.triangleVerts = mImpl->staticTriangleVertices.cudaTexture();
		rayCastInput.numRays = mTargetResolution.x * mTargetResolution.y;
		rayCastInput.rays = mImpl->rayBuffer.cudaPtr();

		if (mImpl->rayCastPerfTestPersistentThreads->boolValue()) {
			launchRayCastKernel(rayCastInput, mImpl->rayResultBuffer.cudaPtr(), mImpl->glDeviceProperties);
		} else {
			launchRayCastNoPersistenceKernel(rayCastInput, mImpl->rayResultBuffer.cudaPtr(), mImpl->glDeviceProperties);
		}

		// Write hits to screen
		launchWriteRayHitsToScreenKernel(mImpl->cudaResultTex.cudaSurface(), mTargetResolution,
		                                 mImpl->rayResultBuffer.cudaPtr());
	}

	// Petorshading path
	else if (mImpl->petorShading->boolValue()) {
		
		uint32_t numRays = (mTargetResolution.x * mTargetResolution.y) / 4; // TODO: More dynamic

		// Generate secondary rays
		ProcessGBufferGenRaysInput input1;
		input1.camPos = mCamera.pos();
		input1.res = mTargetResolution;
		input1.posTex = mImpl->gbuffer.positionSurfaceCuda();
		input1.normalTex = mImpl->gbuffer.normalSurfaceCuda();
		input1.albedoTex = mImpl->gbuffer.albedoSurfaceCuda();
		input1.materialTex = mImpl->gbuffer.materialSurfaceCuda();
		launchProcessGBufferGenRaysKernel(input1, mImpl->petorShadingSecondaryRayBuffer.cudaPtr());

		// Ray cast secondary rays
		RayCastKernelInput rayCastInput;
		rayCastInput.bvhNodes = mImpl->staticBvhNodes.cudaTexture();
		rayCastInput.triangleVerts = mImpl->staticTriangleVertices.cudaTexture();
		rayCastInput.numRays = numRays;
		rayCastInput.rays = mImpl->petorShadingSecondaryRayBuffer.cudaPtr();
		launchRayCastKernel(rayCastInput, mImpl->petorShadingRayHitBuffer.cudaPtr(), mImpl->glDeviceProperties);
		
		// "Interpret" secondary rays (i.e. load their materials)
		InterpretRayHitKernelInput interpretRayHitInput;
		interpretRayHitInput.rays = mImpl->petorShadingSecondaryRayBuffer.cudaPtr();
		interpretRayHitInput.rayHits = mImpl->petorShadingRayHitBuffer.cudaPtr();
		interpretRayHitInput.numRays = numRays;
		interpretRayHitInput.materialsTex = mImpl->materials.cudaTexture();
		interpretRayHitInput.textures = mImpl->cudaTextures.cudaPtr();
		interpretRayHitInput.staticTriangleDatas = mImpl->staticTriangleDatas.cudaPtr();
		launchInterpretRayHitKernel(interpretRayHitInput, mImpl->petorShadingRayHitInfoBuffer.cudaPtr(),
		                            mImpl->glDeviceProperties);

		// Generate shadow rays for secondary hits
		GenSecondaryShadowRaysKernelInput secondaryShadowRayInput;
		secondaryShadowRayInput.rayHitInfos = mImpl->petorShadingRayHitInfoBuffer.cudaPtr();
		secondaryShadowRayInput.numRayHitInfos = numRays;
		secondaryShadowRayInput.staticSphereLights = mImpl->staticSphereLights.cudaPtr();
		secondaryShadowRayInput.numStaticSphereLights = mImpl->staticSphereLights.size();
		launchGenSecondaryShadowRaysKernel(secondaryShadowRayInput, mImpl->petorShadingSecondaryShadowRayBuffer.cudaPtr());
		
		// Cast secondary shadow rays
		rayCastInput.rays = mImpl->petorShadingSecondaryShadowRayBuffer.cudaPtr();
		rayCastInput.numRays = numRays * mImpl->staticSphereLights.size();
		launchShadowRayCastKernel(rayCastInput, mImpl->petorShadingShadowRayInLightBuffer.cudaPtr(), mImpl->glDeviceProperties);

		// Shade secondary hits and generate outgoing light to primary hits
		ShadeSecondaryHitKernelInput shadeSecondaryHitInput;
		shadeSecondaryHitInput.secondaryRays = mImpl->petorShadingSecondaryRayBuffer.cudaPtr();
		shadeSecondaryHitInput.rayHitInfos = mImpl->petorShadingRayHitInfoBuffer.cudaPtr();
		shadeSecondaryHitInput.numRayHitInfos = numRays;
		shadeSecondaryHitInput.res = mTargetResolution;
		shadeSecondaryHitInput.numIncomingLightsPerPixel = mImpl->staticSphereLights.size() + 1;
		shadeSecondaryHitInput.staticSphereLights = mImpl->staticSphereLights.cudaPtr();
		shadeSecondaryHitInput.numStaticSphereLights = mImpl->staticSphereLights.size();
		shadeSecondaryHitInput.shadowRayResults = mImpl->petorShadingShadowRayInLightBuffer.cudaPtr();
		launchShadeSecondaryHitKernel(shadeSecondaryHitInput, mImpl->petorShadingIncomingLightBuffer.cudaPtr());

		// GatherRaysShadeKernel

		GatherRaysShadeKernelInput input2;

		input2.camPos = mCamera.pos();
		
		input2.res = mTargetResolution;
		input2.posTex = mImpl->gbuffer.positionSurfaceCuda();
		input2.normalTex = mImpl->gbuffer.normalSurfaceCuda();
		input2.albedoTex = mImpl->gbuffer.albedoSurfaceCuda();
		input2.materialTex = mImpl->gbuffer.materialSurfaceCuda();

		input2.incomingLights = mImpl->petorShadingIncomingLightBuffer.cudaPtr();
		input2.numIncomingLights = mImpl->staticSphereLights.size() + 1;
		
		input2.staticSphereLights = mImpl->staticSphereLights.cudaPtr();
		input2.numStaticSphereLights = mImpl->staticSphereLights.size();
		
		launchGatherRaysShadeKernel(input2, mImpl->cudaResultTex.cudaSurface());
	}

	// Normal path
	else {
		uint32_t numTargetPixels = mTargetResolution.x * mTargetResolution.y;
		uint32_t numPrimaryShadowRays = numTargetPixels * mImpl->staticSphereLights.size();

		vec2i halfTargetResolution = mTargetResolution / 2;
		uint32_t numSecondaryRays = halfTargetResolution.x * halfTargetResolution.y;
		uint32_t numSecondaryShadowRays = numSecondaryRays * mImpl->staticSphereLights.size();

		launchInitPathStatesKernel(halfTargetResolution, mImpl->pathStates.cudaPtr());

		// Shade first hit and create shadow rays + next ray in path
		GBufferMaterialKernelInput gBufferMaterialKernelInput;
		gBufferMaterialKernelInput.res = mTargetResolution;
		gBufferMaterialKernelInput.camPos = mCamera.pos();
		gBufferMaterialKernelInput.randStates = mImpl->randStates.cudaPtr();
		gBufferMaterialKernelInput.shadowRays = mImpl->shadowRayBuffer.cudaPtr();
		gBufferMaterialKernelInput.lightContributions = mImpl->lightContributions.cudaPtr();
		gBufferMaterialKernelInput.staticSphereLights = mImpl->staticSphereLights.cudaPtr();
		gBufferMaterialKernelInput.numStaticSphereLights = mImpl->staticSphereLights.size();
		gBufferMaterialKernelInput.posTex = mImpl->gbuffer.positionSurfaceCuda();
		gBufferMaterialKernelInput.normalTex = mImpl->gbuffer.normalSurfaceCuda();
		gBufferMaterialKernelInput.albedoTex = mImpl->gbuffer.albedoSurfaceCuda();
		gBufferMaterialKernelInput.materialTex = mImpl->gbuffer.materialSurfaceCuda();
		launchGBufferMaterialKernel(gBufferMaterialKernelInput);

		// Determine if first hit is in shadow
		RayCastKernelInput shadowRayCastInput;
		shadowRayCastInput.bvhNodes = mImpl->staticBvhNodes.cudaTexture();
		shadowRayCastInput.triangleVerts = mImpl->staticTriangleVertices.cudaTexture();
		shadowRayCastInput.numRays = numPrimaryShadowRays;
		shadowRayCastInput.rays = mImpl->shadowRayBuffer.cudaPtr();
		launchShadowRayCastKernel(shadowRayCastInput, mImpl->shadowRayResultBuffer.cudaPtr(), mImpl->glDeviceProperties);

		// Use shading result if not in shadow
		ShadowLogicKernelInput primaryShadowLogicKernelInput;
		primaryShadowLogicKernelInput.surface = mImpl->cudaResultTex.cudaSurface();
		primaryShadowLogicKernelInput.res = mTargetResolution;
		primaryShadowLogicKernelInput.resolutionScale = 1;
		primaryShadowLogicKernelInput.addToSurface = false;
		primaryShadowLogicKernelInput.shadowRayHits = mImpl->shadowRayResultBuffer.cudaPtr();
		primaryShadowLogicKernelInput.pathStates = mImpl->pathStates.cudaPtr();
		primaryShadowLogicKernelInput.lightContributions = mImpl->lightContributions.cudaPtr();
		primaryShadowLogicKernelInput.numStaticSphereLights = mImpl->staticSphereLights.size();
		launchShadowLogicKernel(primaryShadowLogicKernelInput);

		// Create secondary ray
		CreateSecondaryRaysKernelInput createSecondaryRaysKernelInput;
		createSecondaryRaysKernelInput.res = halfTargetResolution;
		createSecondaryRaysKernelInput.camPos = mCamera.pos();
		createSecondaryRaysKernelInput.pathStates = mImpl->pathStates.cudaPtr();
		createSecondaryRaysKernelInput.extensionRays = mImpl->rayBuffer.cudaPtr();
		createSecondaryRaysKernelInput.randStates = mImpl->randStates.cudaPtr();
		createSecondaryRaysKernelInput.posTex = mImpl->gbuffer.positionSurfaceCuda();
		createSecondaryRaysKernelInput.normalTex = mImpl->gbuffer.normalSurfaceCuda();
		createSecondaryRaysKernelInput.albedoTex = mImpl->gbuffer.albedoSurfaceCuda();
		createSecondaryRaysKernelInput.materialTex = mImpl->gbuffer.materialSurfaceCuda();
		launchCreateSecondaryRaysKernel(createSecondaryRaysKernelInput);

		// Cast secondary ray
		RayCastKernelInput secondaryRayCastInput;
		secondaryRayCastInput.bvhNodes = mImpl->staticBvhNodes.cudaTexture();
		secondaryRayCastInput.triangleVerts = mImpl->staticTriangleVertices.cudaTexture();
		secondaryRayCastInput.numRays = numSecondaryRays;
		secondaryRayCastInput.rays = mImpl->rayBuffer.cudaPtr();
		launchRayCastKernel(secondaryRayCastInput, mImpl->rayResultBuffer.cudaPtr(), mImpl->glDeviceProperties);

		// Get material information from ray hits
		InterpretRayHitKernelInput interpretRayHitInput;
		interpretRayHitInput.rays = mImpl->rayBuffer.cudaPtr();
		interpretRayHitInput.rayHits = mImpl->rayResultBuffer.cudaPtr();
		interpretRayHitInput.numRays = numSecondaryRays;
		interpretRayHitInput.materialsTex = mImpl->materials.cudaTexture();
		interpretRayHitInput.textures = mImpl->cudaTextures.cudaPtr();
		interpretRayHitInput.staticTriangleDatas = mImpl->staticTriangleDatas.cudaPtr();
		launchInterpretRayHitKernel(interpretRayHitInput, mImpl->rayHitInfoBuffer.cudaPtr(),
		                            mImpl->glDeviceProperties);

		// Shade secondary ray hit and create shadow rays
		MaterialKernelInput materialKernelInput;
		materialKernelInput.res = halfTargetResolution;
		materialKernelInput.pathStates = mImpl->pathStates.cudaPtr();
		materialKernelInput.randStates = mImpl->randStates.cudaPtr();
		materialKernelInput.rays = mImpl->rayBuffer.cudaPtr();
		materialKernelInput.rayHitInfos = mImpl->rayHitInfoBuffer.cudaPtr();
		materialKernelInput.shadowRays = mImpl->shadowRayBuffer.cudaPtr();
		materialKernelInput.lightContributions = mImpl->lightContributions.cudaPtr();
		materialKernelInput.staticSphereLights = mImpl->staticSphereLights.cudaPtr();
		materialKernelInput.numStaticSphereLights = mImpl->staticSphereLights.size();
		launchMaterialKernel(materialKernelInput);

		// Determine if secondary hit is in shadow
		RayCastKernelInput secondaryShadowRayCastInput;
		secondaryShadowRayCastInput.bvhNodes = mImpl->staticBvhNodes.cudaTexture();
		secondaryShadowRayCastInput.triangleVerts = mImpl->staticTriangleVertices.cudaTexture();
		secondaryShadowRayCastInput.numRays = numSecondaryShadowRays;
		secondaryShadowRayCastInput.rays = mImpl->shadowRayBuffer.cudaPtr();
		launchShadowRayCastKernel(secondaryShadowRayCastInput, mImpl->shadowRayResultBuffer.cudaPtr(), mImpl->glDeviceProperties);

		// Use shading result if not in shadow
		ShadowLogicKernelInput secondaryShadowLogicKernelInput;
		secondaryShadowLogicKernelInput.surface = mImpl->cudaResultTex.cudaSurface();
		secondaryShadowLogicKernelInput.res = mTargetResolution;
		secondaryShadowLogicKernelInput.resolutionScale = 2;
		secondaryShadowLogicKernelInput.addToSurface = true;
		secondaryShadowLogicKernelInput.shadowRayHits = mImpl->shadowRayResultBuffer.cudaPtr();
		secondaryShadowLogicKernelInput.pathStates = mImpl->pathStates.cudaPtr();
		secondaryShadowLogicKernelInput.lightContributions = mImpl->lightContributions.cudaPtr();
		secondaryShadowLogicKernelInput.numStaticSphereLights = mImpl->staticSphereLights.size();
		launchShadowLogicKernel(secondaryShadowLogicKernelInput);
	}

	// Transfer result to resultFB
	// --------------------------------------------------------------------------------------------

	glDisable(GL_BLEND);
	glEnable(GL_CULL_FACE);
	glDisable(GL_DEPTH_TEST);

	// Copy result to resultFB
	resultFB.bindViewport();
	mImpl->transferShader.useProgram();
	gl::setUniform(mImpl->transferShader, "uSrcTexture", 0);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, mImpl->cudaResultTex.glTexture());
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

	/*if (objects.size() > 0) {
		uint32_t numSubBvhs = objects.size();
		DynArray<BVH> bvhs = DynArray<BVH>(0, numSubBvhs);

		for (const DynObject& object : objects) {
			BVH bvh = createDynamicBvh(mImpl->dynMeshes[object.meshIndex], object.transform);
			sanitizeBVH(bvh);
			bvhs.add(bvh);
		}

		mImpl->dynamicBvh = createOuterBvh(bvhs);
		sendDynamicBvhToCuda();
	}*/
}

// CudaTracerRenderer: Protected virtual methods from BaseRenderer interface
// ------------------------------------------------------------------------------------------------

void CudaTracerRenderer::targetResolutionUpdated() noexcept
{
	// Update GBuffer resolution
	mImpl->gbuffer = CudaGLGBuffer(mTargetResolution);

	// Allocate new result texture
	mImpl->cudaResultTex = CudaGLTexture(mTargetResolution);

	// Allocate memory for rays
	uint32_t numTargetPixels = mTargetResolution.x * mTargetResolution.y;
	uint32_t numPrimaryShadowRays = numTargetPixels * mImpl->staticSphereLights.size();
	uint32_t numSecondaryRays = numTargetPixels / 4;
	uint32_t numSecondaryShadowRays = numSecondaryRays * mImpl->staticSphereLights.size();
	mImpl->rayBuffer.destroy();
	mImpl->rayBuffer.create(numSecondaryRays);
	mImpl->shadowRayBuffer.destroy();
	mImpl->shadowRayBuffer.create(numPrimaryShadowRays);
	mImpl->rayResultBuffer.destroy();
	mImpl->rayResultBuffer.create(numSecondaryRays);
	mImpl->shadowRayResultBuffer.destroy();
	mImpl->shadowRayResultBuffer.create(numPrimaryShadowRays);
	mImpl->rayHitInfoBuffer.destroy();
	mImpl->rayHitInfoBuffer.create(numSecondaryRays);
	mImpl->pathStates.destroy();
	mImpl->pathStates.create(numSecondaryRays);
	mImpl->lightContributions.destroy();
	mImpl->lightContributions.create(numPrimaryShadowRays);
	mImpl->randStates.destroy();
	mImpl->randStates.create(numTargetPixels);

	launchInitCurandKernel(mTargetResolution, mImpl->randStates.cudaPtr());

	// Petorshading ray memory allocation
	uint32_t petorShadingNumRays = (mImpl->staticSphereLights.size() + 1)
	                             * mTargetResolution.x * mTargetResolution.y;
	mImpl->petorShadingSecondaryRayBuffer.destroy();
	mImpl->petorShadingSecondaryRayBuffer.create(petorShadingNumRays);
	mImpl->petorShadingRayHitBuffer.destroy();
	mImpl->petorShadingRayHitBuffer.create(petorShadingNumRays);
	mImpl->petorShadingRayHitInfoBuffer.destroy();
	mImpl->petorShadingRayHitInfoBuffer.create(petorShadingNumRays);
	mImpl->petorShadingSecondaryShadowRayBuffer.destroy();
	mImpl->petorShadingSecondaryShadowRayBuffer.create(petorShadingNumRays);
	mImpl->petorShadingShadowRayInLightBuffer.destroy();
	mImpl->petorShadingShadowRayInLightBuffer.create(petorShadingNumRays);
	mImpl->petorShadingIncomingLightBuffer.destroy();
	mImpl->petorShadingIncomingLightBuffer.create(petorShadingNumRays);
}

} // namespace phe
