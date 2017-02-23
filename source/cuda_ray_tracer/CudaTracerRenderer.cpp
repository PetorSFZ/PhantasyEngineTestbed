// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#include "CudaTracerRenderer.hpp"

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>

#include <sfz/gl/IncludeOpenGL.hpp>
#include <cuda_gl_interop.h>

#include <sfz/containers/DynArray.hpp>
#include <sfz/Cuda.hpp>
#include <sfz/gl/Program.hpp>
#include <sfz/math/ProjectionMatrices.hpp>
#include <sfz/memory/New.hpp>
#include <sfz/strings/StackString.hpp>
#include <sfz/util/IO.hpp>

#include <phantasy_engine/deferred_renderer/GLTexture.hpp>
#include <phantasy_engine/deferred_renderer/SSBO.hpp>
#include <phantasy_engine/config/GlobalConfig.hpp>
#include <phantasy_engine/deferred_renderer/GLModel.hpp>
#include <phantasy_engine/rendering/FullscreenTriangle.hpp>

#include "CudaBindlessTexture.hpp"
#include "CudaGLInterop.hpp"
#include "CudaTextureBuffer.hpp"
#include "kernels/GenSecondaryRaysKernel.hpp"
#include "kernels/GenSecondaryShadowRaysKernel.hpp"
#include "kernels/InitCurandKernel.hpp"
#include "kernels/MaterialKernel.hpp"
#include "kernels/InterpretRayHitKernel.hpp"
#include "kernels/PetorShading.hpp"
#include "kernels/ProcessGBufferKernel.hpp"
#include "kernels/RayCastKernel.hpp"
#include "kernels/ShadeSecondaryHit.hpp"
#include "ShadowCubeMapCudaGL.hpp"

namespace phe {

using namespace sfz;
using namespace sfz::gl;

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
	Program shadowMapGenShader, gbufferGenShader, transferShader;

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
	cuda::Buffer<cudaTextureObject_t> cudaTextures;

	// Cuda static geometry
	CudaTextureBuffer<BVHNode> staticBvhNodes;
	CudaTextureBuffer<TriangleVertices> staticTriangleVertices;
	cuda::Buffer<TriangleData> staticTriangleDatas;

	// Cuda static light sources
	cuda::Buffer<SphereLight> staticSphereLights;
	DynArray<ShadowCubeMapCudaGL> staticShadowMaps;

	// Shading buffers
	cuda::Buffer<PathState> pathStates;
	cuda::Buffer<vec3> lightContributions;

	// Raycast input and output buffers
	const uint32_t NUM_RAYS_PER_PIXEL_PER_BATCH = 1;
	cuda::Buffer<RayIn> rayBuffer;
	cuda::Buffer<RayIn> shadowRayBuffer;
	cuda::Buffer<RayHit> rayResultBuffer;
	cuda::Buffer<RayHitInfo> rayHitInfoBuffer;
	cuda::Buffer<bool> shadowRayResultBuffer;

	// CUDA RNG state
	cuda::Buffer<curandState> randStates;

	// PetorShading stuff
	cuda::Buffer<RayIn> psSecondaryRayBuffer;
	cuda::Buffer<RayHit> psSecondaryRayHitBuffer;
	cuda::Buffer<RayHitInfo> psSecondaryRayHitInfoBuffer;

	cuda::Buffer<RayIn> psShadowRayBuffer;
	cuda::Buffer<bool> psShadowRayInLightBuffer;

	cuda::Buffer<RayIn> psSecondaryShadowRayBuffer;
	cuda::Buffer<bool> psSecondaryShadowRayInLightBuffer;

	cuda::Buffer<IncomingLight> psLightStreamBuffer;

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

			shadowMapGenShader = Program::fromFile(shadersPath.str, "cuda_gl_shadow_map_gen.vert",
			                                       "cuda_gl_shadow_map_gen.geom", "cuda_gl_shadow_map_gen.frag",
			[](uint32_t shaderProgram) {
				glBindAttribLocation(shaderProgram, 0, "inPosition");
			});

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
	mImpl = sfzNewDefault<CudaTracerRendererImpl>();
}

CudaTracerRenderer::~CudaTracerRenderer() noexcept
{
	sfzDeleteDefault(mImpl);
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
	mImpl->cudaTextures = cuda::Buffer<cudaTextureObject_t>(tmpTextureHandles.data(),
	                                                        tmpTextureHandles.size());
}

void CudaTracerRenderer::addTexture(const RawImage&) noexcept
{
	sfz::error("CudaTracerRenderer: addTexture() not implemented");
}

void CudaTracerRenderer::addMaterial(const Material&) noexcept
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
	mImpl->staticTriangleDatas = cuda::Buffer<TriangleData>(bvh.triangleDatas.data(),
	                                                        bvh.triangleVerts.size());

	// Upload static lights to Cuda
	mImpl->staticSphereLights = cuda::Buffer<SphereLight>(staticScene.sphereLights.data(),
	                                                      staticScene.sphereLights.size());

	// Static shadow maps
	mImpl->staticShadowMaps.clear();
	for (const SphereLight& light : staticScene.sphereLights) {
		ShadowCubeMapCudaGL shadowMap(2096u);

		// Note: We don't use reverse-z here, unecessary since we overwrite the depth anyway

		glDisable(GL_BLEND);
		glEnable(GL_CULL_FACE);
		glEnable(GL_DEPTH_TEST);
		glDepthFunc(GL_LESS);

		glBindFramebuffer(GL_FRAMEBUFFER, shadowMap.fbo());
		glViewport(0, 0, shadowMap.resolution().x, shadowMap.resolution().y);
		glClearDepth(1.0f);
		glClear(GL_DEPTH_BUFFER_BIT);
		
		mImpl->shadowMapGenShader.useProgram();

		// Model matrix is identity since static scene is defined in world space
		gl::setUniform(mImpl->shadowMapGenShader, "uModelMatrix", mat44::identity());

		// View and projection matrices for each face
		const mat4 projMatrix = mat44::scaling3(1.0f, -1.0f, 1.0f) * sfz::perspectiveProjectionVkD3d(90.0f, 1.0f, 0.01f, light.range);
		mat4 viewProjMatrices[6];
		viewProjMatrices[0] = projMatrix * sfz::viewMatrixGL(light.pos, vec3(1.0f, 0.0f, 0.0f), vec3(0.0f, -1.0f, 0.0f)); // right
		viewProjMatrices[1] = projMatrix * sfz::viewMatrixGL(light.pos, vec3(-1.0f, 0.0f, 0.0f), vec3(0.0f, -1.0f, 0.0f)); // left
		viewProjMatrices[2] = projMatrix * sfz::viewMatrixGL(light.pos, vec3(0.0f, 1.0f, 0.0f), vec3(0.0f, 0.0f, 1.0f)); // top
		viewProjMatrices[3] = projMatrix * sfz::viewMatrixGL(light.pos, vec3(0.0f, -1.0f, 0.0f), vec3(0.0f, 0.0f, -1.0f)); // bottom
		viewProjMatrices[4] = projMatrix * sfz::viewMatrixGL(light.pos, vec3(0.0f, 0.0f, 1.0f), vec3(0.0f, -1.0f, 0.0f)); // near
		viewProjMatrices[5] = projMatrix * sfz::viewMatrixGL(light.pos, vec3(0.0f, 0.0f, -1.0f), vec3(0.0f, -1.0f, 0.0f)); // far
		gl::setUniform(mImpl->shadowMapGenShader, "uViewProjMatrices", viewProjMatrices, 6);

		// Light information
		gl::setUniform(mImpl->shadowMapGenShader, "uLightPosWS", light.pos);
		gl::setUniform(mImpl->shadowMapGenShader, "uLightRange", light.range);

		for (const GLModel& model : mImpl->staticGLModels) {
			model.draw();
		}

		mImpl->staticShadowMaps.add(std::move(shadowMap));
	}
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

void CudaTracerRenderer::addDynamicMesh(const RawMesh&) noexcept
{
	sfz::error("CudaTracerRenderer: addDynamicMesh() not implemented");
}

RenderResult CudaTracerRenderer::render(const RenderComponent* renderComponents, uint32_t numComponents,
                                        const DynArray<SphereLight>&) noexcept
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

	const mat4 modelMatrix = mat44::identity();
	gl::setUniform(gbufferGenShader, "uProjMatrix", projMatrix);
	gl::setUniform(gbufferGenShader, "uViewMatrix", viewMatrix);

	// Bind SSBOs
	mImpl->materialsSSBO.bind(0);
	mImpl->texturesSSBO.bind(1);

	const int modelMatrixLoc = glGetUniformLocation(gbufferGenShader.handle(), "uModelMatrix");
	const int normalMatrixLoc = glGetUniformLocation(gbufferGenShader.handle(), "uNormalMatrix");
	const int worldVelocityLoc = glGetUniformLocation(gbufferGenShader.handle(), "uWorldVelocity");

	// For the static scene the model matrix should be identity
	// normalMatrix = inverse(transpose(modelMatrix)) since we want worldspace normals
	gl::setUniform(modelMatrixLoc, mat44::identity());
	gl::setUniform(normalMatrixLoc, mat44::identity());

	// Simply skip the draw calls if we are performance testing the ray cast kernel with primary rays
	if (!(mImpl->rayCastPerfTest->boolValue() && mImpl->rayCastPerfTestPrimaryRays->boolValue())) {

		// Set velocity of static geometry to 0
		gl::setUniform(worldVelocityLoc, vec3(0.0f));
		for (const GLModel& model : mImpl->staticGLModels) {
			model.draw();
		}

		for (uint32_t i = 0; i < numComponents; i++) {
			const RenderComponent& obj = renderComponents[i];

			gl::setUniform(modelMatrixLoc, obj.transform);
			gl::setUniform(normalMatrixLoc, inverse(transpose(viewMatrix * obj.transform)));
			gl::setUniform(worldVelocityLoc, obj.velocity);
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
			                                     mCamera.verticalFov() * DEG_TO_RAD,
			                                     vec2(mTargetResolution));
			launchGenPrimaryRaysKernel(mImpl->rayBuffer.data(), camDef, mTargetResolution);
		}
		else {
			launchGenSecondaryRaysKernel(mImpl->rayBuffer.data(), mCamera.pos(),
			                             mTargetResolution, mImpl->gbuffer.positionSurfaceCuda(),
			                             mImpl->gbuffer.normalSurfaceCuda());
		}

		RayCastKernelInput rayCastInput;
		rayCastInput.bvhNodes = mImpl->staticBvhNodes.cudaTexture();
		rayCastInput.triangleVerts = mImpl->staticTriangleVertices.cudaTexture();
		rayCastInput.numRays = mTargetResolution.x * mTargetResolution.y;
		rayCastInput.rays = mImpl->rayBuffer.data();

		if (mImpl->rayCastPerfTestPersistentThreads->boolValue()) {
			launchRayCastKernel(rayCastInput, mImpl->rayResultBuffer.data(), mImpl->glDeviceProperties);
		} else {
			launchRayCastNoPersistenceKernel(rayCastInput, mImpl->rayResultBuffer.data(), mImpl->glDeviceProperties);
		}

		// Write hits to screen
		launchWriteRayHitsToScreenKernel(mImpl->cudaResultTex.cudaSurface(), mTargetResolution,
		                                 mImpl->rayResultBuffer.data());
	}

	// Petorshading path
	else if (mImpl->petorShading->boolValue()) {
		
		uint32_t numRays = (mTargetResolution.x * mTargetResolution.y) / 4; // TODO: More dynamic

		const uint32_t numFullResPixels = mTargetResolution.x * mTargetResolution.y;
		const uint32_t numHalfResPixels = numFullResPixels / 4;
		const uint32_t totalNumLights = mImpl->staticSphereLights.capacity();


		// Generate secondary rays and shadow rays
		GenSecondaryRaysKernelInput genSecondaryRaysInput;

		genSecondaryRaysInput.camPos = mCamera.pos();
		
		genSecondaryRaysInput.res = vec2u(mTargetResolution);
		genSecondaryRaysInput.posTex = mImpl->gbuffer.positionSurfaceCuda();
		genSecondaryRaysInput.normalTex = mImpl->gbuffer.normalSurfaceCuda();
		genSecondaryRaysInput.albedoTex = mImpl->gbuffer.albedoSurfaceCuda();
		genSecondaryRaysInput.materialTex = mImpl->gbuffer.materialSurfaceCuda();

		genSecondaryRaysInput.staticSphereLights = mImpl->staticSphereLights.data();
		genSecondaryRaysInput.numStaticSphereLights = mImpl->staticSphereLights.capacity();

		launchGenSecondaryRaysKernel(genSecondaryRaysInput, mImpl->psSecondaryRayBuffer.data(),
		                             mImpl->psShadowRayBuffer.data());

		// Ray cast secondary rays
		RayCastKernelInput rayCastInput;
		rayCastInput.bvhNodes = mImpl->staticBvhNodes.cudaTexture();
		rayCastInput.triangleVerts = mImpl->staticTriangleVertices.cudaTexture();
		rayCastInput.numRays = numRays;
		rayCastInput.rays = mImpl->psSecondaryRayBuffer.data();
		launchRayCastKernel(rayCastInput, mImpl->psSecondaryRayHitBuffer.data(), mImpl->glDeviceProperties);
		
		// "Interpret" secondary rays (i.e. load their materials)
		InterpretRayHitKernelInput interpretRayHitInput;
		interpretRayHitInput.rays = mImpl->psSecondaryRayBuffer.data();
		interpretRayHitInput.rayHits = mImpl->psSecondaryRayHitBuffer.data();
		interpretRayHitInput.numRays = numRays;
		interpretRayHitInput.materialsTex = mImpl->materials.cudaTexture();
		interpretRayHitInput.textures = mImpl->cudaTextures.data();
		interpretRayHitInput.staticTriangleDatas = mImpl->staticTriangleDatas.data();
		launchInterpretRayHitKernel(interpretRayHitInput, mImpl->psSecondaryRayHitInfoBuffer.data(),
		                            mImpl->glDeviceProperties);

		// Generate shadow rays for secondary hits
		GenSecondaryShadowRaysKernelInput secondaryShadowRayInput;
		secondaryShadowRayInput.rayHitInfos = mImpl->psSecondaryRayHitInfoBuffer.data();
		secondaryShadowRayInput.numRayHitInfos = numRays;
		secondaryShadowRayInput.staticSphereLights = mImpl->staticSphereLights.data();
		secondaryShadowRayInput.numStaticSphereLights = mImpl->staticSphereLights.capacity();
		launchGenSecondaryShadowRaysKernel(secondaryShadowRayInput, mImpl->psSecondaryShadowRayBuffer.data());
		
		// Cast secondary shadow rays
		rayCastInput.rays = mImpl->psSecondaryShadowRayBuffer.data();
		rayCastInput.numRays = numRays * mImpl->staticSphereLights.capacity();
		launchShadowRayCastKernel(rayCastInput, mImpl->psSecondaryShadowRayInLightBuffer.data(), mImpl->glDeviceProperties);

		// Shade secondary hits and generate outgoing light to primary hits
		ShadeSecondaryHitKernelInput shadeSecondaryHitInput;
		shadeSecondaryHitInput.secondaryRays = mImpl->psSecondaryRayBuffer.data();
		shadeSecondaryHitInput.rayHitInfos = mImpl->psSecondaryRayHitInfoBuffer.data();
		shadeSecondaryHitInput.numRayHitInfos = numRays;
		shadeSecondaryHitInput.res = mTargetResolution;
		shadeSecondaryHitInput.numIncomingLightsPerPixel = mImpl->staticSphereLights.capacity() + 1;
		shadeSecondaryHitInput.staticSphereLights = mImpl->staticSphereLights.data();
		shadeSecondaryHitInput.numStaticSphereLights = mImpl->staticSphereLights.capacity();
		shadeSecondaryHitInput.shadowRayResults = mImpl->psSecondaryShadowRayInLightBuffer.data();
		launchShadeSecondaryHitKernel(shadeSecondaryHitInput, mImpl->psLightStreamBuffer.data());

		// Cast primary shadow rays
		rayCastInput.rays = mImpl->psShadowRayBuffer.data();
		rayCastInput.numRays = numFullResPixels * totalNumLights;
		launchShadowRayCastKernel(rayCastInput, mImpl->psShadowRayInLightBuffer.data(),
		                          mImpl->glDeviceProperties);

		// GatherRaysShadeKernel

		GatherRaysShadeKernelInput input2;

		input2.camPos = mCamera.pos();
		
		input2.res = mTargetResolution;
		input2.posTex = mImpl->gbuffer.positionSurfaceCuda();
		input2.normalTex = mImpl->gbuffer.normalSurfaceCuda();
		input2.albedoTex = mImpl->gbuffer.albedoSurfaceCuda();
		input2.materialTex = mImpl->gbuffer.materialSurfaceCuda();

		input2.incomingLights = mImpl->psLightStreamBuffer.data();
		input2.numIncomingLights = mImpl->staticSphereLights.capacity() + 1;
		
		input2.staticSphereLights = mImpl->staticSphereLights.data();
		input2.inLights = mImpl->psShadowRayInLightBuffer.data();
		input2.numStaticSphereLights = mImpl->staticSphereLights.capacity();
		
		launchGatherRaysShadeKernel(input2, mImpl->cudaResultTex.cudaSurface());
	}

	// Normal path
	else {
		uint32_t numTargetPixels = mTargetResolution.x * mTargetResolution.y;
		uint32_t numPrimaryShadowRays = numTargetPixels * mImpl->staticSphereLights.capacity();

		vec2u targetResolution = vec2u(mTargetResolution);
		vec2u halfTargetResolution = targetResolution / 2u;
		uint32_t numSecondaryRays = halfTargetResolution.x * halfTargetResolution.y;
		uint32_t numSecondaryShadowRays = numSecondaryRays * mImpl->staticSphereLights.capacity();

		launchInitPathStatesKernel(halfTargetResolution, mImpl->pathStates.data());

		// Shade first hit and create shadow rays + next ray in path
		GBufferMaterialKernelInput gBufferMaterialKernelInput;
		gBufferMaterialKernelInput.res = targetResolution;
		gBufferMaterialKernelInput.camPos = mCamera.pos();
		gBufferMaterialKernelInput.randStates = mImpl->randStates.data();
		gBufferMaterialKernelInput.shadowRays = mImpl->shadowRayBuffer.data();
		gBufferMaterialKernelInput.lightContributions = mImpl->lightContributions.data();
		gBufferMaterialKernelInput.staticSphereLights = mImpl->staticSphereLights.data();
		gBufferMaterialKernelInput.numStaticSphereLights = mImpl->staticSphereLights.capacity();
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
		shadowRayCastInput.rays = mImpl->shadowRayBuffer.data();
		launchShadowRayCastKernel(shadowRayCastInput, mImpl->shadowRayResultBuffer.data(), mImpl->glDeviceProperties);

		// Use shading result if not in shadow
		ShadowLogicKernelInput primaryShadowLogicKernelInput;
		primaryShadowLogicKernelInput.surface = mImpl->cudaResultTex.cudaSurface();
		primaryShadowLogicKernelInput.res = targetResolution;
		primaryShadowLogicKernelInput.resolutionScale = 1;
		primaryShadowLogicKernelInput.addToSurface = false;
		primaryShadowLogicKernelInput.shadowRayHits = mImpl->shadowRayResultBuffer.data();
		primaryShadowLogicKernelInput.pathStates = mImpl->pathStates.data();
		primaryShadowLogicKernelInput.lightContributions = mImpl->lightContributions.data();
		primaryShadowLogicKernelInput.numStaticSphereLights = mImpl->staticSphereLights.capacity();
		launchShadowLogicKernel(primaryShadowLogicKernelInput);

		// Create secondary ray
		CreateSecondaryRaysKernelInput createSecondaryRaysKernelInput;
		createSecondaryRaysKernelInput.res = halfTargetResolution;
		createSecondaryRaysKernelInput.camPos = mCamera.pos();
		createSecondaryRaysKernelInput.pathStates = mImpl->pathStates.data();
		createSecondaryRaysKernelInput.extensionRays = mImpl->rayBuffer.data();
		createSecondaryRaysKernelInput.randStates = mImpl->randStates.data();
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
		secondaryRayCastInput.rays = mImpl->rayBuffer.data();
		launchRayCastKernel(secondaryRayCastInput, mImpl->rayResultBuffer.data(), mImpl->glDeviceProperties);

		// Get material information from ray hits
		InterpretRayHitKernelInput interpretRayHitInput;
		interpretRayHitInput.rays = mImpl->rayBuffer.data();
		interpretRayHitInput.rayHits = mImpl->rayResultBuffer.data();
		interpretRayHitInput.numRays = numSecondaryRays;
		interpretRayHitInput.materialsTex = mImpl->materials.cudaTexture();
		interpretRayHitInput.textures = mImpl->cudaTextures.data();
		interpretRayHitInput.staticTriangleDatas = mImpl->staticTriangleDatas.data();
		launchInterpretRayHitKernel(interpretRayHitInput, mImpl->rayHitInfoBuffer.data(),
		                            mImpl->glDeviceProperties);

		// Shade secondary ray hit and create shadow rays
		MaterialKernelInput materialKernelInput;
		materialKernelInput.res = halfTargetResolution;
		materialKernelInput.pathStates = mImpl->pathStates.data();
		materialKernelInput.randStates = mImpl->randStates.data();
		materialKernelInput.rays = mImpl->rayBuffer.data();
		materialKernelInput.rayHitInfos = mImpl->rayHitInfoBuffer.data();
		materialKernelInput.shadowRays = mImpl->shadowRayBuffer.data();
		materialKernelInput.lightContributions = mImpl->lightContributions.data();
		materialKernelInput.staticSphereLights = mImpl->staticSphereLights.data();
		materialKernelInput.numStaticSphereLights = mImpl->staticSphereLights.capacity();
		launchMaterialKernel(materialKernelInput);

		// Determine if secondary hit is in shadow
		RayCastKernelInput secondaryShadowRayCastInput;
		secondaryShadowRayCastInput.bvhNodes = mImpl->staticBvhNodes.cudaTexture();
		secondaryShadowRayCastInput.triangleVerts = mImpl->staticTriangleVertices.cudaTexture();
		secondaryShadowRayCastInput.numRays = numSecondaryShadowRays;
		secondaryShadowRayCastInput.rays = mImpl->shadowRayBuffer.data();
		launchShadowRayCastKernel(secondaryShadowRayCastInput, mImpl->shadowRayResultBuffer.data(), mImpl->glDeviceProperties);

		// Use shading result if not in shadow
		ShadowLogicKernelInput secondaryShadowLogicKernelInput;
		secondaryShadowLogicKernelInput.surface = mImpl->cudaResultTex.cudaSurface();
		secondaryShadowLogicKernelInput.res = targetResolution;
		secondaryShadowLogicKernelInput.resolutionScale = 2;
		secondaryShadowLogicKernelInput.addToSurface = true;
		secondaryShadowLogicKernelInput.shadowRayHits = mImpl->shadowRayResultBuffer.data();
		secondaryShadowLogicKernelInput.pathStates = mImpl->pathStates.data();
		secondaryShadowLogicKernelInput.lightContributions = mImpl->lightContributions.data();
		secondaryShadowLogicKernelInput.numStaticSphereLights = mImpl->staticSphereLights.capacity();
		launchShadowLogicKernel(secondaryShadowLogicKernelInput);
	}

	RenderResult result;
	result.renderedRes = mTargetResolution;
	result.depthTexture = gbuffer.depthTextureGL();
	result.colorTexture = mImpl->cudaResultTex.glTexture();
	result.materialTexture = mImpl->gbuffer.materialTextureGL();
	result.velocityTexture = mImpl->gbuffer.velocityTextureGL();
	return result;
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
	uint32_t numPrimaryShadowRays = numTargetPixels * mImpl->staticSphereLights.capacity();
	uint32_t numSecondaryRays = numTargetPixels / 4;
	//uint32_t numSecondaryShadowRays = numSecondaryRays * mImpl->staticSphereLights.capacity();
	mImpl->rayBuffer.destroy();
	mImpl->rayBuffer.create(numTargetPixels);
	mImpl->shadowRayBuffer.destroy();
	mImpl->shadowRayBuffer.create(numPrimaryShadowRays);
	mImpl->rayResultBuffer.destroy();
	mImpl->rayResultBuffer.create(numTargetPixels);
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

	launchInitCurandKernel(mTargetResolution, mImpl->randStates.data());


	// Petor shading memory allocation
	const uint32_t numFullResPixels = mTargetResolution.x * mTargetResolution.y;
	const uint32_t numHalfResPixels = numFullResPixels / 4;
	const uint32_t totalNumLights = mImpl->staticSphereLights.capacity();


	mImpl->psSecondaryRayBuffer.destroy();
	mImpl->psSecondaryRayBuffer.create(numHalfResPixels);
	mImpl->psSecondaryRayHitBuffer.destroy();
	mImpl->psSecondaryRayHitBuffer.create(numHalfResPixels);
	mImpl->psSecondaryRayHitInfoBuffer.destroy();
	mImpl->psSecondaryRayHitInfoBuffer.create(numHalfResPixels);

	mImpl->psShadowRayBuffer.destroy();
	mImpl->psShadowRayBuffer.create(numFullResPixels * totalNumLights);
	mImpl->psShadowRayInLightBuffer.destroy();
	mImpl->psShadowRayInLightBuffer.create(numFullResPixels * totalNumLights);

	mImpl->psSecondaryShadowRayBuffer.destroy();
	mImpl->psSecondaryShadowRayBuffer.create(numHalfResPixels * totalNumLights);
	mImpl->psSecondaryShadowRayInLightBuffer.destroy();
	mImpl->psSecondaryShadowRayInLightBuffer.create(numHalfResPixels * totalNumLights);

	mImpl->psLightStreamBuffer.destroy();
	mImpl->psLightStreamBuffer.create(numFullResPixels * (totalNumLights + 1u));
}

} // namespace phe
