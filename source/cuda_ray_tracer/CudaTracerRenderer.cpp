// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#include "CudaTracerRenderer.hpp"

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <sfz/gl/IncludeOpenGL.hpp>
#include <cuda_gl_interop.h>

#include <sfz/containers/DynArray.hpp>
#include <sfz/containers/StackString.hpp>
#include <sfz/gl/Program.hpp>
#include <sfz/memory/New.hpp>
#include <sfz/util/IO.hpp>

#include <phantasy_engine/config/GlobalConfig.hpp>
#include <phantasy_engine/deferred_renderer/GLModel.hpp>
#include <phantasy_engine/rendering/FullscreenTriangle.hpp>

#include "CudaBindlessTexture.hpp"
#include "CudaBuffer.hpp"
#include "CudaGLInterop.hpp"
#include "CudaHelpers.hpp"
#include "CudaTextureBuffer.hpp"
#include "kernels/ProcessGBufferKernel.hpp"
#include "kernels/RayCastKernel.hpp"

namespace phe {

using namespace sfz;
using sfz::gl::Program;

// CudaTracerRendererImpl
// ------------------------------------------------------------------------------------------------

class CudaTracerRendererImpl final {
public:
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

	// Cuda materials & textures
	CudaBuffer<Material> materials;
	DynArray<CudaBindlessTexture> textureWrappers;
	CudaBuffer<cudaTextureObject_t> textures;

	// Cuda static geometry
	CudaTextureBuffer<BVHNode> staticBvhNodes;
	CudaTextureBuffer<TriangleVertices> staticTriangleVertices;
	CudaBuffer<TriangleData> staticTriangleDatas;

	// Cuda static light sources
	CudaBuffer<SphereLight> staticSphereLights;

	// Raycast input and output buffers
	const uint32_t NUM_RAYS_PER_PIXEL_PER_BATCH = 1u;
	CudaBuffer<RayIn> rayBuffer;
	CudaBuffer<RayHit> rayResultBuffer;

	CudaTracerRendererImpl() noexcept
	{
		// Initialize settings
		GlobalConfig& cfg = GlobalConfig::instance();
		// TODO: All eventual cuda settings should be initialized here, section = "CudaTracer"

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
	// Copy materials to Cuda
	mImpl->materials = CudaBuffer<Material>(materials.data(), materials.size());

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
	mImpl->textures = CudaBuffer<cudaTextureObject_t>(tmpTextureHandles.data(),
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

	// Temporary cuda kernel
	// --------------------------------------------------------------------------------------------

	CreateReflectRaysInput reflectRaysInput;
	reflectRaysInput.camPos = mCamera.pos();
	reflectRaysInput.res = mTargetResolution;
	reflectRaysInput.posTex = mImpl->gbuffer.positionSurfaceCuda();
	reflectRaysInput.normalTex = mImpl->gbuffer.normalSurfaceCuda();
	reflectRaysInput.materialIdTex = mImpl->gbuffer.materialIdSurfaceCuda();
	launchCreateReflectRaysKernel(reflectRaysInput, mImpl->rayBuffer.cudaPtr());
	
	//CameraDef camDef = generateCameraDef(mMatrices.position, mMatrices.forward, mMatrices.up,
	//                                     mMatrices.vertFovRad, vec2(mTargetResolution));
	//launchGenPrimaryRaysKernel(mImpl->rayBuffer.cudaPtr(), camDef, mTargetResolution);
	
	RayCastKernelInput rayCastInput;
	rayCastInput.bvhNodes = mImpl->staticBvhNodes.cudaTexture();
	rayCastInput.triangleVerts = mImpl->staticTriangleVertices.cudaTexture();
	rayCastInput.numRays = mTargetResolution.x * mTargetResolution.y;
	rayCastInput.rays = mImpl->rayBuffer.cudaPtr();
	launchRayCastKernel(rayCastInput, mImpl->rayResultBuffer.cudaPtr(), mImpl->glDeviceProperties);

	launchWriteRayHitsToScreenKernel(mImpl->cudaResultTex.cudaSurface(), mTargetResolution,
	                                 mImpl->rayResultBuffer.cudaPtr());

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
	uint32_t numRaysPerBatch = mTargetResolution.x * mTargetResolution.y
	                           * mImpl->NUM_RAYS_PER_PIXEL_PER_BATCH;
	mImpl->rayBuffer.destroy();
	mImpl->rayBuffer.create(numRaysPerBatch);
	mImpl->rayResultBuffer.destroy();
	mImpl->rayResultBuffer.create(numRaysPerBatch);
}

} // namespace phe
