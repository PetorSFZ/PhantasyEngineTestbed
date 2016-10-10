#version 450

// Input, output and uniforms
// ------------------------------------------------------------------------------------------------

// Input
in vec3 inPosition;
in vec3 inNormal;
in vec2 inUV;
in uint inMaterialId;

// Output
out vec3 posWS;
out vec3 normalWS;
out vec2 uv;
flat out ivec4 textureIndices;
flat out vec4 albedoValue;
flat out vec4 materialValue;

// Uniforms
uniform mat4 uProjMatrix;
uniform mat4 uViewMatrix;
uniform mat4 uModelMatrix;
uniform mat4 uNormalMatrix; // inverse(transpose(modelMatrix)) for non-uniform scaling

// Shader Storage Buffer Objects
struct CompactMaterial {
	ivec4 textureIndices; // [albedoTexIndex, roughnessTexIndex, metallicTexIndex, padding]
	vec4 albedoValue;
	vec4 materialValue; // [roughnessValue, metallicValue, padding, padding]
};

layout(std430, binding = 0) buffer MaterialSSBO
{
	CompactMaterial materials[];
};

// Main
// ------------------------------------------------------------------------------------------------

void main()
{
	gl_Position = uProjMatrix * uViewMatrix * uModelMatrix * vec4(inPosition, 1.0);
	vec4 tmpPosWS = uModelMatrix * vec4(inPosition, 1.0);
	posWS = tmpPosWS.xyz / tmpPosWS.w;
	normalWS = (uNormalMatrix * vec4(inNormal, 0.0)).xyz;
	uv = inUV;

	CompactMaterial mat = materials[inMaterialId];
	textureIndices = mat.textureIndices;
	albedoValue = mat.albedoValue;
	materialValue = mat.materialValue;
}
