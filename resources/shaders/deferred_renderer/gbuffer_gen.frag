#version 450
#extension GL_ARB_bindless_texture : require

// Input, output and uniforms
// ------------------------------------------------------------------------------------------------

// Input
in vec3 normal;
in vec2 uv;
flat in uint materialId;

// Output
layout(location = 0) out vec4 outFragNormal;
layout(location = 1) out vec4 outFragAlbedo;
layout(location = 2) out vec4 outFragMaterial;

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

layout(std430, binding = 1) buffer TextureSSBO
{
	sampler2D textures[];
};

// Main
// ------------------------------------------------------------------------------------------------

void main()
{
	CompactMaterial mat = materials[materialId];

	// TODO: Normal mapping
	outFragNormal = vec4(normalize(normal), 1.0);

	// Albedo
	int albedoIndex = mat.textureIndices.x;
	vec4 albedo = mat.albedoValue;
	if (albedoIndex >= 0) {
		albedo = texture(textures[albedoIndex], uv);

		if (albedo.a < 0.1) {
			discard;
			return;
		}
	}
	outFragAlbedo = vec4(albedo.rgb, 1.0);

	// Material
	int roughnessIndex = mat.textureIndices.y;
	float roughness = mat.materialValue.x;
	if (roughnessIndex >= 0) {
		roughness = texture(textures[roughnessIndex], uv).r;
	}

	int metallicIndex = mat.textureIndices.z;
	float metallic = mat.materialValue.y;
	if (metallicIndex >= 0) {
		metallic = texture(textures[metallicIndex], uv).r;
	}

	outFragMaterial = vec4(roughness, metallic, 0.0, 0.0);
}
