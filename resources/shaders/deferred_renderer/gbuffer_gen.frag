#version 450
#extension GL_ARB_bindless_texture : require

// Input, output and uniforms
// ------------------------------------------------------------------------------------------------

// Input
in vec3 normal;
in vec2 uv;
flat in ivec4 textureIndices;
flat in vec4 albedoValue;
flat in vec4 materialValue;

// Output
layout(location = 0) out vec4 outFragNormal;
layout(location = 1) out vec4 outFragAlbedo;
layout(location = 2) out vec4 outFragMaterial;

layout(std430, binding = 1) buffer TextureSSBO
{
	sampler2D textures[];
};

// Main
// ------------------------------------------------------------------------------------------------

void main()
{
	// TODO: Normal mapping
	outFragNormal = vec4(normalize(normal), 1.0);

	// Albedo
	int albedoIndex = textureIndices.x;
	vec4 albedo = albedoValue;
	if (albedoIndex >= 0) {
		albedo = texture(textures[albedoIndex], uv);

		if (albedo.a < 0.1) {
			discard;
			return;
		}
	}
	outFragAlbedo = vec4(albedo.rgb, 1.0);

	// Material
	int roughnessIndex = textureIndices.y;
	float roughness = materialValue.x;
	if (roughnessIndex >= 0) {
		roughness = texture(textures[roughnessIndex], uv).r;
	}

	int metallicIndex = textureIndices.z;
	float metallic = materialValue.y;
	if (metallicIndex >= 0) {
		metallic = texture(textures[metallicIndex], uv).r;
	}

	outFragMaterial = vec4(roughness, metallic, 0.0, 0.0);
}
