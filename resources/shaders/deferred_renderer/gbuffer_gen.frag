#version 450

// Material struct
// ------------------------------------------------------------------------------------------------

struct Material {
	vec4 albedoValue;
	int albedoIndex;

	float roughnessValue;
	int roughnessIndex;

	float metallicValue;
	int metallicIndex;
};

struct ParsedMaterial {
	vec4 albedo;
	float roughness;
	float metallic;
};

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

// Uniforms
uniform int uHasAlbedoTexture = 0;
uniform sampler2D uAlbedoTexture;
uniform vec4 uAlbedoValue = vec4(vec3(0.0), 1.0);

uniform int uHasRoughnessTexture = 0;
uniform sampler2D uRoughnessTexture;
uniform float uRoughnessValue = 0.0;

uniform int uHasMetallicTexture = 0;
uniform sampler2D uMetallicTexture;
uniform float uMetallicValue = 0.0;

uniform Material uMaterials[128];

// Main
// ------------------------------------------------------------------------------------------------

void main()
{
	Material mat = uMaterials[materialId];

	// TODO: Normal mapping
	outFragNormal = vec4(normalize(normal), 1.0);

	// Albedo
	vec4 albedo = mat.albedoValue;
	/*if (mat.albedoIndex != -1) {

		// albedo = sample

		if (albedo.a < 0.1) {
			discard;
			return;
		}
	}*/
	outFragAlbedo = vec4(albedo.rgb, 1.0);

	/*// Materials
	float roughness = mat.roughnessValue;
	if (mat.roughnessIndex != -1) {
		// roughness = sample
	}
	float metallic = mat.metallicValue;
	if (mat.metallicValue != -1) {
		// metallic = sample
	}
	outFragMaterial = vec4(roughness, metallic, 0.0, 1.0);*/
}
