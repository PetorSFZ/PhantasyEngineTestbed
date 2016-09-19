#version 450

// Material struct
// ------------------------------------------------------------------------------------------------

struct Material {
	vec4 albedoValue;
	uint albedoIndex;

	float roughnessValue;
	uint roughnessIndex;

	float metallicValue;
	uint metallicIndex;
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

uniform Material uMaterials[256];

// Main
// ------------------------------------------------------------------------------------------------

void main()
{
	// TODO: Normal mapping
	outFragNormal = vec4(normalize(normal), 1.0);

	vec4 albedo = uAlbedoValue;
	if (uHasAlbedoTexture != 0) {
		albedo = texture(uAlbedoTexture, uv);
		if (albedo.a < 0.1) {
			discard;
			return;
		}
	}
	outFragAlbedo = vec4(albedo.rgb, 1.0);

	float roughness = uRoughnessValue;
	if (uHasRoughnessTexture != 0) {
		roughness = texture(uRoughnessTexture, uv).r;
	}

	float metallic = uMetallicValue;
	if (uHasMetallicTexture != 0) {
		metallic = texture(uMetallicTexture, uv).r;
	}

	outFragMaterial = vec4(roughness, metallic, 0.0, 1.0);
}
