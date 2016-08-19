#version 330

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
uniform vec3 uAlbedoValue = vec3(0.0);

uniform int uHasRoughnessTexture = 0;
uniform sampler2D uRoughnessTexture;
uniform float uRoughnessValue = 0.0;

uniform int uHasMetallicTexture = 0;
uniform sampler2D uMetallicTexture;
uniform float uMetallicValue = 0.0;

uniform int uHasSpecularTexture = 0;
uniform sampler2D uSpecularTexture;
uniform float uSpecularValue = 0.5; // Should be 0.5 for 99% of materials, according to UE4 docs

// Main
// ------------------------------------------------------------------------------------------------

void main()
{
	// TODO: Normal mapping
	outFragNormal = vec4(normalize(normal), 1.0);

	vec3 albedo = uAlbedoValue;
	if (uHasAlbedoTexture != 0) {
		vec4 albedoTmp = texture(uAlbedoTexture, uv);
		if (albedoTmp.a < 0.1) {
			discard;
			return;
		}
		albedo = albedoTmp.rgb;
	}
	outFragAlbedo = vec4(albedo, 1.0);

	float roughness = uRoughnessValue;
	if (uHasRoughnessTexture != 0) {
		roughness = texture(uRoughnessTexture, uv).r;
	}

	float metallic = uMetallicValue;
	if (uHasMetallicTexture != 0) {
		metallic = texture(uMetallicTexture, uv).r;
	}

	float specular = uSpecularValue;
	if (uHasSpecularTexture != 0) {
		specular = texture(uSpecularTexture, uv).r;
	}

	outFragMaterial = vec4(roughness, metallic, specular, 1.0);
}
