#version 450

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

// Helper functions
// ------------------------------------------------------------------------------------------------

const vec3 gamma = vec3(2.2);

vec3 linearize(vec3 rgbGamma)
{
	return pow(rgbGamma, gamma);
}

vec4 linearize(vec4 rgbaGamma)
{
	return vec4(linearize(rgbaGamma.rgb), rgbaGamma.a);
}

// Main
// ------------------------------------------------------------------------------------------------

void main()
{
	// TODO: Normal mapping
	outFragNormal = vec4(normalize(normal), 1.0);

	// Albedo is defined in gamma space
	vec3 albedo = linearize(uAlbedoValue);
	if (uHasAlbedoTexture != 0) {
		vec4 albedoTmp = linearize(texture(uAlbedoTexture, uv));
		if (albedoTmp.a < 0.1) {
			discard;
			return;
		}
		albedo = albedoTmp.rgb;
	}
	outFragAlbedo = vec4(albedo, 1.0);

	// Roughness is defined in linear space
	float roughness = uRoughnessValue;
	if (uHasRoughnessTexture != 0) {
		roughness = texture(uRoughnessTexture, uv).r;
	}

	// Metallic is defined in linear space
	float metallic = uMetallicValue;
	if (uHasMetallicTexture != 0) {
		metallic = texture(uMetallicTexture, uv).r;
	}

	outFragMaterial = vec4(roughness, metallic, 0.0, 1.0);
}
