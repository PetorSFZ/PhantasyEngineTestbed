#version 330

// Input, output and uniforms
// ------------------------------------------------------------------------------------------------

// Input
in vec2 uvCoord;
in vec3 nonNormRayDir;

// Output
out vec4 outFragColor;

// Uniforms
uniform mat4 uInvProjMatrix;
uniform sampler2D uDepthTexture;
uniform sampler2D uNormalTexture;
uniform sampler2D uAlbedoTexture;
uniform sampler2D uMaterialTexture;

// Functions
// ------------------------------------------------------------------------------------------------

vec3 getPosition(vec2 coord)
{
	float depth = texture(uDepthTexture, coord).r;
	vec4 clipSpacePos = vec4(2.0 * coord - 1.0, 2.0 * depth - 1.0, 1.0);
	vec4 posTmp = uInvProjMatrix * clipSpacePos;
	posTmp.xyz /= posTmp.w;
	return posTmp.xyz;
}

// Main
// ------------------------------------------------------------------------------------------------

void main()
{
	// Retrieve information from GBuffer
	vec3 pos = getPosition(uvCoord);
	vec3 normal = texture(uNormalTexture, uvCoord).rgb;
	vec3 albedo = texture(uAlbedoTexture, uvCoord).rgb;
	vec3 material = texture(uMaterialTexture, uvCoord).rgb;
	float roughness = material.r;
	float metallic = material.g;
	float specular = material.b; // Should be 0.5 for 99% of materials in UE4

	//outFragColor = vec4(pos, 1.0);
	outFragColor = vec4(normal, 1.0);
	//outFragColor = vec4(albedo, 1.0);
}
