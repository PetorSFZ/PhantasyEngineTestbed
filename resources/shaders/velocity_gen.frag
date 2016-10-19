#version 450

// Input, output and uniforms
// ------------------------------------------------------------------------------------------------

// Input
in vec2 uvCoord;

// Output
out vec4 outFragColor;

// Uniforms
uniform sampler2D uCurrDepthTexture;
uniform sampler2D uPrevDepthTexture;
uniform sampler2D uWorldVelocityTexture;

uniform mat4 uCurrInvViewMatrix;
uniform mat4 uCurrInvProjMatrix;
uniform mat4 uPrevViewMatrix;
uniform mat4 uPrevProjMatrix;

uniform vec2 uJitter;
uniform float uDeltaTime;

// Transform functions
// ------------------------------------------------------------------------------------------------

vec3 getVSPosition(float depth, vec2 coord, mat4 invProjMatrix)
{
	coord.y = 1.0 - coord.y; // Need to convert coord from GL to D3D clip space
	vec4 clipSpacePos = vec4(2.0 * coord - 1.0, depth, 1.0);
	vec4 posTmp = invProjMatrix * clipSpacePos;
	posTmp.xyz /= posTmp.w;
	return posTmp.xyz;
}

vec2 getUVCoord(mat4 projMatrix, vec3 vsPos)
{
	vec4 projPos = projMatrix * vec4(vsPos, 1.0);
	projPos.xyz /= projPos.w;
	vec2 clipSpacePos = vec2((projPos.xy + 1.0) / 2);
	clipSpacePos.y = 1.0 - clipSpacePos.y; // Need to convert coord from GL to D3D clip space
	return clipSpacePos;
}

// Main
// ------------------------------------------------------------------------------------------------

void main()
{
	float depth = texture(uCurrDepthTexture, uvCoord).r;

	// Get world position
	vec3 vsPos = getVSPosition(depth, uvCoord, uCurrInvProjMatrix);
	vec3 worldPos = (uCurrInvViewMatrix * vec4(vsPos, 1.0)).xyz;

	// Compensate for dynamic objects' world velocity
	vec3 worldVelocity = texture(uWorldVelocityTexture, uvCoord).rgb;
	vec3 frameVelocity = uDeltaTime * worldVelocity;
	vec4 prevWorldPos = vec4(worldPos - frameVelocity, 1.0);

	// Find texture coordinate from last frame
	vec3 vsPosInPrev = (uPrevViewMatrix * prevWorldPos).xyz;
	vec2 uvInPrev = getUVCoord(uPrevProjMatrix, vsPosInPrev);

	// Include difference in depth since last frame
	float prevDepth = texture(uPrevDepthTexture, uvCoord).r;
	vec3 uvDepthDiff = vec3(uvCoord - uvInPrev, depth - prevDepth);

	outFragColor = vec4(uvDepthDiff, 1.0);
}
