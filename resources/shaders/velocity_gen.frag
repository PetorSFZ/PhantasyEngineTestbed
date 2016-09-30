#version 450

// Input, output and uniforms
// ------------------------------------------------------------------------------------------------

// Input
in vec2 uvCoord;

// Output
out vec4 outFragColor;

// Uniforms
uniform sampler2D uCurrDepthTexture;

uniform mat4 uCurrInvViewMatrix;
uniform mat4 uCurrInvProjMatrix;
uniform mat4 uPrevViewMatrix;
uniform mat4 uPrevProjMatrix;

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

vec3 getUVCoordAndDepth(mat4 projMatrix, vec3 vsPos)
{
	vec4 projPos = projMatrix * vec4(vsPos, 1.0);
	projPos.xyz /= projPos.w;
	vec4 clipSpacePos = vec4((projPos.xy + 1.0) / 2, projPos.z, 1.0);
	clipSpacePos.y = 1.0 - clipSpacePos.y; // Need to convert coord from D3D to GL clip space
	return clipSpacePos.xyz;
}

// Main
// ------------------------------------------------------------------------------------------------

void main()
{
	float depth = texture(uCurrDepthTexture, uvCoord).r;
	vec3 currVSPos = getVSPosition(depth, uvCoord, uCurrInvProjMatrix);
	vec4 worldPos = uCurrInvViewMatrix * vec4(currVSPos, 1.0);
	vec3 oldVSPos = (uPrevViewMatrix * worldPos).xyz;

	vec3 uvDepthInCurr = vec3(uvCoord, depth);
	vec3 uvDepthInPrev = getUVCoordAndDepth(uPrevProjMatrix, oldVSPos);
	vec2 uvInPrev = uvDepthInPrev.xy;
	vec3 uvDepthDiff = uvDepthInCurr - uvDepthInPrev;
	outFragColor = vec4(uvDepthDiff, 1.0);
}
