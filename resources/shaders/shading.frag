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

// Input, output and uniforms
// ------------------------------------------------------------------------------------------------

void main()
{
	outFragColor = vec4(texture(uNormalTexture, uvCoord).rgb, 1.0);
}
