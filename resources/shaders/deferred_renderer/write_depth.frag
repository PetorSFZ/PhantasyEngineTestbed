#version 450

// Input, output and uniforms
// ------------------------------------------------------------------------------------------------

// Input
in vec2 uvCoord;
in vec3 nonNormRayDir;

// Uniforms
uniform sampler2D uDepthTexture;

// Main
// ------------------------------------------------------------------------------------------------

void main()
{
	gl_FragDepth = texture(uDepthTexture, uvCoord).r;
}
