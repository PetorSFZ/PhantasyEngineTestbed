#version 450

// Input, output and uniforms
// ------------------------------------------------------------------------------------------------

// Input
in vec2 uvCoord;

// Output
out vec4 outFragColor;

// Uniforms
uniform sampler2D uSrcTexture;
uniform float uAccumulationPasses;

// Main
// ------------------------------------------------------------------------------------------------

void main()
{
	vec3 value = texture(uSrcTexture, uvCoord).rgb / uAccumulationPasses;
	outFragColor = vec4(value, 1.0);
}
