#version 450

// Input, output and uniforms
// ------------------------------------------------------------------------------------------------

// Input
in vec2 uvCoord;
in vec3 nonNormRayDir;

// Output
out vec4 outFragColor;

// Uniforms
uniform sampler2D uSrcTexture;

// Main
// ------------------------------------------------------------------------------------------------

void main()
{
	vec3 value = texture(uSrcTexture, uvCoord).rgb;
	outFragColor = vec4(value, 1.0);
}
