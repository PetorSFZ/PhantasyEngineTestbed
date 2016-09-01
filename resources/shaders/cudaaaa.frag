#version 450

// Input, output and uniforms
// ------------------------------------------------------------------------------------------------

// Input
in vec2 uvCoord;
in vec3 nonNormRayDir;

// Output
out vec4 outFragColor;

// Uniforms
uniform sampler2D uCudaResultTex;

// Main
// ------------------------------------------------------------------------------------------------

void main()
{
	vec4 floooaats = texture(uCudaResultTex, uvCoord);
	outFragColor = clamp(floooaats, vec4(0.0), vec4(1.0));
}
