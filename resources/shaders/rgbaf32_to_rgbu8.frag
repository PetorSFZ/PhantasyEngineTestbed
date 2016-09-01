#version 450

// Input, output and uniforms
// ------------------------------------------------------------------------------------------------

// Input
in vec2 uvCoord;
in vec3 nonNormRayDir;

// Output
out vec4 outFragColor;

// Uniforms
uniform sampler2D uFloatTexture;

// Main
// ------------------------------------------------------------------------------------------------

void main()
{
	vec3 floats = texture(uFloatTexture, uvCoord).rgb;
	vec3 clampedFloats = clamp(floats, vec3(0.0), vec3(1.0));
	outFragColor = vec4(clampedFloats, 1.0);
}
