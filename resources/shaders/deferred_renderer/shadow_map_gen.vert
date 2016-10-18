#version 450

// Input, output and uniforms
// ------------------------------------------------------------------------------------------------

// Input
in vec3 inPosition;

// Uniforms
uniform mat4 uModelMatrix;

// Main
// ------------------------------------------------------------------------------------------------

void main()
{
	// Transform input position to model space
	gl_Position = uModelMatrix * vec4(inPosition, 1.0);
}
