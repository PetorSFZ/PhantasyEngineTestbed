#version 450

// Input, output and uniforms
// ------------------------------------------------------------------------------------------------

// Input
in vec3 inPosition;
in vec3 inNormal;
in vec2 inUV;

// Output
out vec3 normal;
out vec2 uv;

// Uniforms
uniform mat4 uProjMatrix;
uniform mat4 uViewMatrix;
uniform mat4 uModelMatrix;
uniform mat4 uNormalMatrix; // inverse(transpose(modelViewMatrix)) for non-uniform scaling

// Main
// ------------------------------------------------------------------------------------------------

void main()
{
	gl_Position = uProjMatrix * uViewMatrix * uModelMatrix * vec4(inPosition, 1.0);
	normal = (uNormalMatrix * vec4(inNormal, 0.0)).xyz;
	uv = inUV;
}
