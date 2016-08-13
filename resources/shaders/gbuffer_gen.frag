#version 330

// Input, output and uniforms
// ------------------------------------------------------------------------------------------------

// Input
in vec3 normal;
in vec2 uv;

// Output
layout(location = 0) out vec4 outFragNormal;

// Main
// ------------------------------------------------------------------------------------------------

void main()
{
	outFragNormal = vec4(normalize(normal), 1.0);
}
