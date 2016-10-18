#version 450

// Input, output and uniforms
// ------------------------------------------------------------------------------------------------

// Input
in vec3 posWS;

// Uniform
uniform vec3 uLightPosWS;
uniform float uLightRange;

// Main
// ------------------------------------------------------------------------------------------------

void main()
{
	// Calculate distance to light
	vec3 toLight = uLightPosWS - posWS;
	float toLightDist = length(toLight);

	// Scale distance into [0, 1] range by dividing by light range (far plane)
	float depth = toLightDist / uLightRange;
	gl_FragDepth = depth;
}