#version 450

// Input, output and uniforms
// ------------------------------------------------------------------------------------------------

// Input
in vec2 uvCoord;
in vec3 nonNormRayDir;

// Output
out vec4 outFragColor;

// Uniforms
uniform sampler2D uLinearTexture;
uniform float uGamma = 2.2;

// Main
// ------------------------------------------------------------------------------------------------

// Gamma 2.2 is default
vec4 applyGammaCorrection(vec4 rgba, float gamma)
{
	return vec4(pow(rgba.rgb, vec3(1.0 / gamma)), rgba.a);   
}

void main()
{
	vec4 linearValue = texture(uLinearTexture, uvCoord);
	outFragColor = applyGammaCorrection(linearValue, uGamma);
}
