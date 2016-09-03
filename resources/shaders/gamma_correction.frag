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

// Gamma correction
// ------------------------------------------------------------------------------------------------

vec4 applyGammaCorrection(vec4 linearValue, float gamma)
{
	return vec4(pow(linearValue.rgb, vec3(1.0 / gamma)), linearValue.a);   
}

// Uncharted 2 Tonemap
// ------------------------------------------------------------------------------------------------

// References
// http://www.slideshare.net/ozlael/hable-john-uncharted2-hdr-lighting  @page ~100
// http://filmicgames.com/archives/75

const float A = 0.15; // Shoulder strength (default: 0.22)
const float B = 0.50; // Linear strength (default: 0.30)
const float C = 0.10; // Linear angle (default: 0.10)
const float D = 0.20; // Toe strength (default: 0.20)
const float E = 0.02; // Toe numerator (default: 0.01)
const float F = 0.30; // Toe denominator (default: 0.30)
const float W = 11.2; // Linear white point value (default: 11.2)

vec3 u2Tonemap(vec3 x)
{
	return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - (E / F);
}

vec4 applyU2Tonemap(vec4 linearValue)
{
	float exposureBias = 2.0; // TODO: Expose as uniform???
	vec3 res = u2Tonemap(exposureBias * linearValue.rgb) / u2Tonemap(vec3(W));
	return vec4(res, linearValue.a);
}

// Haarm-Peter Duiker’s curve, optimized version by Jim Hejl and Richard Burgess-Dawson
// ------------------------------------------------------------------------------------------------

// Also applies gamma correction with gamma = 2.2
vec4 applyFilmicTonemapping(vec4 linearValue)
{
	vec3 x = max(vec3(0.0), linearValue.rgb - 0.004);
	return vec4((x * (6.2 * x + 0.5)) / (x * (6.2 * x + 1.7) + 0.06), linearValue.a);
}

// Main
// ------------------------------------------------------------------------------------------------

void main()
{
	vec4 linearValue = texture(uLinearTexture, uvCoord);
	//outFragColor = applyGammaCorrection(linearValue, uGamma);
	//outFragColor = applyGammaCorrection(applyU2Tonemap(linearValue), uGamma);
	outFragColor = applyFilmicTonemapping(linearValue);
	//aoutFragColor = linearValue;
}
