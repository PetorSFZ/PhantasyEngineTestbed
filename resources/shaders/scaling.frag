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
uniform vec2 uViewportRes;
uniform vec2 uDstRes;

// Main
// ------------------------------------------------------------------------------------------------

void main()
{
	vec2 srcRes = vec2(textureSize(uSrcTexture, 0));

	vec2 dstFragSize = 1.0 / uDstRes;
	vec2 offs = dstFragSize * 0.25;

	vec2 coord = uvCoord;
	//coord.y = 1.0 - coord.y;
	coord = coord * uViewportRes / srcRes;
	//coord.y = 1.0 - coord.y;

	vec3 sample1 = texture(uSrcTexture, coord + vec2(-offs.x, -offs.y)).rgb;
	vec3 sample2 = texture(uSrcTexture, coord + vec2(offs.x, -offs.y)).rgb;
	vec3 sample3 = texture(uSrcTexture, coord + vec2(-offs.x, offs.y)).rgb;
	vec3 sample4 = texture(uSrcTexture, coord + vec2(offs.x, offs.y)).rgb;

	outFragColor = vec4((sample1 + sample2 + sample3 + sample4) / 4.0, 1.0);
}
