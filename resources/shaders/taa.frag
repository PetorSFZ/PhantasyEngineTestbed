#version 450

// Input, output and uniforms
// ------------------------------------------------------------------------------------------------

// Input
in vec2 uvCoord;

// Output
layout(location = 0) out vec4 outColor;
layout(location = 1) out vec4 outHistory;

// Uniforms
uniform sampler2D uSrcTexture;
uniform sampler2D uHistoryTexture;
uniform sampler2D uVelocityTexture;
uniform sampler2D uPrevVelocityTexture;
uniform sampler2D uMaterialTexture;

uniform vec2 uResolution;

// Main
// ------------------------------------------------------------------------------------------------

void main()
{
	vec3 currentFrameColor = texture(uSrcTexture, uvCoord).rgb;
	float roughness = texture(uMaterialTexture, uvCoord).r;

	vec3 uvDepthVelocity = texture(uVelocityTexture, uvCoord).rgb;
	vec2 uvVelocity = uvDepthVelocity.rg;
	vec2 historyCoord = uvCoord - uvVelocity;

	// Check if history texture contains the reprojected position
	if (historyCoord.x < 0.0 || historyCoord.x > 1.0 ||
	    historyCoord.y < 0.0 || historyCoord.y > 1.0) {
		// Use current frame only
		outHistory = vec4(currentFrameColor, 1.0);
		outColor = vec4(vec3(outHistory.xyz), 1.0);
		return;
	}

	vec3 uvDepthHistoryVelocity = texture(uPrevVelocityTexture, historyCoord).rgb;
	vec2 historyVelocity = uvDepthHistoryVelocity.rg;

	vec2 pixelVelocity = uResolution * uvVelocity;
	vec2 historyPixelVelocity = uResolution * historyVelocity;

	// Weight history using velocity similar to method in SMAA paper, "SMAA: Enhanced Subpixel
	// Morphological Antialiasing" [Jimenez12].
	float velocityDistrust = min(1.0, 1.2 * sqrt(abs(length(pixelVelocity) - length(historyPixelVelocity))));

	// Trust less if the object has low roughness
	float roughnessDistrust = mix(3.0, 1.0, roughness);

	float trust = clamp(1.0 - roughnessDistrust * velocityDistrust, 0.0, 1.0);

	vec4 history = texture(uHistoryTexture, historyCoord).rgba;
	vec3 historyColor = history.rgb;

	// Alpha channel contains effective number of samples mixed in history texture.
	// Blend of current and history is done so if trust always is 1, all samples
	// in history and the current sample is weighted equally in result.
	float historySamples = history.a;
	float newHistorySamples = historySamples * trust + 1.0;

	float historyBlend = 1.0 - 1.0 / newHistorySamples;

	outHistory = vec4(mix(currentFrameColor, historyColor, historyBlend), newHistorySamples);
	outColor = vec4(outHistory.xyz, 1.0);
}
