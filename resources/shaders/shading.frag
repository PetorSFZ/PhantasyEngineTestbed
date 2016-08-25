#version 450

// Input, output and uniforms
// ------------------------------------------------------------------------------------------------

// Input
in vec2 uvCoord;
in vec3 nonNormRayDir;

// Output
out vec4 outFragColor;

// Uniforms
uniform mat4 uInvProjMatrix;
uniform sampler2D uDepthTexture;
uniform sampler2D uNormalTexture;
uniform sampler2D uAlbedoTexture;
uniform sampler2D uMaterialTexture;

uniform vec3 uLightPos;

// Constants
// ------------------------------------------------------------------------------------------------

const float PI = 3.14159265359;

// Functions
// ------------------------------------------------------------------------------------------------

vec3 getPosition(vec2 coord)
{
	float depth = texture(uDepthTexture, coord).r;
	vec4 clipSpacePos = vec4(2.0 * coord - 1.0, 2.0 * depth - 1.0, 1.0);
	vec4 posTmp = uInvProjMatrix * clipSpacePos;
	posTmp.xyz /= posTmp.w;
	return posTmp.xyz;
}

// PBR shading functions
// ------------------------------------------------------------------------------------------------

// References used:
// https://de45xmedrsdbp.cloudfront.net/Resources/files/2013SiggraphPresentationsNotes-26915738.pdf
// http://blog.selfshadow.com/publications/s2016-shading-course/
// http://www.codinglabs.net/article_physically_based_rendering_cook_torrance.aspx
// http://graphicrants.blogspot.se/2013/08/specular-brdf-reference.html

// Normal distribution function, GGX/Trowbridge-Reitz
// a = roughness^2, UE4 parameterization
// dot(n,h) term should be clamped to 0 if negative
float ggx(float nDotH, float a)
{
	float a2 = a * a;
	float div = PI * pow(nDotH * nDotH * (a2 - 1.0) + 1.0, 2);
	return a2 / div;
}

// Schlick's model adjusted to fit Smith's method
// k = a/2, where a = roughness^2, however, for analytical light sources (non image based)
// roughness is first remapped to roughness = (roughnessOrg + 1) / 2.
// Essentially, for analytical light sources:
// k = (roughness + 1)^2 / 8
// For image based lighting:
// k = roughness^2 / 2
float geometricSchlick(float nDotL, float nDotV, float k)
{
	float g1 = nDotL / (nDotL * (1.0 - k) + k);
	float g2 = nDotV / (nDotV * (1.0 - k) + k);
	return g1 * g2;
}

// Schlick's approximation. F0 should typically be 0.04 for dielectrics
vec3 fresnelSchlick(float nDotL, vec3 f0)
{
	return f0 + (vec3(1.0) - f0) * clamp(pow(1.0 - nDotL, 5), 0.0, 1.0);
}

// Main
// ------------------------------------------------------------------------------------------------

void main()
{
	// Retrieve position and normal from GBuffer
	vec3 p = getPosition(uvCoord);
	vec3 n = texture(uNormalTexture, uvCoord).rgb;

	// Shading parameters
	vec3 toLight = uLightPos - p;
	float toLightDist = length(toLight);
	vec3 l = toLight / toLightDist; // to light
	vec3 v = normalize(-p); // to view
	vec3 h = normalize(l + v); // half vector (normal of microfacet)

	// If nDotL is <= 0 then the light source is not in the hemisphere of the surface, i.e.
	// no shading needs to be performed
	float nDotL = dot(n, l);
	float nDotV = dot(n, v);
	if (nDotL <= 0.0 || nDotV <= 0.0) {
		outFragColor = vec4(0.0, 0.0, 0.0, 1.0);
		return;
	}

	// Retrieve material information from GBuffer
	vec3 albedo = texture(uAlbedoTexture, uvCoord).rgb;
	vec3 material = texture(uMaterialTexture, uvCoord).rgb;
	float roughness = material.r;
	float metallic = material.g;

	// Lambert diffuse
	vec3 diffuse = albedo / PI;

	// Cook-Torrance specular
	// Normal distribution function
	float nDotH = max(dot(n, h), 0.0); // max() should be superfluous here
	float ctD = ggx(nDotH, roughness * roughness);

	// Geometric self-shadowing term
	float k = pow(roughness + 1.0, 2) / 8.0;
	float ctG = geometricSchlick(nDotL, nDotV, k);

	// Fresnel function
	// Assume all dielectrics have a f0 of 0.04, for metals we assume f0 == albedo
	vec3 f0 = mix(vec3(0.04), albedo, metallic);
	vec3 ctF = fresnelSchlick(nDotL, f0);

	// Calculate final Cook-Torrance specular value
	vec3 specular = ctD * ctF * ctG / (4.0 * nDotL * nDotV);

	// Calculates light strength
	const float uLightRadius = 500.0;
	const float uLightStrength = 500.0;
	float fallofNumerator = pow(clamp(1.0 - pow(toLightDist / uLightRadius, 4), 0.0, 1.0), 2);
	float fallofDenominator = (toLightDist * toLightDist + 1.0);
	float falloff = fallofNumerator / fallofDenominator;
	float light = falloff * uLightStrength;

	// "Solves" reflectance equation under the assumption that the light source is a point light
	// and that there is no global illumination.
	vec3 res = (diffuse + specular) * light * nDotL;
	outFragColor = vec4(res, 1.0);
}
