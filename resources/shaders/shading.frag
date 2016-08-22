#version 330

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

/*vec3 fspec(vec3 l, vec3 v)
{

}

vec3 fdiff(vec3 l, vec3 v, float albedo)
{
	const float PI_INV = 1.0 / 3.14159265359;
	return albedo * PI_INV;
}

/// l = to light dir
/// v = to view dir
vec3 f(vec3 l, vec3 v, )
{
	return fspec(l, v) + fdiff(l, v);
}*/

//float F()

// Geometric attenuation sub-term
/*float G1(vec3 v, vec3 n, float k)
{
	float nDotV = dot(n, v);
	return nDotV / (nDotV * (1.0 - k) + k);
}

// Geometric shadowing (Schlick model)
// (Not appropriate for image-based lighting, only analytical light sources)
float G(vec3 l, vec3 v, vec3 h, vec3 n, float roughness)
{
	float k = pow(roughness + 1.0, 2) / 8.0;
	return G1(l, n, k) * G1(v, n, k);
}

// Normal distribution function, GGX/Trowbridge-Reitz
// a = roughness^2
float D(vec3 h, vec3 n, float a)
{
	float a2 = a * a;
	float nDotH = dot(n, h);
	float div = PI * pow(nDotH * nDotH * (a2 - 1.0) + 1.0, 2);
	return a2 / div;
}*/

// Schlick's approximation. F0 should typically be 0.04 for dielectrics
float FSchlick(float f0, vec3 l, vec3 n)
{
	// TODO: Need clamping
	return f0 + (1.0 - f0) * pow(1 - dot(l, n), 5);
}

float D()
{
	return 0.0;
}

float F()
{
	return 0.0;
}

float G()
{
	return 0.0;
}

// l = to light dir
// v = to view dir
// n = normal of surface
float cookTorrance(vec3 l, vec3 v, vec3 n, float roughness)
{
	vec3 h = normalize(l + v); 
	return D() * F() * G() / (4.0 * dot(n, l) * dot(n, v));
}

float lambertDiffuse()
{
	return 0.0;
}

// Main
// ------------------------------------------------------------------------------------------------

void main()
{
	// Retrieve information from GBuffer
	vec3 pos = getPosition(uvCoord);
	vec3 normal = texture(uNormalTexture, uvCoord).rgb;
	vec3 albedo = texture(uAlbedoTexture, uvCoord).rgb;
	vec3 material = texture(uMaterialTexture, uvCoord).rgb;
	float roughness = material.r;
	float metallic = material.g;

	// Shading parameters
	vec3 l = normalize(uLightPos - pos); // to light
	vec3 v = normalize(-pos); // to view
	vec3 h = normalize(l + v); // half vector (normal of microfacet)
	vec3 n = normal;

	// If nDotL is <= 0 then the light source is not in the hemisphere of the surface, i.e.
	// no shading needs to be performed
	float nDotL = dot(n, l);
	if (nDotL <= 0.0) {
		outFragColor = vec4(0.0, 0.0, 0.0, 1.0);
		return;
	}

	// Lambert diffuse
	vec3 diffuse = albedo / PI;

	// TODO: Cook-Torrance specular
	vec3 specular = vec3(0.0);

	// "Solves" reflectance equation under the assumption that the light source is a point light
	// and that there is no global illumination.
	vec3 res = (diffuse + specular) * nDotL;
	outFragColor = vec4(res, 1.0);
}
