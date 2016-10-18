#version 450
#extension GL_ARB_bindless_texture : require

// Input, output and uniforms
// ------------------------------------------------------------------------------------------------

// Input
in vec3 posWS;
in vec3 normalWS;
in vec2 uv;
flat in ivec4 textureIndices;
flat in vec4 albedoValue;
flat in vec4 materialValue;

// Output
layout(location = 0) out vec4 outPosition;
layout(location = 1) out vec4 outNormal;
layout(location = 2) out vec4 outAlbedo;
layout(location = 3) out vec4 outMaterial;
layout(location = 4) out vec3 outWorldVelocity;

layout(std430, binding = 1) buffer TextureSSBO
{
	sampler2D textures[];
};

// Uniforms
uniform vec3 uWorldVelocity;

// Main
// ------------------------------------------------------------------------------------------------

void main()
{
	// Stores uv in w component of position and normal buffers
	outPosition = vec4(posWS, uv.x);
	outNormal = vec4(normalize(normalWS), uv.y);

	// Albedo
	int albedoIndex = textureIndices.x;
	vec4 albedo = albedoValue;
	if (albedoIndex >= 0) {
		albedo = texture(textures[albedoIndex], uv);

		if (albedo.a < 0.1) {
			discard;
			return;
		}
	}
	outAlbedo = vec4(albedo.rgb, 1.0);

	// Material
	int roughnessIndex = textureIndices.y;
	float roughness = materialValue.x;
	if (roughnessIndex >= 0) {
		roughness = texture(textures[roughnessIndex], uv).r;
	}

	int metallicIndex = textureIndices.z;
	float metallic = materialValue.y;
	if (metallicIndex >= 0) {
		metallic = texture(textures[metallicIndex], uv).r;
	}

	outMaterial = vec4(roughness, metallic, 0.0, 0.0);
	outWorldVelocity = uWorldVelocity;
}
