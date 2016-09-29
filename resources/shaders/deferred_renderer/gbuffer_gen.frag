#version 450

// Input and output
// ------------------------------------------------------------------------------------------------

// Input
in vec3 normal;
in vec2 uv;
flat in uint materialId;

// Output
layout(location = 0) out vec4 outFragNormal;
//layout(location = 1) out vec4 outFragAlbedo;
//layout(location = 2) out vec4 outFragMaterial;
layout(location = 3) out uint outFragMaterialId;
layout(location = 4) out vec2 outFragUV;

// Main
// ------------------------------------------------------------------------------------------------

void main()
{
	// TODO: Normal mapping
	outFragNormal = vec4(normalize(normal), 1.0);
	outFragMaterialId = materialId;
	outFragUV = uv;
}
