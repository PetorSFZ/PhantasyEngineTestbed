#version 450

// Input, output and uniforms
// ------------------------------------------------------------------------------------------------

// Input
in vec3 posWS;
in vec3 normalWS;
in vec2 uv;
flat in uint materialId;

// Output
layout(location = 0) out vec4 outPosition;
layout(location = 1) out vec4 outNormal;
layout(location = 2) out uint outMaterialId;

// Main
// ------------------------------------------------------------------------------------------------

void main()
{
	// Stores uv in w component of position and normal buffers
	outPosition = vec4(posWS, uv.x);
	outNormal = vec4(normalize(normalWS), uv.y);
	outMaterialId = materialId;
}
