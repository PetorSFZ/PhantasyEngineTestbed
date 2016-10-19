#version 450

// Input, output and uniforms
// ------------------------------------------------------------------------------------------------

// Geometry shader layout
layout(triangles) in;
layout(triangle_strip, max_vertices = 18) out;

// Output (per emitvertex())
out vec3 posWS;

// Uniforms
uniform mat4 uViewProjMatrices[6];

// Main
// ------------------------------------------------------------------------------------------------

void main()
{
	for (int face = 0; face < 6; face++) {

		// Specify which face we should render to
		gl_Layer = face;

		// Transform each vertex and emit to fragment shader for given face of shadow cube map
		mat4 viewProjMatrix = uViewProjMatrices[face];
		for (int vertex = 0; vertex < 3; vertex++) {
			
			vec4 vertexPosWS = gl_in[vertex].gl_Position;
			gl_Position = viewProjMatrix * vertexPosWS;
			posWS = vertexPosWS.xyz / vertexPosWS.w;

			EmitVertex();
		}

		EndPrimitive();
	}
}
