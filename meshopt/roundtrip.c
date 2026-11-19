/* Roundtrip check: re-parse an optimized GLB with the bundled cgltf and
 * summarize its mesh/primitive/morph structure. Exits non-zero on any
 * parse or buffer-load failure, which would indicate the optimizer wrote
 * a structurally invalid document. */
#define CGLTF_IMPLEMENTATION
#include "../cgltf.h"

#include <stdio.h>

int main(int argc, char** argv)
{
	(void)argc;
	if (argc < 2)
	{
		fprintf(stderr, "usage: roundtrip <file.glb>\n");
		return 2;
	}

	cgltf_options options = {0};
	cgltf_data* data = NULL;
	cgltf_result result = cgltf_parse_file(&options, argv[1], &data);
	if (result != cgltf_result_success)
	{
		printf("PARSE_FAIL %d\n", (int)result);
		return 1;
	}
	result = cgltf_load_buffers(&options, data, argv[1]);
	if (result != cgltf_result_success)
	{
		printf("LOAD_FAIL %d\n", (int)result);
		cgltf_free(data);
		return 1;
	}

	for (cgltf_size i = 0; i < data->meshes_count; ++i)
	{
		cgltf_mesh* mesh = &data->meshes[i];
		for (cgltf_size j = 0; j < mesh->primitives_count; ++j)
		{
			cgltf_primitive* prim = &mesh->primitives[j];
			printf("mesh=%zu prim=%zu mode=%d indexed=%d attrs=%zu targets=%zu "
					"verts=%zu idx=%zu material=%d\n",
					i, j, (int)prim->type, prim->indices != NULL,
					prim->attributes_count, prim->targets_count,
					prim->attributes[0].data->count,
					prim->indices ? prim->indices->count : 0,
					prim->material != NULL);
		}
	}

	cgltf_free(data);
	printf("ROUNDTRIP_OK\n");
	return 0;
}
