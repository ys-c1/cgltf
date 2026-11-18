/*
 * gltf_resource_graph.c — external resource reference graph analyzer for cgltf.
 *
 * Built on top of cgltf parsing (../cgltf.h, repository-local, no third-party
 * dependencies). For a given .gltf/.glb file it builds a directed reference
 * graph of buffers / images / textures / materials (+ meshes/nodes as graph
 * members), detects:
 *
 *   - missing external files (buffers and images referenced by URI)
 *   - circular references (cycles in the object graph, including
 *     self-referencing external resources such as a buffer whose URI points
 *     back at the document being analyzed)
 *   - duplicate references (one resource referenced from more than one place)
 *
 * and emits a severity-ordered report (ERROR > WARN > INFO).
 *
 * Usage:
 *   gltf_resource_graph <model.gltf|model.glb> [--json]
 *   gltf_resource_graph --perf <model.gltf|model.glb> [iterations]
 *
 * Exit code is 0 even when issues are found (the report is the result);
 * a non-zero exit code means the analyzer itself failed (e.g. parse error).
 *
 * C99, no network access, no additional libraries.
 */

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 199309L /* clock_gettime */
#endif

#define CGLTF_IMPLEMENTATION
#include "../cgltf.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ------------------------------------------------------------------ */
/* Public-ish data structures                                         */
/* ------------------------------------------------------------------ */

typedef enum grg_severity
{
	GRG_SEVERITY_INFO = 0,
	GRG_SEVERITY_WARN = 1,
	GRG_SEVERITY_ERROR = 2
} grg_severity;

typedef enum grg_node_kind
{
	GRG_KIND_DOCUMENT = 0,
	GRG_KIND_BUFFER,
	GRG_KIND_IMAGE,
	GRG_KIND_TEXTURE,
	GRG_KIND_MATERIAL,
	GRG_KIND_MESH,
	GRG_KIND_NODE,
	GRG_KIND_FILE /* synthetic node for an external file referenced by URI */
} grg_node_kind;

typedef enum grg_edge_kind
{
	GRG_EDGE_DOC_BUFFER = 0,  /* document -> buffer            */
	GRG_EDGE_DOC_IMAGE,       /* document -> image             */
	GRG_EDGE_DOC_TEXTURE,     /* document -> texture           */
	GRG_EDGE_DOC_MATERIAL,    /* document -> material          */
	GRG_EDGE_DOC_MESH,        /* document -> mesh              */
	GRG_EDGE_DOC_NODE,        /* document -> node              */
	GRG_EDGE_IMAGE_BUFFER,    /* image -> buffer (bufferView)  */
	GRG_EDGE_TEX_IMAGE,       /* texture -> image              */
	GRG_EDGE_MAT_TEX,         /* material -> texture           */
	GRG_EDGE_MESH_MAT,        /* mesh -> material              */
	GRG_EDGE_NODE_MESH,       /* node -> mesh                  */
	GRG_EDGE_NODE_NODE,       /* node -> node (hierarchy)      */
	GRG_EDGE_RESOURCE_FILE,   /* buffer/image -> external file */
	GRG_EDGE_FILE_DOCUMENT,   /* external file -> document (self-referencing URI) */
	GRG_EDGE_MAX_ENUM
} grg_edge_kind;

typedef struct grg_node
{
	grg_node_kind kind;
	cgltf_size index;       /* index inside the owning cgltf_data array */
	const char* name;       /* may be NULL                              */
	const char* uri;        /* FILE nodes: resolved absolute/relative path;
	                           buffer/image nodes: raw URI when present  */
	cgltf_size in_degree;
	cgltf_size out_degree;
} grg_node;

typedef struct grg_edge
{
	cgltf_size src;
	cgltf_size dst;
	grg_edge_kind kind;
} grg_edge;

typedef struct grg_finding
{
	grg_severity severity;
	cgltf_size node_id;      /* primary node, or (cgltf_size)-1 */
	char message[512];
} grg_finding;

typedef struct grg_graph
{
	grg_node* nodes;
	cgltf_size nodes_count;
	cgltf_size nodes_cap;

	grg_edge* edges;
	cgltf_size edges_count;
	cgltf_size edges_cap;

	/* adjacency: per node, contiguous range in adj (sorted by node id) */
	cgltf_size* adj_head;   /* size nodes_count + 1 */
	cgltf_size* adj;        /* size edges_count, entries are edge indices */

	grg_finding* findings;
	cgltf_size findings_count;
	cgltf_size findings_cap;

	cgltf_size count_by_severity[3];
	cgltf_size count_missing;
	cgltf_size count_cycles;
	cgltf_size count_duplicates;

	/* per-node incoming reference count over semantic edges (no FILE edges) */
	cgltf_size* refcount;
} grg_graph;

/* ------------------------------------------------------------------ */
/* Small helpers                                                      */
/* ------------------------------------------------------------------ */

static const char* grg_kind_name(grg_node_kind kind)
{
	switch (kind)
	{
	case GRG_KIND_DOCUMENT: return "document";
	case GRG_KIND_BUFFER:   return "buffer";
	case GRG_KIND_IMAGE:    return "image";
	case GRG_KIND_TEXTURE:  return "texture";
	case GRG_KIND_MATERIAL: return "material";
	case GRG_KIND_MESH:     return "mesh";
	case GRG_KIND_NODE:     return "node";
	case GRG_KIND_FILE:     return "file";
	}
	return "unknown";
}

static const char* grg_severity_name(grg_severity s)
{
	return s == GRG_SEVERITY_ERROR ? "ERROR" : (s == GRG_SEVERITY_WARN ? "WARN" : "INFO");
}

static const char* grg_edge_name(grg_edge_kind k)
{
	switch (k)
	{
	case GRG_EDGE_DOC_BUFFER:    return "declares_buffer";
	case GRG_EDGE_DOC_IMAGE:     return "declares_image";
	case GRG_EDGE_DOC_TEXTURE:   return "declares_texture";
	case GRG_EDGE_DOC_MATERIAL:  return "declares_material";
	case GRG_EDGE_DOC_MESH:      return "declares_mesh";
	case GRG_EDGE_DOC_NODE:      return "declares_node";
	case GRG_EDGE_IMAGE_BUFFER:  return "uses_buffer";
	case GRG_EDGE_TEX_IMAGE:     return "uses_image";
	case GRG_EDGE_MAT_TEX:       return "uses_texture";
	case GRG_EDGE_MESH_MAT:      return "uses_material";
	case GRG_EDGE_NODE_MESH:     return "uses_mesh";
	case GRG_EDGE_NODE_NODE:     return "parent_of";
	case GRG_EDGE_RESOURCE_FILE: return "loads_file";
	case GRG_EDGE_FILE_DOCUMENT: return "points_back_to";
	case GRG_EDGE_MAX_ENUM:      break;
	}
	return "unknown";
}

static int grg_is_data_uri(const char* uri)
{
	return uri && strncmp(uri, "data:", 5) == 0;
}

static int grg_is_remote_uri(const char* uri)
{
	return uri && strstr(uri, "://") != NULL;
}

/* dirname(path) + "/" + uri. Unlike cgltf's cgltf_combine_paths this always
 * treats the document path as a file and resolves the URI against its
 * containing directory, so "a.gltf" + "b.bin" never collapses into "a.gltfb.bin".
 * The URI tail is percent-decoded in place, matching cgltf's file loader. */
static void grg_resolve_path(char* out, cgltf_size out_size, const char* base_path, const char* uri)
{
	const char* s0 = strrchr(base_path, '/');
	const char* s1 = strrchr(base_path, '\\');
	const char* slash = s0 ? (s1 && s1 > s0 ? s1 : s0) : s1;
	cgltf_size prefix = slash ? (cgltf_size)(slash - base_path) + 1 : 0;

	if (prefix + strlen(uri) + 1 > out_size)
	{
		/* cannot happen with the bounded URIs we deal with */
		out[0] = 0;
		return;
	}

	if (prefix)
	{
		memcpy(out, base_path, prefix);
		out[prefix] = 0;
	}
	else
	{
		out[0] = 0;
	}
	strcat(out, uri);

	/* percent-decode the URI tail the same way cgltf does */
	cgltf_decode_uri(out + prefix);
}

static int grg_file_exists(const char* path)
{
	FILE* f = fopen(path, "rb");
	if (!f)
	{
		return 0;
	}
	fclose(f);
	return 1;
}

/* ------------------------------------------------------------------ */
/* Graph construction                                                 */
/* ------------------------------------------------------------------ */

static void grg_graph_init(grg_graph* g)
{
	memset(g, 0, sizeof(*g));
}

static void grg_graph_free(grg_graph* g)
{
	/* only FILE nodes own their name/uri strings (heap copies) */
	for (cgltf_size i = 0; i < g->nodes_count; ++i)
	{
		if (g->nodes[i].kind == GRG_KIND_FILE)
		{
			free((void*)g->nodes[i].uri);
		}
	}
	free(g->nodes);
	free(g->edges);
	free(g->adj_head);
	free(g->adj);
	free(g->findings);
	free(g->refcount);
	memset(g, 0, sizeof(*g));
}

static int grg_add_node(grg_graph* g, grg_node_kind kind, cgltf_size index,
		const char* name, const char* uri, cgltf_size* out_id)
{
	if (g->nodes_count == g->nodes_cap)
	{
		cgltf_size new_cap = g->nodes_cap ? g->nodes_cap * 2 : 64;
		grg_node* p = (grg_node*)realloc(g->nodes, new_cap * sizeof(grg_node));
		if (!p) return -1;
		g->nodes = p;
		g->nodes_cap = new_cap;
	}
	grg_node* n = &g->nodes[g->nodes_count];
	n->kind = kind;
	n->index = index;
	n->name = name;
	n->uri = uri;
	n->in_degree = 0;
	n->out_degree = 0;
	*out_id = g->nodes_count++;
	return 0;
}

static int grg_add_finding(grg_graph* g, grg_severity sev, cgltf_size node_id, const char* fmt, ...)
{
	if (g->findings_count == g->findings_cap)
	{
		cgltf_size new_cap = g->findings_cap ? g->findings_cap * 2 : 16;
		grg_finding* p = (grg_finding*)realloc(g->findings, new_cap * sizeof(grg_finding));
		if (!p) return -1;
		g->findings = p;
		g->findings_cap = new_cap;
	}
	grg_finding* f = &g->findings[g->findings_count++];
	f->severity = sev;
	f->node_id = node_id;
	va_list ap;
	va_start(ap, fmt);
	vsnprintf(f->message, sizeof(f->message), fmt, ap);
	va_end(ap);
	g->count_by_severity[sev]++;
	return 0;
}

/* Document membership edges give every object one "structural" incoming
 * edge; they must not count as semantic references or every object would
 * look duplicated. FILE edges describe filesystem dependencies and are
 * excluded from duplicate accounting as well. */
static int grg_is_membership_edge(grg_edge_kind k)
{
	return k == GRG_EDGE_DOC_BUFFER || k == GRG_EDGE_DOC_IMAGE ||
			k == GRG_EDGE_DOC_TEXTURE || k == GRG_EDGE_DOC_MATERIAL ||
			k == GRG_EDGE_DOC_MESH || k == GRG_EDGE_DOC_NODE;
}

static int grg_add_edge(grg_graph* g, cgltf_size src, cgltf_size dst, grg_edge_kind kind,
		int* dup_flag)
{
	if (g->edges_count == g->edges_cap)
	{
		cgltf_size new_cap = g->edges_cap ? g->edges_cap * 2 : 128;
		grg_edge* p = (grg_edge*)realloc(g->edges, new_cap * sizeof(grg_edge));
		if (!p) return -1;
		g->edges = p;
		g->edges_cap = new_cap;
	}
	grg_edge* e = &g->edges[g->edges_count++];
	e->src = src;
	e->dst = dst;
	e->kind = kind;
	g->nodes[src].out_degree++;
	g->nodes[dst].in_degree++;
	if (!grg_is_membership_edge(kind) && kind != GRG_EDGE_RESOURCE_FILE)
	{
		g->refcount[dst]++;
		if (g->refcount[dst] == 2)
		{
			*dup_flag = 1; /* crossed the duplicate threshold */
		}
	}
	return 0;
}

/* Wrapper that also reports a duplicate finding when the target crosses
 * two incoming semantic references. */
static int grg_link(grg_graph* g, cgltf_size src, cgltf_size dst, grg_edge_kind kind)
{
	int dup = 0;
	if (grg_add_edge(g, src, dst, kind, &dup) != 0) return -1;
	if (dup)
	{
		g->count_duplicates++;
		grg_node* t = &g->nodes[dst];
		if (t->name)
			return grg_add_finding(g, GRG_SEVERITY_WARN, dst,
					"duplicate reference: %s \"%s\" is referenced by %u or more graph members",
					grg_kind_name(t->kind), t->name, (unsigned)g->refcount[dst]);
		return grg_add_finding(g, GRG_SEVERITY_WARN, dst,
				"duplicate reference: %s[%u] is referenced by %u or more graph members",
				grg_kind_name(t->kind), (unsigned)t->index, (unsigned)g->refcount[dst]);
	}
	return 0;
}

/* IDs of the primary cgltf objects, allocated up front */
typedef struct grg_id_map
{
	cgltf_size document;
	cgltf_size* buffers;
	cgltf_size* images;
	cgltf_size* textures;
	cgltf_size* materials;
	cgltf_size* meshes;
	cgltf_size* nodes;
} grg_id_map;

/* Find-or-create a FILE node for an external resolved path. The path lives
 * in a caller stack buffer and is overwritten on every call, so it must be
 * duplicated; the copy is freed with the graph. */
static cgltf_size grg_file_node(grg_graph* g, const char* path)
{
	for (cgltf_size i = 0; i < g->nodes_count; ++i)
	{
		if (g->nodes[i].kind == GRG_KIND_FILE && g->nodes[i].uri &&
				strcmp(g->nodes[i].uri, path) == 0)
		{
			return i;
		}
	}
	char* path_copy = (char*)malloc(strlen(path) + 1);
	if (!path_copy) return (cgltf_size)-1;
	strcpy(path_copy, path);
	cgltf_size id;
	if (grg_add_node(g, GRG_KIND_FILE, 0, path_copy, path_copy, &id) != 0)
	{
		free(path_copy);
		return (cgltf_size)-1;
	}
	return id;
}

/* Resolve a buffer/image URI to a FILE node, check existence, and detect
 * self-referencing URIs (an external resource pointing back at the document
 * itself, which closes a filesystem-level cycle). */
static int grg_attach_external_file(grg_graph* g, cgltf_size resource_node,
		const char* uri, const char* doc_path, cgltf_size doc_node,
		grg_severity missing_severity)
{
	if (grg_is_data_uri(uri))
	{
		return grg_add_finding(g, GRG_SEVERITY_INFO, resource_node,
				"embedded data URI (no external file required)");
	}
	if (grg_is_remote_uri(uri))
	{
		/* network access is out of scope; record without stat()ing */
		return grg_add_finding(g, GRG_SEVERITY_INFO, resource_node,
				"remote URI not checked (network access disabled): %s", uri);
	}

	char resolved[1024];
	grg_resolve_path(resolved, sizeof(resolved), doc_path, uri);

	cgltf_size file_id = grg_file_node(g, resolved);
	if (file_id == (cgltf_size)-1) return -1;

	int dup = 0;
	if (grg_add_edge(g, resource_node, file_id, GRG_EDGE_RESOURCE_FILE, &dup) != 0)
	{
		return -1;
	}

	if (doc_path && strcmp(resolved, doc_path) == 0)
	{
		/* file -> document closes a cycle that the DFS pass will report;
		 * multiple resources may share the same FILE node, so the cycle
		 * count is owned by the DFS back-edge detection, not incremented
		 * here. */
		if (g->nodes[file_id].out_degree == 0)
		{
			int ignored = 0;
			grg_add_edge(g, file_id, doc_node, GRG_EDGE_FILE_DOCUMENT, &ignored);
		}
		grg_add_finding(g, GRG_SEVERITY_ERROR, resource_node,
				"self-referencing external URI points back to the document: %s", uri);
	}

	if (!grg_file_exists(resolved))
	{
		g->count_missing++;
		return grg_add_finding(g, missing_severity, resource_node,
				"missing external file: %s (uri \"%s\")", resolved, uri);
	}
	return 0;
}

/* Build the adjacency arrays after all edges have been collected. */
static int grg_build_adjacency(grg_graph* g)
{
	g->adj_head = (cgltf_size*)calloc(g->nodes_count + 1, sizeof(cgltf_size));
	g->adj = (cgltf_size*)malloc((g->edges_count ? g->edges_count : 1) * sizeof(cgltf_size));
	if (!g->adj_head || !g->adj) return -1;

	for (cgltf_size i = 0; i < g->edges_count; ++i)
	{
		g->adj_head[g->edges[i].src + 1]++;
	}
	for (cgltf_size i = 1; i <= g->nodes_count; ++i)
	{
		g->adj_head[i] += g->adj_head[i - 1];
	}
	cgltf_size* cursor = (cgltf_size*)calloc(g->nodes_count, sizeof(cgltf_size));
	if (!cursor) return -1;
	for (cgltf_size i = 0; i < g->edges_count; ++i)
	{
		cgltf_size s = g->edges[i].src;
		g->adj[g->adj_head[s] + cursor[s]++] = i;
	}
	free(cursor);
	return 0;
}

/* ------------------------------------------------------------------ */
/* Cycle detection: iterative DFS with three colors                   */
/* ------------------------------------------------------------------ */

static int grg_detect_cycles(grg_graph* g)
{
	unsigned char* color = (unsigned char*)calloc(g->nodes_count, 1); /* 0 white */
	cgltf_size* stack = (cgltf_size*)malloc((g->nodes_count + 1) * sizeof(cgltf_size));
	cgltf_size* edge_pos = (cgltf_size*)malloc(g->nodes_count * sizeof(cgltf_size));
	if (!color || !stack || !edge_pos)
	{
		free(color); free(stack); free(edge_pos);
		return -1;
	}

	for (cgltf_size root = 0; root < g->nodes_count; ++root)
	{
		if (color[root] != 0) continue;

		cgltf_ssize top = 0;
		stack[top] = root;
		edge_pos[top] = g->adj_head[root];
		color[root] = 1; /* gray */

		while (top >= 0)
		{
			cgltf_size u = stack[top];
			if (edge_pos[top] < g->adj_head[u + 1])
			{
				cgltf_size ei = g->adj[edge_pos[top]++];
				cgltf_size v = g->edges[ei].dst;
				if (color[v] == 0)
				{
					color[v] = 1;
					top++;
					stack[top] = v;
					edge_pos[top] = g->adj_head[v];
				}
				else if (color[v] == 1)
				{
					/* back edge u -> v: walk the stack to name the cycle */
					char chain[768];
					chain[0] = 0;
					cgltf_size start = (cgltf_size)top;
					for (cgltf_ssize k = 0; k <= top; ++k)
					{
						if (stack[k] == v) { start = (cgltf_size)k; break; }
					}
					for (cgltf_size k = start; k <= (cgltf_size)top; ++k)
					{
						grg_node* nd = &g->nodes[stack[k]];
						char piece[160];
						if (nd->name)
							snprintf(piece, sizeof(piece), "%s(%s) -> ",
									grg_kind_name(nd->kind), nd->name);
						else
							snprintf(piece, sizeof(piece), "%s[%u] -> ",
									grg_kind_name(nd->kind), (unsigned)nd->index);
						strncat(chain, piece, sizeof(chain) - strlen(chain) - 1);
					}
					strncat(chain, grg_kind_name(g->nodes[v].kind),
							sizeof(chain) - strlen(chain) - 1);
					g->count_cycles++;
					grg_add_finding(g, GRG_SEVERITY_ERROR, u,
							"circular reference detected: %s", chain);
				}
			}
			else
			{
				color[u] = 2; /* black */
				top--;
			}
		}
	}

	free(color);
	free(stack);
	free(edge_pos);
	return 0;
}

/* ------------------------------------------------------------------ */
/* Main graph build over a parsed cgltf_data                          */
/* ------------------------------------------------------------------ */

static int grg_build(cgltf_data* data, const char* doc_path, grg_graph* g)
{
	grg_graph_init(g);

	cgltf_size total = 1 + data->buffers_count + data->images_count +
			data->textures_count + data->materials_count +
			data->meshes_count + data->nodes_count;
	g->refcount = (cgltf_size*)calloc(total + 1, sizeof(cgltf_size));
	if (!g->refcount) return -1;

	grg_id_map m;
	memset(&m, 0, sizeof(m));
	m.buffers = (cgltf_size*)calloc(data->buffers_count ? data->buffers_count : 1, sizeof(cgltf_size));
	m.images = (cgltf_size*)calloc(data->images_count ? data->images_count : 1, sizeof(cgltf_size));
	m.textures = (cgltf_size*)calloc(data->textures_count ? data->textures_count : 1, sizeof(cgltf_size));
	m.materials = (cgltf_size*)calloc(data->materials_count ? data->materials_count : 1, sizeof(cgltf_size));
	m.meshes = (cgltf_size*)calloc(data->meshes_count ? data->meshes_count : 1, sizeof(cgltf_size));
	m.nodes = (cgltf_size*)calloc(data->nodes_count ? data->nodes_count : 1, sizeof(cgltf_size));
	if (!m.buffers || !m.images || !m.textures || !m.materials || !m.meshes || !m.nodes)
	{
		free(m.buffers); free(m.images); free(m.textures);
		free(m.materials); free(m.meshes); free(m.nodes);
		return -1;
	}

	/* 1. one node per object */
	grg_add_node(g, GRG_KIND_DOCUMENT, 0, doc_path, NULL, &m.document);

	for (cgltf_size i = 0; i < data->buffers_count; ++i)
		grg_add_node(g, GRG_KIND_BUFFER, i, data->buffers[i].name, data->buffers[i].uri, &m.buffers[i]);
	for (cgltf_size i = 0; i < data->images_count; ++i)
		grg_add_node(g, GRG_KIND_IMAGE, i, data->images[i].name, data->images[i].uri, &m.images[i]);
	for (cgltf_size i = 0; i < data->textures_count; ++i)
		grg_add_node(g, GRG_KIND_TEXTURE, i, data->textures[i].name, NULL, &m.textures[i]);
	for (cgltf_size i = 0; i < data->materials_count; ++i)
		grg_add_node(g, GRG_KIND_MATERIAL, i, data->materials[i].name, NULL, &m.materials[i]);
	for (cgltf_size i = 0; i < data->meshes_count; ++i)
		grg_add_node(g, GRG_KIND_MESH, i, data->meshes[i].name, NULL, &m.meshes[i]);
	for (cgltf_size i = 0; i < data->nodes_count; ++i)
		grg_add_node(g, GRG_KIND_NODE, i, data->nodes[i].name, NULL, &m.nodes[i]);

	/* 2. document membership edges */
	for (cgltf_size i = 0; i < data->buffers_count; ++i)
		grg_link(g, m.document, m.buffers[i], GRG_EDGE_DOC_BUFFER);
	for (cgltf_size i = 0; i < data->images_count; ++i)
		grg_link(g, m.document, m.images[i], GRG_EDGE_DOC_IMAGE);
	for (cgltf_size i = 0; i < data->textures_count; ++i)
		grg_link(g, m.document, m.textures[i], GRG_EDGE_DOC_TEXTURE);
	for (cgltf_size i = 0; i < data->materials_count; ++i)
		grg_link(g, m.document, m.materials[i], GRG_EDGE_DOC_MATERIAL);
	for (cgltf_size i = 0; i < data->meshes_count; ++i)
		grg_link(g, m.document, m.meshes[i], GRG_EDGE_DOC_MESH);
	for (cgltf_size i = 0; i < data->nodes_count; ++i)
		grg_link(g, m.document, m.nodes[i], GRG_EDGE_DOC_NODE);

	/* 3. buffers: external files (missing/self reference) */
	for (cgltf_size i = 0; i < data->buffers_count; ++i)
	{
		cgltf_buffer* b = &data->buffers[i];
		if (b->uri)
		{
			grg_attach_external_file(g, m.buffers[i], b->uri, doc_path,
					m.document, GRG_SEVERITY_ERROR);
		}
		else
		{
			grg_add_finding(g, GRG_SEVERITY_INFO, m.buffers[i],
					"embedded in GLB binary chunk (no external file)");
		}
	}

	/* 4. images: bufferView -> buffer, or external URI -> file */
	for (cgltf_size i = 0; i < data->images_count; ++i)
	{
		cgltf_image* im = &data->images[i];
		if (im->buffer_view && im->buffer_view->buffer)
		{
			cgltf_size buf_id = (cgltf_size)(im->buffer_view->buffer - data->buffers);
			grg_link(g, m.images[i], m.buffers[buf_id], GRG_EDGE_IMAGE_BUFFER);
			grg_add_finding(g, GRG_SEVERITY_INFO, m.images[i],
					"stored in buffer %u (bufferView)", (unsigned)buf_id);
		}
		else if (im->uri)
		{
			grg_attach_external_file(g, m.images[i], im->uri, doc_path,
					m.document, GRG_SEVERITY_ERROR);
		}
		else
		{
			g->count_missing++;
			grg_add_finding(g, GRG_SEVERITY_ERROR, m.images[i],
					"image has neither URI nor bufferView");
		}
	}

	/* 5. textures -> images (including KHR extension images when present) */
	for (cgltf_size i = 0; i < data->textures_count; ++i)
	{
		cgltf_texture* t = &data->textures[i];
		if (t->image)
		{
			cgltf_size img_id = (cgltf_size)(t->image - data->images);
			grg_link(g, m.textures[i], m.images[img_id], GRG_EDGE_TEX_IMAGE);
		}
		if (t->has_basisu && t->basisu_image)
		{
			cgltf_size img_id = (cgltf_size)(t->basisu_image - data->images);
			grg_link(g, m.textures[i], m.images[img_id], GRG_EDGE_TEX_IMAGE);
		}
		if (t->has_webp && t->webp_image)
		{
			cgltf_size img_id = (cgltf_size)(t->webp_image - data->images);
			grg_link(g, m.textures[i], m.images[img_id], GRG_EDGE_TEX_IMAGE);
		}
	}

	/* 6. materials -> textures (every texture view slot) */
	for (cgltf_size i = 0; i < data->materials_count; ++i)
	{
		cgltf_material* mat = &data->materials[i];
		const cgltf_texture_view* views[20];
		cgltf_size n = 0;
		views[n++] = &mat->pbr_metallic_roughness.base_color_texture;
		views[n++] = &mat->pbr_metallic_roughness.metallic_roughness_texture;
		views[n++] = &mat->pbr_specular_glossiness.diffuse_texture;
		views[n++] = &mat->pbr_specular_glossiness.specular_glossiness_texture;
		views[n++] = &mat->clearcoat.clearcoat_texture;
		views[n++] = &mat->clearcoat.clearcoat_roughness_texture;
		views[n++] = &mat->clearcoat.clearcoat_normal_texture;
		views[n++] = &mat->transmission.transmission_texture;
		views[n++] = &mat->volume.thickness_texture;
		views[n++] = &mat->specular.specular_texture;
		views[n++] = &mat->specular.specular_color_texture;
		views[n++] = &mat->sheen.sheen_color_texture;
		views[n++] = &mat->sheen.sheen_roughness_texture;
		views[n++] = &mat->iridescence.iridescence_texture;
		views[n++] = &mat->iridescence.iridescence_thickness_texture;
		views[n++] = &mat->diffuse_transmission.diffuse_transmission_texture;
		views[n++] = &mat->diffuse_transmission.diffuse_transmission_color_texture;
		views[n++] = &mat->anisotropy.anisotropy_texture;
		views[n++] = &mat->normal_texture;
		views[n++] = &mat->occlusion_texture;
		/* emissive is appended separately to stay within the array bound */
		for (cgltf_size k = 0; k < n; ++k)
		{
			if (views[k] && views[k]->texture)
			{
				cgltf_size tex_id = (cgltf_size)(views[k]->texture - data->textures);
				grg_link(g, m.materials[i], m.textures[tex_id], GRG_EDGE_MAT_TEX);
			}
		}
		if (mat->emissive_texture.texture)
		{
			cgltf_size tex_id = (cgltf_size)(mat->emissive_texture.texture - data->textures);
			grg_link(g, m.materials[i], m.textures[tex_id], GRG_EDGE_MAT_TEX);
		}
	}

	/* 7. meshes -> materials via primitives */
	for (cgltf_size i = 0; i < data->meshes_count; ++i)
	{
		cgltf_mesh* mesh = &data->meshes[i];
		for (cgltf_size p = 0; p < mesh->primitives_count; ++p)
		{
			cgltf_primitive* prim = &mesh->primitives[p];
			if (prim->material)
			{
				cgltf_size mat_id = (cgltf_size)(prim->material - data->materials);
				grg_link(g, m.meshes[i], m.materials[mat_id], GRG_EDGE_MESH_MAT);
			}
			for (cgltf_size mp = 0; mp < prim->mappings_count; ++mp)
			{
				if (prim->mappings[mp].material)
				{
					cgltf_size mat_id = (cgltf_size)(prim->mappings[mp].material - data->materials);
					grg_link(g, m.meshes[i], m.materials[mat_id], GRG_EDGE_MESH_MAT);
				}
			}
		}
	}

	/* 8. nodes -> mesh and node hierarchy edges */
	for (cgltf_size i = 0; i < data->nodes_count; ++i)
	{
		cgltf_node* nd = &data->nodes[i];
		if (nd->mesh)
		{
			cgltf_size mesh_id = (cgltf_size)(nd->mesh - data->meshes);
			grg_link(g, m.nodes[i], m.meshes[mesh_id], GRG_EDGE_NODE_MESH);
		}
		for (cgltf_size c = 0; c < nd->children_count; ++c)
		{
			if (nd->children[c])
			{
				cgltf_size child_id = (cgltf_size)(nd->children[c] - data->nodes);
				grg_link(g, m.nodes[i], m.nodes[child_id], GRG_EDGE_NODE_NODE);
			}
		}
	}

	/* 9. unused-resource check: semantic in-degree zero means nothing in
	 * the object graph consumes the resource (membership edges excluded).
	 * Buffers/nodes are intentionally not checked (bufferViews and scene
	 * roots are legitimate consumers not modeled as edges here). */
	for (cgltf_size i = 0; i < data->images_count; ++i)
	{
		if (g->refcount[m.images[i]] == 0)
			grg_add_finding(g, GRG_SEVERITY_WARN, m.images[i],
					"image %u is not referenced by any texture", (unsigned)i);
	}
	for (cgltf_size i = 0; i < data->textures_count; ++i)
	{
		if (g->refcount[m.textures[i]] == 0)
			grg_add_finding(g, GRG_SEVERITY_WARN, m.textures[i],
					"texture %u is not referenced by any material", (unsigned)i);
	}
	for (cgltf_size i = 0; i < data->materials_count; ++i)
	{
		if (g->refcount[m.materials[i]] == 0)
			grg_add_finding(g, GRG_SEVERITY_WARN, m.materials[i],
					"material %u is not referenced by any mesh primitive", (unsigned)i);
	}
	for (cgltf_size i = 0; i < data->meshes_count; ++i)
	{
		if (g->refcount[m.meshes[i]] == 0)
			grg_add_finding(g, GRG_SEVERITY_WARN, m.meshes[i],
					"mesh %u is not referenced by any node", (unsigned)i);
	}

	/* 10. adjacency and cycle detection */
	if (grg_build_adjacency(g) != 0)
	{
		free(m.buffers); free(m.images); free(m.textures);
		free(m.materials); free(m.meshes); free(m.nodes);
		return -1;
	}
	grg_detect_cycles(g);

	free(m.buffers); free(m.images); free(m.textures);
	free(m.materials); free(m.meshes); free(m.nodes);
	return 0;
}

/* ------------------------------------------------------------------ */
/* Reporting                                                          */
/* ------------------------------------------------------------------ */

static int grg_finding_order(const grg_finding* a, const grg_finding* b)
{
	/* higher severity first; ties broken alphabetically for stable output */
	if (a->severity != b->severity) return (int)b->severity - (int)a->severity;
	return strcmp(a->message, b->message);
}

static void grg_sort_findings(grg_graph* g)
{
	/* insertion sort: finding counts are small; deterministic ordering */
	for (cgltf_size i = 1; i < g->findings_count; ++i)
	{
		grg_finding key = g->findings[i];
		cgltf_size j = i;
		while (j > 0 && grg_finding_order(&g->findings[j - 1], &key) > 0)
		{
			g->findings[j] = g->findings[j - 1];
			--j;
		}
		g->findings[j] = key;
	}
}

static void grg_print_text(const char* doc_path, const grg_graph* g)
{
	printf("== glTF resource reference report ==\n");
	printf("document: %s\n", doc_path);
	printf("graph: %u nodes, %u edges\n",
			(unsigned)g->nodes_count, (unsigned)g->edges_count);
	printf("findings: %u total (ERROR=%u WARN=%u INFO=%u)\n",
			(unsigned)g->findings_count,
			(unsigned)g->count_by_severity[GRG_SEVERITY_ERROR],
			(unsigned)g->count_by_severity[GRG_SEVERITY_WARN],
			(unsigned)g->count_by_severity[GRG_SEVERITY_INFO]);
	printf("  missing files: %u | circular refs: %u | duplicate refs: %u\n",
			(unsigned)g->count_missing, (unsigned)g->count_cycles,
			(unsigned)g->count_duplicates);

	/* node inventory */
	printf("\n-- nodes --\n");
	for (cgltf_size i = 0; i < g->nodes_count; ++i)
	{
		const grg_node* n = &g->nodes[i];
		printf("[%u] %s", (unsigned)i, grg_kind_name(n->kind));
		if (n->name) printf(" name=\"%s\"", n->name);
		if (n->kind != GRG_KIND_DOCUMENT && n->kind != GRG_KIND_FILE)
			printf(" index=%u", (unsigned)n->index);
		if (n->uri && n->kind != GRG_KIND_FILE) printf(" uri=\"%s\"", n->uri);
		if (n->kind == GRG_KIND_FILE) printf(" path=\"%s\"", n->uri);
		printf(" in=%u out=%u\n", (unsigned)n->in_degree, (unsigned)n->out_degree);
	}

	printf("\n-- edges --\n");
	for (cgltf_size i = 0; i < g->edges_count; ++i)
	{
		const grg_edge* e = &g->edges[i];
		printf("[%u] -> [%u] %s\n", (unsigned)e->src, (unsigned)e->dst,
				grg_edge_name(e->kind));
	}

	printf("\n-- findings (severity order) --\n");
	for (cgltf_size i = 0; i < g->findings_count; ++i)
	{
		const grg_finding* f = &g->findings[i];
		printf("%-5s %s\n", grg_severity_name(f->severity), f->message);
	}
}

static void grg_print_json(const char* doc_path, const grg_graph* g)
{
	printf("{\n");
	printf("  \"document\": \"%s\",\n", doc_path);
	printf("  \"nodes_count\": %u,\n", (unsigned)g->nodes_count);
	printf("  \"edges_count\": %u,\n", (unsigned)g->edges_count);
	printf("  \"summary\": {\"error\": %u, \"warn\": %u, \"info\": %u,\n",
			(unsigned)g->count_by_severity[GRG_SEVERITY_ERROR],
			(unsigned)g->count_by_severity[GRG_SEVERITY_WARN],
			(unsigned)g->count_by_severity[GRG_SEVERITY_INFO]);
	printf("    \"missing\": %u, \"circular\": %u, \"duplicate\": %u},\n",
			(unsigned)g->count_missing, (unsigned)g->count_cycles,
			(unsigned)g->count_duplicates);
	printf("  \"findings\": [\n");
	for (cgltf_size i = 0; i < g->findings_count; ++i)
	{
		const grg_finding* f = &g->findings[i];
		/* JSON-escape the message (messages contain only paths/names we control) */
		printf("    {\"severity\": \"%s\", \"message\": \"", grg_severity_name(f->severity));
		for (const char* p = f->message; *p; ++p)
		{
			if (*p == '"' || *p == '\\') putchar('\\');
			putchar(*p);
		}
		printf("\"}%s\n", i + 1 == g->findings_count ? "" : ",");
	}
	printf("  ]\n}\n");
}

/* ------------------------------------------------------------------ */
/* Perf mode                                                          */
/* ------------------------------------------------------------------ */

static double grg_seconds_now(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int grg_perf_compare_double(const void* a, const void* b)
{
	double da = *(const double*)a, db = *(const double*)b;
	return (da > db) - (da < db);
}

static int run_perf(cgltf_data* data, const char* doc_path, int iterations)
{
	if (iterations <= 0) iterations = 200;

	double* times = (double*)malloc(sizeof(double) * iterations);
	grg_graph g;
	if (!times) return -1;

	/* warmup + correctness: every iteration must produce the same counts */
	if (grg_build(data, doc_path, &g) != 0)
	{
		fprintf(stderr, "graph build failed\n");
		free(times);
		return -1;
	}
	unsigned expect_nodes = (unsigned)g.nodes_count;
	unsigned expect_edges = (unsigned)g.edges_count;
	unsigned expect_findings = (unsigned)g.findings_count;
	grg_graph_free(&g);

	for (int it = 0; it < iterations; ++it)
	{
		double t0 = grg_seconds_now();
		if (grg_build(data, doc_path, &g) != 0)
		{
			fprintf(stderr, "graph build failed on iteration %d\n", it);
			free(times);
			return -1;
		}
		double t1 = grg_seconds_now();
		if ((unsigned)g.nodes_count != expect_nodes ||
				(unsigned)g.edges_count != expect_edges ||
				(unsigned)g.findings_count != expect_findings)
		{
			fprintf(stderr, "non-deterministic graph output on iteration %d\n", it);
			grg_graph_free(&g);
			free(times);
			return -1;
		}
		times[it] = (t1 - t0) * 1000.0;
		grg_graph_free(&g);
	}

	qsort(times, iterations, sizeof(double), grg_perf_compare_double);
	double sum = 0;
	for (int i = 0; i < iterations; ++i) sum += times[i];
	printf("perf: iterations=%d nodes(per run)=%u edges=%u\n",
			iterations, expect_nodes, expect_edges);
	printf("time_ms: min=%.4f median=%.4f mean=%.4f max=%.4f\n",
			times[0], times[iterations / 2], sum / iterations, times[iterations - 1]);
	printf("budget_20ms: %s\n", times[iterations / 2] < 20.0 ? "PASS" : "FAIL");
	free(times);
	return 0;
}

/* ------------------------------------------------------------------ */
/* Entry point                                                        */
/* ------------------------------------------------------------------ */

int main(int argc, char** argv)
{
	int perf_mode = 0;
	int json_mode = 0;
	const char* path = NULL;
	int perf_iterations = 200;

	for (int i = 1; i < argc; ++i)
	{
		if (strcmp(argv[i], "--perf") == 0) perf_mode = 1;
		else if (strcmp(argv[i], "--json") == 0) json_mode = 1;
		else if (argv[i][0] >= '0' && argv[i][0] <= '9') perf_iterations = atoi(argv[i]);
		else path = argv[i];
	}

	if (!path)
	{
		fprintf(stderr,
				"usage: %s [--perf [iterations]] [--json] <model.gltf|model.glb>\n",
				argv[0]);
		return 2;
	}

	cgltf_options options;
	memset(&options, 0, sizeof(options));
	cgltf_data* data = NULL;
	cgltf_result result = cgltf_parse_file(&options, path, &data);
	if (result != cgltf_result_success)
	{
		fprintf(stderr, "cgltf_parse_file failed: %d\n", (int)result);
		return 1;
	}

	int rc = 0;
	if (perf_mode)
	{
		rc = run_perf(data, path, perf_iterations);
	}
	else
	{
		grg_graph g;
		if (grg_build(data, path, &g) != 0)
		{
			fprintf(stderr, "graph build failed (out of memory)\n");
			cgltf_free(data);
			return 1;
		}
		grg_sort_findings(&g);
		if (json_mode) grg_print_json(path, &g);
		else grg_print_text(path, &g);
		grg_graph_free(&g);
	}

	cgltf_free(data);
	return rc;
}
