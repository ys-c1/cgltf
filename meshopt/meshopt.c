/*
 * meshopt.c - glTF mesh optimizer built on top of cgltf.
 *
 * Pipeline per triangle primitive:
 *   1. Vertex deduplication - vertices whose position/normal/UV (and every
 *      other per-vertex attribute, including morph target deltas) are bit
 *      identical are merged; primitives without indices are indexed first.
 *   2. Triangle reordering  - Tom Forsyth's linear-speed vertex cache
 *      optimisation (no third-party code, pure C99).
 *   3. Vertex reordering    - vertices are remapped in first-use order so
 *      post-transform cache locality and vertex prefetch improve.
 *   4. The optimised mesh is written out as a self-contained GLB via the
 *      bundled cgltf_write.h; statistics (vertex/index counts, triangle
 *      order changes, simulated post-transform cache and vertex-fetch +
 *      index bandwidth estimates) are emitted as JSON on stdout.
 *
 * Usage:
 *   meshopt <input.gltf|input.glb> <output.glb> [--bench]
 *
 * With --bench the tool additionally runs the optimisation on a synthetic
 * 100k-vertex indexed mesh and reports wall-clock timing.
 *
 * No third-party dependencies beyond the cgltf.h / cgltf_write.h shipped in
 * this repository; no network access.
 */

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 199309L /* clock_gettime */
#endif

#define CGLTF_IMPLEMENTATION
#define CGLTF_WRITE_IMPLEMENTATION
#include "../cgltf_write.h"

#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* cgltf only typedefs a few integer widths; use the stdint names it
 * already relies on internally. */
typedef uint8_t cgltf_ubyte;
typedef uint16_t cgltf_ushort;
typedef uint32_t cgltf_uint32;

/* ------------------------------------------------------------------ */
/* Small helpers                                                       */
/* ------------------------------------------------------------------ */

static void die(const char* fmt, ...)
{
	va_list ap;
	va_start(ap, fmt);
	vfprintf(stderr, fmt, ap);
	va_end(ap);
	fputc('\n', stderr);
	exit(1);
}

static void* xmalloc(size_t size)
{
	void* p = malloc(size ? size : 1);
	if (!p) die("out of memory (requested %zu bytes)", size);
	return p;
}

static void* xcalloc(size_t n, size_t size)
{
	void* p = calloc(n ? n : 1, size ? size : 1);
	if (!p) die("out of memory (requested %zu*%zu bytes)", n, size);
	return p;
}

static char* xstrdup(const char* s)
{
	size_t n = strlen(s) + 1;
	char* p = (char*)xmalloc(n);
	memcpy(p, s, n);
	return p;
}

/* ------------------------------------------------------------------ */
/* FNV-1a hashing over raw vertex attribute bytes                      */
/* ------------------------------------------------------------------ */

#define HASH_EMPTY ((cgltf_uint)0xFFFFFFFFu)

/* Fast hash over a byte range: mix 32-bit words (the common case for
 * float/uint vertex data) then fold in any 1-3 trailing bytes. */
static cgltf_uint hash_fast(const cgltf_ubyte* p, cgltf_size n, cgltf_uint h)
{
	cgltf_size words = n >> 2;
	const cgltf_uint32* wp = (const cgltf_uint32*)p;
	for (cgltf_size i = 0; i < words; ++i)
	{
		cgltf_uint32 w = wp[i];
		h ^= w;
		h *= 16777619u;
		h ^= w >> 16;
	}
	for (cgltf_size i = words * 4; i < n; ++i)
	{
		h ^= p[i];
		h *= 16777619u;
	}
	return h;
}

/* ------------------------------------------------------------------ */
/* Attribute blobs                                                     */
/*                                                                     */
/* Float attributes keep their raw float bits so exact duplicates      */
/* merge while distinct values never do. Integer attributes (JOINTS_*) */
/* keep one uint32 per component. All reads go through cgltf so packed */
/* component types, normalization and sparse accessors are handled.    */
/* ------------------------------------------------------------------ */

typedef struct AttrBlob
{
	const char* name;       /* glTF attribute name, e.g. POSITION/NORMAL  */
	cgltf_attribute_type type;
	cgltf_int set_index;    /* attribute set index (-1 for none)         */
	cgltf_size components;
	cgltf_size elem_bytes;
	cgltf_int is_integer;
	cgltf_ubyte* bytes;     /* vertex_count * elem_bytes                 */
} AttrBlob;

static AttrBlob make_float_blob(cgltf_attribute* attr, cgltf_size vertex_count)
{
	cgltf_accessor* acc = attr->data;
	cgltf_size comps = cgltf_num_components(acc->type);
	AttrBlob blob;
	blob.name = attr->name ? attr->name : "";
	blob.type = attr->type;
	blob.set_index = attr->index;
	blob.components = comps;
	blob.elem_bytes = comps * sizeof(cgltf_float);
	blob.is_integer = 0;
	blob.bytes = (cgltf_ubyte*)xmalloc(vertex_count * blob.elem_bytes);

	/* unpack_floats emits tightly packed floats and ignores interleaved
	 * buffer-view strides, so read vertex-by-vertex through the public
	 * read_float API instead (handles stride, normalization and sparse).
	 * read_float returns cgltf_bool (1 on success), not a component count. */
	cgltf_float tmp[16];
	for (cgltf_size v = 0; v < vertex_count; ++v)
	{
		if (!cgltf_accessor_read_float(acc, v, tmp, comps))
			die("failed to read float attribute at vertex %zu", v);
		memcpy(blob.bytes + v * blob.elem_bytes, tmp, blob.elem_bytes);
	}
	return blob;
}

static AttrBlob make_integer_blob(cgltf_attribute* attr, cgltf_size vertex_count)
{
	cgltf_accessor* acc = attr->data;
	cgltf_size comps = cgltf_num_components(acc->type);
	AttrBlob blob;
	blob.name = attr->name ? attr->name : "";
	blob.type = attr->type;
	blob.set_index = attr->index;
	blob.components = comps;
	blob.elem_bytes = comps * sizeof(cgltf_uint32);
	blob.is_integer = 1;
	blob.bytes = (cgltf_ubyte*)xmalloc(vertex_count * blob.elem_bytes);

	for (cgltf_size v = 0; v < vertex_count; ++v)
	{
		cgltf_uint32* out = (cgltf_uint32*)(blob.bytes + v * blob.elem_bytes);
		for (cgltf_size c = 0; c < comps; ++c)
		{
			cgltf_uint val = 0;
			if (!cgltf_accessor_read_uint(acc, v * comps + c, &val, 1))
				die("failed to read integer attribute at vertex %zu", v);
			out[c] = (cgltf_uint32)val;
		}
	}
	return blob;
}

static void free_blob(AttrBlob* blob)
{
	free(blob->bytes);
}

/* Morph target attributes first (target order preserved), then regular
 * primitive attributes. */
static AttrBlob* collect_blobs(cgltf_primitive* prim, cgltf_size vertex_count,
		cgltf_size* out_count)
{
	cgltf_size cap = prim->attributes_count;
	for (cgltf_size m = 0; m < prim->targets_count; ++m)
		cap += prim->targets[m].attributes_count;

	AttrBlob* blobs = (AttrBlob*)xcalloc(cap, sizeof(AttrBlob));
	cgltf_size n = 0;

	for (cgltf_size m = 0; m < prim->targets_count; ++m)
	{
		for (cgltf_size a = 0; a < prim->targets[m].attributes_count; ++a)
		{
			cgltf_attribute* attr = &prim->targets[m].attributes[a];
			blobs[n++] = attr->type == cgltf_attribute_type_joints
					? make_integer_blob(attr, vertex_count)
					: make_float_blob(attr, vertex_count);
		}
	}
	for (cgltf_size a = 0; a < prim->attributes_count; ++a)
	{
		cgltf_attribute* attr = &prim->attributes[a];
		blobs[n++] = attr->type == cgltf_attribute_type_joints
				? make_integer_blob(attr, vertex_count)
				: make_float_blob(attr, vertex_count);
	}

	*out_count = n;
	return blobs;
}

/* ------------------------------------------------------------------ */
/* Post-transform cache simulation (16-entry FIFO, Forsyth model)      */
/* ------------------------------------------------------------------ */

static cgltf_size simulate_cache(const cgltf_uint32* indices, cgltf_size tri_count,
		cgltf_size cache_size)
{
	cgltf_size misses = 0;
	cgltf_uint32* cache = (cgltf_uint32*)xmalloc(cache_size * sizeof(cgltf_uint32));
	for (cgltf_size i = 0; i < cache_size; ++i) cache[i] = HASH_EMPTY;
	cgltf_size fill = 0;

	for (cgltf_size t = 0; t < tri_count; ++t)
	{
		for (cgltf_size k = 0; k < 3; ++k)
		{
			cgltf_uint32 v = indices[t * 3 + k];
			cgltf_size hit = 0;
			for (cgltf_size j = 0; j < fill; ++j)
			{
				if (cache[j] == v) { hit = 1; break; }
			}
			if (!hit)
			{
				++misses;
				if (fill < cache_size)
				{
					cache[fill++] = v;
				}
				else
				{
					memmove(cache, cache + 1, (cache_size - 1) * sizeof(cgltf_uint32));
					cache[cache_size - 1] = v;
				}
			}
		}
	}
	free(cache);
	return misses;
}

/* ------------------------------------------------------------------ */
/* Forsyth's linear-speed vertex cache optimisation                    */
/* Reference: Tom Forsyth, "Linear-Speed Vertex Cache Optimisation".   */
/* Reorders triangles in place; winding order is preserved.            */
/* ------------------------------------------------------------------ */

#define FORSYTH_CACHE_SIZE 16
#define FORSYTH_LAST_TRI_SCORE 0.75f
#define FORSYTH_VALENCE_BOOST_SCALE 2.0f

/* Precomputed position component of the Forsyth score. */
static cgltf_float forsyth_pos_score[FORSYTH_CACHE_SIZE];
/* Precomputed valence boost 2/sqrt(v) for small valences; practical
 * meshes rarely exceed a handful of triangles per vertex. */
#define FORSYTH_VALENCE_TABLE 64
static cgltf_float forsyth_valence_boost[FORSYTH_VALENCE_TABLE + 1];
static cgltf_int forsyth_tables_ready = 0;

static void forsyth_init_tables(void)
{
	if (forsyth_tables_ready) return;
	for (cgltf_int i = 0; i < FORSYTH_CACHE_SIZE; ++i)
	{
		if (i < 3)
			forsyth_pos_score[i] = FORSYTH_LAST_TRI_SCORE;
		else
		{
			cgltf_float f = 1.0f - (cgltf_float)(i - 3)
					/ (cgltf_float)(FORSYTH_CACHE_SIZE - 3);
			forsyth_pos_score[i] = f * f;
		}
	}
	for (cgltf_int v = 1; v <= FORSYTH_VALENCE_TABLE; ++v)
		forsyth_valence_boost[v] =
				FORSYTH_VALENCE_BOOST_SCALE / sqrtf((cgltf_float)v);
	forsyth_tables_ready = 1;
}

static cgltf_float forsyth_score(cgltf_uint32 num_active_tris, cgltf_int cache_pos)
{
	if (num_active_tris == 0) return -1.0f;

	cgltf_float score = (cache_pos >= 0)
			? forsyth_pos_score[cache_pos]
			: 0.0f; /* not cached: only the valence boost applies */
	if (num_active_tris <= FORSYTH_VALENCE_TABLE)
		score += forsyth_valence_boost[num_active_tris];
	else
		score += FORSYTH_VALENCE_BOOST_SCALE
				/ sqrtf((cgltf_float)num_active_tris);
	return score;
}

/* True linear-time Forsyth optimisation.
 *
 * Each vertex owns an intrusive linked list of its still-active
 * triangles; a FIFO "ready" queue holds every active triangle that
 * touches one of the vertices in the 16-entry LRU cache. When the cache
 * evicts a vertex its remaining triangles lose readiness, and when the
 * queue drains the next unprocessed triangle is found via a monotonic
 * cursor (each triangle is scanned at most once that way). Every
 * vertex/triangle is touched a bounded number of times, so the whole
 * pass is O(vertex_count + tri_count).
 *
 * out_tri_order[t] receives the original index of the t-th emitted
 * triangle so the caller can quantify order changes. */
static void forsyth_reorder(cgltf_uint32* indices, cgltf_size index_count,
		cgltf_size vertex_count, cgltf_uint32* out_tri_order)
{
	cgltf_size tri_count = index_count / 3;
	if (tri_count == 0) return;
	forsyth_init_tables();

	/* Linear-time Forsyth with a shift-free circular vertex cache.
	 *
	 * The cache is a fixed ring; an emitting triangle writes its three
	 * corners at three cursor slots, overwriting the evicted entries, so
	 * there is never any memmove. Each vertex's slot is mirrored in
	 * cache_pos[] and its recency is the forward distance from the cursor
	 * (used directly as the Forsyth cache position). A 32-entry candidate
	 * ring is maintained with O(1) membership flags; CSR adjacency is
	 * enumerated only when a vertex enters the cache. Scores come from
	 * precomputed tables. O(vertex_count + tri_count). */

	enum { CACHE_SIZE = FORSYTH_CACHE_SIZE };
	enum { RING_SIZE  = FORSYTH_CACHE_SIZE * 2 };

	cgltf_uint32* v_active  = (cgltf_uint32*)xcalloc(vertex_count, sizeof(cgltf_uint32));
	cgltf_int*    cache_pos = (cgltf_int*)xmalloc(vertex_count * sizeof(cgltf_int));
	cgltf_float*  v_score   = (cgltf_float*)xmalloc(vertex_count * sizeof(cgltf_float));
	for (cgltf_size v = 0; v < vertex_count; ++v) cache_pos[v] = -1;

	for (cgltf_size i = 0; i < index_count; ++i)
		++v_active[indices[i]];
	for (cgltf_size v = 0; v < vertex_count; ++v)
		v_score[v] = forsyth_score(v_active[v], -1);

	/* CSR per-vertex triangle adjacency. */
	cgltf_uint32* vt_begin = (cgltf_uint32*)xmalloc((vertex_count + 1) * sizeof(cgltf_uint32));
	memset(vt_begin, 0, (vertex_count + 1) * sizeof(cgltf_uint32));
	for (cgltf_size i = 0; i < index_count; ++i)
		++vt_begin[indices[i] + 1];
	for (cgltf_size v = 1; v <= vertex_count; ++v)
		vt_begin[v] += vt_begin[v - 1];
	cgltf_uint32* vt_list = (cgltf_uint32*)xmalloc(index_count * sizeof(cgltf_uint32));
	cgltf_uint32* vt_cur  = (cgltf_uint32*)xmalloc(vertex_count * sizeof(cgltf_uint32));
	memcpy(vt_cur, vt_begin, vertex_count * sizeof(cgltf_uint32));
	for (cgltf_uint32 t = 0; t < tri_count; ++t)
		for (cgltf_size k = 0; k < 3; ++k)
			vt_list[vt_cur[indices[t * 3 + k]]++] = t;
	free(vt_cur);

	cgltf_ubyte* t_alive  = (cgltf_ubyte*)xcalloc(tri_count, 1);
	cgltf_ubyte* t_inring = (cgltf_ubyte*)xcalloc(tri_count, 1);
	for (cgltf_uint32 t = 0; t < tri_count; ++t) t_alive[t] = 1;

	cgltf_uint32 ring[RING_SIZE];
	cgltf_int ring_count = 0;
	cgltf_uint32 next_tri = 0;

	cgltf_uint32* work = (cgltf_uint32*)xmalloc(index_count * sizeof(cgltf_uint32));

	cgltf_uint32 vcache[CACHE_SIZE];
	for (cgltf_int i = 0; i < CACHE_SIZE; ++i) vcache[i] = HASH_EMPTY;
	/* After writing three corners, slot cursor..cursor+2 are newest; the
	 * oldest cached vertex is nearest (cursor+3) mod CACHE_SIZE. */
	cgltf_int cursor = 0;
	cgltf_int cache_fill = 0;
	cgltf_uint32 emitted = 0;

	#define RING_ADD(t) do {                                                  \
		cgltf_uint32 _t = (t);                                               \
		if (t_alive[_t] && !t_inring[_t] && ring_count < RING_SIZE)         \
		{                                                                    \
			t_inring[_t] = 1;                                                \
			ring[ring_count++] = _t;                                         \
		}                                                                    \
	} while (0)

	/* Insert a vertex at slot (cursor + slot_off), evicting the occupant.
	 * The vertex's adjacency is enumerated only on a genuine insertion. */
	#define CACHE_INSERT_AT(v, slot_off) do {                                 \
		cgltf_uint32 _v = (v);                                               \
		if (cache_pos[_v] < 0)                                               \
		{                                                                    \
			cgltf_int _slot = (cursor + (slot_off)) % CACHE_SIZE;          \
			cgltf_uint32 _old = vcache[_slot];                              \
			if (_old != HASH_EMPTY) cache_pos[_old] = -1;                   \
			vcache[_slot] = _v;                                              \
			cache_pos[_v] = _slot;                                           \
			if (cache_fill < CACHE_SIZE) ++cache_fill;                      \
			for (cgltf_uint32 j = vt_begin[_v]; j < vt_begin[_v + 1]; ++j) \
				RING_ADD(vt_list[j]);                                        \
		}                                                                    \
	} while (0)

	/* Re-score every occupied slot from its recency distance. Newest three
	 * positions (0..2) get the last-triangle score. */
	#define RESCORE_CACHE() do {                                             \
		for (cgltf_int c = 0; c < cache_fill; ++c)                          \
		{                                                                    \
			cgltf_int slot = (cursor + c) % CACHE_SIZE;                    \
			cgltf_uint32 v = vcache[slot];                                  \
			if (v == HASH_EMPTY) continue;                                  \
			cache_pos[v] = slot;                                            \
			v_score[v] = forsyth_score(v_active[v], c);                    \
		}                                                                    \
	} while (0)

	while (emitted < tri_count)
	{
		cgltf_uint32 best_tri = HASH_EMPTY;
		cgltf_int best_slot = -1;

		/* Compact the ring and pick the best live candidate. */
		cgltf_float best_score = -1.0f;
		cgltf_int write = 0;
		for (cgltf_int i = 0; i < ring_count; ++i)
		{
			cgltf_uint32 t = ring[i];
			if (!t_alive[t]) { t_inring[t] = 0; continue; }
			ring[write++] = t;
			cgltf_float s = v_score[indices[t * 3]]
					+ v_score[indices[t * 3 + 1]]
					+ v_score[indices[t * 3 + 2]];
			if (s > best_score) { best_score = s; best_tri = t; best_slot = write - 1; }
		}
		ring_count = write;

		if (best_tri == HASH_EMPTY)
		{
			while (next_tri < tri_count && !t_alive[next_tri])
				++next_tri;
			if (next_tri >= tri_count) break;
			best_tri = next_tri++;
			best_slot = -1;
		}

		out_tri_order[emitted] = best_tri;
		memcpy(work + emitted * 3, indices + best_tri * 3,
				3 * sizeof(cgltf_uint32));
		t_alive[best_tri] = 0;
		t_inring[best_tri] = 0;
		++emitted;

		if (best_slot >= 0 && ring_count > 0)
		{
			ring[best_slot] = ring[--ring_count];
			t_inring[ring[best_slot]] = 1;
		}

		for (cgltf_size k = 0; k < 3; ++k)
			--v_active[indices[best_tri * 3 + k]];

		/* Write the three corners newest-first at descending cursor slots,
		 * then re-score the small cache. */
		/* Reserve three consecutive newest slots (cursor..cursor+2),
		 * evicting the occupants, then re-score the small cache. */
		cursor = (cursor + CACHE_SIZE - 3) % CACHE_SIZE;
		CACHE_INSERT_AT(indices[best_tri * 3], 0);
		CACHE_INSERT_AT(indices[best_tri * 3 + 1], 1);
		CACHE_INSERT_AT(indices[best_tri * 3 + 2], 2);
		RESCORE_CACHE();
	}

	#undef RESCORE_CACHE
	#undef CACHE_INSERT_AT
	#undef RING_ADD

	memcpy(indices, work, index_count * sizeof(cgltf_uint32));
	free(vt_list);
	free(vt_begin);
	free(work);
	free(t_inring);
	free(t_alive);
	free(v_score);
	free(cache_pos);
	free(v_active);
}
/* ------------------------------------------------------------------ */
/* Per-primitive optimisation                                          */
/* ------------------------------------------------------------------ */

typedef struct OptResult
{
	cgltf_size in_vertex_count;
	cgltf_size in_index_count;     /* 0 for a non-indexed primitive     */
	cgltf_size tri_count;

	cgltf_size out_vertex_count;
	cgltf_uint32* out_indices;     /* tri_count * 3                     */

	cgltf_size blob_count;
	AttrBlob* in_blobs;            /* raw, input vertex order           */
	AttrBlob* out_blobs;           /* compacted, first-use vertex order */

	cgltf_size triangles_reordered;
	cgltf_size fetched_before;
	cgltf_size fetched_after;
	double acmr_before;
	double acmr_after;
} OptResult;

static void free_opt(OptResult* res)
{
	if (!res) return;
	for (cgltf_size i = 0; i < res->blob_count; ++i)
	{
		free_blob(&res->in_blobs[i]);
		free_blob(&res->out_blobs[i]);
	}
	free(res->in_blobs);
	free(res->out_blobs);
	free(res->out_indices);
	free(res);
}

static OptResult* optimize_primitive(cgltf_primitive* prim)
{
	if (prim->type != cgltf_primitive_type_triangles || prim->has_draco_mesh_compression)
		return NULL;
	if (prim->attributes_count == 0)
		return NULL;

	cgltf_size in_vertex_count = prim->attributes[0].data->count;
	for (cgltf_size a = 0; a < prim->attributes_count; ++a)
	{
		if (prim->attributes[a].data->count != in_vertex_count)
			die("primitive attribute count mismatch");
	}

	/* 1. Corner index list (indexed primitives stay indexed; non-indexed
	 *    primitives get an implicit 0..N-1 index buffer). */
	cgltf_size in_index_count = prim->indices ? prim->indices->count : 0;
	cgltf_size tri_count;
	cgltf_uint32* corner;
	if (prim->indices)
	{
		tri_count = in_index_count / 3;
		corner = (cgltf_uint32*)xmalloc(in_index_count * sizeof(cgltf_uint32));
		for (cgltf_size i = 0; i < in_index_count; ++i)
		{
			cgltf_uint v = 0;
			if (!cgltf_accessor_read_uint(prim->indices, i, &v, 1))
				die("failed to read index %zu", i);
			corner[i] = (cgltf_uint32)v;
		}
	}
	else
	{
		tri_count = in_vertex_count / 3;
		in_index_count = tri_count * 3;
		corner = (cgltf_uint32*)xmalloc(in_index_count * sizeof(cgltf_uint32));
		for (cgltf_size i = 0; i < in_index_count; ++i) corner[i] = (cgltf_uint32)i;
	}

	OptResult* res = (OptResult*)xcalloc(1, sizeof(OptResult));
	res->in_vertex_count = in_vertex_count;
	res->in_index_count = prim->indices ? in_index_count : 0;
	res->tri_count = tri_count;

	res->fetched_before = simulate_cache(corner, tri_count, FORSYTH_CACHE_SIZE);
	res->acmr_before = tri_count ? (double)res->fetched_before / (double)tri_count : 0.0;

	/* 2. Load every per-vertex attribute (morph deltas included). */
	res->in_blobs = collect_blobs(prim, in_vertex_count, &res->blob_count);
	res->blob_count = res->blob_count; /* collect_blobs already sets via ptr */

	/* 3. Deduplicate: open-addressing hash on concatenated attribute bytes.
	 *    Each vertex's attributes are gathered into one contiguous key so
	 *    hashing and comparison touch a single cache line sequence.
	 *    dedup[v] = unique id; unique_src[id] = original vertex index. */
	cgltf_size key_stride = 0;
	for (cgltf_size b = 0; b < res->blob_count; ++b)
		key_stride += res->in_blobs[b].elem_bytes;

	cgltf_ubyte* keys = (cgltf_ubyte*)xmalloc(
			in_vertex_count * (key_stride ? key_stride : 1));
	for (cgltf_size v = 0; v < in_vertex_count; ++v)
	{
		cgltf_size off = 0;
		for (cgltf_size b = 0; b < res->blob_count; ++b)
		{
			AttrBlob* blob = &res->in_blobs[b];
			memcpy(keys + v * key_stride + off,
					blob->bytes + v * blob->elem_bytes, blob->elem_bytes);
			off += blob->elem_bytes;
		}
	}

	/* Power-of-two table (~50% load) so probing uses a mask, not a
	 * modulo, and the table stays cache-resident. */
	cgltf_size table_cap = 16;
	while (table_cap < in_vertex_count * 2u) table_cap <<= 1;
	cgltf_size table_mask = table_cap - 1;
	cgltf_uint* table = (cgltf_uint*)xmalloc(table_cap * sizeof(cgltf_uint));
	for (cgltf_size i = 0; i < table_cap; ++i) table[i] = HASH_EMPTY;

	cgltf_uint32* dedup = (cgltf_uint32*)xmalloc(in_vertex_count * sizeof(cgltf_uint32));
	cgltf_uint32* unique_src = (cgltf_uint32*)xmalloc(in_vertex_count * sizeof(cgltf_uint32));
	cgltf_size unique_count = 0;

	for (cgltf_size v = 0; v < in_vertex_count; ++v)
	{
		const cgltf_ubyte* kv = keys + (cgltf_size)v * key_stride;
		cgltf_uint h = hash_fast(kv, key_stride, 2166136261u);
		cgltf_size slot = h & table_mask;
		while (table[slot] != HASH_EMPTY)
		{
			cgltf_uint32 cand_src = unique_src[table[slot]];
			if (memcmp(kv, keys + (cgltf_size)cand_src * key_stride,
					key_stride) == 0)
			{
				dedup[v] = table[slot];
				goto found;
			}
			slot = (slot + 1) & table_mask;
		}
		dedup[v] = (cgltf_uint32)unique_count;
		unique_src[unique_count] = (cgltf_uint32)v;
		table[slot] = (cgltf_uint)unique_count;
		++unique_count;
found:;
	}
	free(keys);
	free(table);

	for (cgltf_size i = 0; i < in_index_count; ++i)
		corner[i] = dedup[corner[i]];
	free(dedup);

	/* 4. Forsyth triangle reordering (on deduplicated indices). */
	cgltf_uint32* tri_order = (cgltf_uint32*)xmalloc(tri_count * sizeof(cgltf_uint32));
	forsyth_reorder(corner, in_index_count, unique_count, tri_order);
	for (cgltf_size t = 0; t < tri_count; ++t)
		if (tri_order[t] != t) ++res->triangles_reordered;
	free(tri_order);

	/* 5. Vertex compaction in first-use order.
	 *    compact[u] = new vertex id; new_to_src[n] = original vertex. */
	cgltf_uint32* compact = (cgltf_uint32*)xmalloc(unique_count * sizeof(cgltf_uint32));
	cgltf_uint32* new_to_src = (cgltf_uint32*)xmalloc(unique_count * sizeof(cgltf_uint32));
	cgltf_ubyte* seen = (cgltf_ubyte*)xcalloc(unique_count, 1);
	cgltf_size new_count = 0;
	for (cgltf_size i = 0; i < in_index_count; ++i)
	{
		cgltf_uint32 u = corner[i];
		if (!seen[u])
		{
			seen[u] = 1;
			compact[u] = (cgltf_uint32)new_count;
			new_to_src[new_count] = unique_src[u];
			++new_count;
		}
	}
	free(seen);
	free(unique_src);
	for (cgltf_size i = 0; i < in_index_count; ++i)
		corner[i] = compact[corner[i]];
	free(compact);

	res->out_vertex_count = new_count;
	res->out_indices = corner;

	/* 6. Rewrite attribute blobs in the compacted order. */
	res->out_blobs = (AttrBlob*)xcalloc(res->blob_count, sizeof(AttrBlob));
	for (cgltf_size b = 0; b < res->blob_count; ++b)
	{
		AttrBlob* src = &res->in_blobs[b];
		AttrBlob* dst = &res->out_blobs[b];
		dst->name = src->name;
		dst->type = src->type;
		dst->set_index = src->set_index;
		dst->components = src->components;
		dst->elem_bytes = src->elem_bytes;
		dst->is_integer = src->is_integer;
		dst->bytes = (cgltf_ubyte*)xmalloc(new_count * src->elem_bytes);
		for (cgltf_size n = 0; n < new_count; ++n)
		{
			memcpy(dst->bytes + n * src->elem_bytes,
					src->bytes + (cgltf_size)new_to_src[n] * src->elem_bytes,
					src->elem_bytes);
		}
	}
	free(new_to_src);

	res->fetched_after = simulate_cache(corner, tri_count, FORSYTH_CACHE_SIZE);
	res->acmr_after = tri_count ? (double)res->fetched_after / (double)tri_count : 0.0;
	return res;
}

/* ------------------------------------------------------------------ */
/* GLB output: one embedded buffer, one buffer view per accessor       */
/* ------------------------------------------------------------------ */

typedef struct OutModel
{
	cgltf_data* data;
	cgltf_buffer* buffer;
	cgltf_ubyte* bin;
	cgltf_size bin_size;
	cgltf_size view_count;
	cgltf_size accessor_count;
	cgltf_size view_cap;
	cgltf_size accessor_cap;
	cgltf_material* src_materials; /* base of the parsed material array */
} OutModel;

static cgltf_size align4(cgltf_size x)
{
	return (x + 3) & ~(cgltf_size)3;
}

static OutModel* out_model_create(void)
{
	OutModel* m = (OutModel*)xcalloc(1, sizeof(OutModel));
	m->data = (cgltf_data*)xcalloc(1, sizeof(cgltf_data));
	m->data->asset.version = xstrdup("2.0");
	m->data->asset.generator = xstrdup("cgltf-meshopt");
	m->data->file_type = cgltf_file_type_glb;
	return m;
}

/* Pre-allocate the accessor / buffer-view arrays up front. The bundled
 * writer derives array indices from pointer differences
 * (ptr - array_base), so the bases must never move afterwards. */
static void out_model_reserve(OutModel* m, cgltf_size view_cap,
		cgltf_size accessor_cap)
{
	m->view_cap = view_cap;
	m->accessor_cap = accessor_cap;
	m->data->buffer_views = (cgltf_buffer_view*)xcalloc(
			view_cap ? view_cap : 1, sizeof(cgltf_buffer_view));
	m->data->accessors = (cgltf_accessor*)xcalloc(
			accessor_cap ? accessor_cap : 1, sizeof(cgltf_accessor));
}

static cgltf_accessor* out_add_accessor(OutModel* m, const cgltf_ubyte* data,
		cgltf_size byte_length, cgltf_component_type ct, cgltf_type type,
		cgltf_size count, cgltf_size stride)
{
	if (m->view_count >= m->view_cap || m->accessor_count >= m->accessor_cap)
		die("internal error: output capacity exceeded");

	/* Grow bin with 4-byte alignment. */
	cgltf_size offset = align4(m->bin_size);
	cgltf_size new_size = offset + byte_length;
	m->bin = (cgltf_ubyte*)realloc(m->bin, new_size ? new_size : 1);
	if (!m->bin) die("out of memory growing output bin");
	memset(m->bin + m->bin_size, 0, offset - m->bin_size);
	memcpy(m->bin + offset, data, byte_length);
	m->bin_size = new_size;

	cgltf_buffer_view* view = &m->data->buffer_views[m->view_count];
	memset(view, 0, sizeof(*view));
	view->buffer = m->buffer; /* fixed up in out_model_finalize */
	view->offset = offset;
	view->size = byte_length;
	view->stride = stride;

	cgltf_accessor* acc = &m->data->accessors[m->accessor_count];
	memset(acc, 0, sizeof(*acc));
	acc->component_type = ct;
	acc->type = type;
	acc->count = count;
	acc->offset = 0;
	acc->buffer_view = view;

	++m->view_count;
	++m->accessor_count;
	m->data->buffer_views_count = m->view_count;
	m->data->accessors_count = m->accessor_count;
	return acc;
}

static cgltf_type type_for_components(cgltf_size comps)
{
	switch (comps)
	{
	case 1: return cgltf_type_scalar;
	case 2: return cgltf_type_vec2;
	case 3: return cgltf_type_vec3;
	case 4: return cgltf_type_vec4;
	default: die("unsupported component count %zu", comps);
	}
	return cgltf_type_scalar;
}

static void out_add_primitive(OutModel* m, cgltf_mesh* mesh, cgltf_primitive* src_prim,
		OptResult* res)
{
	/* mesh->primitives was pre-sized in the first pass. */
	cgltf_primitive* prim = &mesh->primitives[mesh->primitives_count];
	memset(prim, 0, sizeof(*prim));
	prim->type = cgltf_primitive_type_triangles;
	if (src_prim->material && m->data->materials_count)
	{
		/* Remap the source material pointer into the output array. */
		cgltf_size idx = (cgltf_size)(src_prim->material - m->src_materials);
		if (idx < m->data->materials_count)
			prim->material = &m->data->materials[idx];
	}
	++mesh->primitives_count;

	/* Index accessor: smallest legal unsigned integer type. */
	cgltf_uint32 max_index = 0;
	for (cgltf_size i = 0; i < res->tri_count * 3; ++i)
		if (res->out_indices[i] > max_index) max_index = res->out_indices[i];

	cgltf_accessor* idx_acc;
	if (max_index <= 0xFFFF)
	{
		cgltf_size n = res->tri_count * 3;
		cgltf_ushort* u16 = (cgltf_ushort*)xmalloc(n * sizeof(cgltf_ushort));
		for (cgltf_size i = 0; i < n; ++i) u16[i] = (cgltf_ushort)res->out_indices[i];
		idx_acc = out_add_accessor(m, (cgltf_ubyte*)u16,
				n * sizeof(cgltf_ushort), cgltf_component_type_r_16u,
				cgltf_type_scalar, n, 0);
		free(u16);
	}
	else
	{
		cgltf_size n = res->tri_count * 3;
		idx_acc = out_add_accessor(m, (cgltf_ubyte*)res->out_indices,
				n * sizeof(cgltf_uint32), cgltf_component_type_r_32u,
				cgltf_type_scalar, n, 0);
	}
	prim->indices = idx_acc;

	/* Split blobs back into regular attributes vs morph targets, keeping
	 * the source grouping. */
	cgltf_size reg_count = src_prim->attributes_count;
	cgltf_size target_blob_cursor = 0;
	cgltf_size reg_blob_cursor = 0;
	cgltf_size target_blob_total = 0;
	for (cgltf_size t = 0; t < src_prim->targets_count; ++t)
		target_blob_total += src_prim->targets[t].attributes_count;

	/* Regular attributes come after all target blobs in in_blobs/out_blobs. */
	reg_blob_cursor = target_blob_total;

	prim->attributes_count = reg_count;
	prim->attributes = (cgltf_attribute*)xcalloc(reg_count, sizeof(cgltf_attribute));
	for (cgltf_size a = 0; a < reg_count; ++a)
	{
		AttrBlob* blob = &res->out_blobs[reg_blob_cursor + a];
		cgltf_component_type ct = blob->is_integer
				? cgltf_component_type_r_32u : cgltf_component_type_r_32f;
		cgltf_accessor* acc = out_add_accessor(m, blob->bytes,
				res->out_vertex_count * blob->elem_bytes, ct,
				type_for_components(blob->components), res->out_vertex_count,
				blob->elem_bytes);
		prim->attributes[a].name = src_prim->attributes[a].name;
		prim->attributes[a].type = src_prim->attributes[a].type;
		prim->attributes[a].index = src_prim->attributes[a].index;
		prim->attributes[a].data = acc;
	}

	/* Morph targets: rebuild the same target grouping. */
	prim->targets_count = src_prim->targets_count;
	if (src_prim->targets_count)
	{
		prim->targets = (cgltf_morph_target*)xcalloc(src_prim->targets_count,
				sizeof(cgltf_morph_target));
		for (cgltf_size t = 0; t < src_prim->targets_count; ++t)
		{
			cgltf_size ac = src_prim->targets[t].attributes_count;
			prim->targets[t].attributes_count = ac;
			prim->targets[t].attributes = (cgltf_attribute*)xcalloc(ac,
					sizeof(cgltf_attribute));
			for (cgltf_size a = 0; a < ac; ++a)
			{
				AttrBlob* blob = &res->out_blobs[target_blob_cursor++];
				cgltf_accessor* acc = out_add_accessor(m, blob->bytes,
						res->out_vertex_count * blob->elem_bytes,
						cgltf_component_type_r_32f,
						type_for_components(blob->components),
						res->out_vertex_count, blob->elem_bytes);
				prim->targets[t].attributes[a].name =
						src_prim->targets[t].attributes[a].name;
				prim->targets[t].attributes[a].type =
						src_prim->targets[t].attributes[a].type;
				prim->targets[t].attributes[a].index =
						src_prim->targets[t].attributes[a].index;
				prim->targets[t].attributes[a].data = acc;
			}
		}
	}
}

static void out_model_finalize(OutModel* m, cgltf_mesh* meshes, cgltf_size mesh_count)
{
	m->data->meshes = meshes;
	m->data->meshes_count = mesh_count;

	m->data->buffers = (cgltf_buffer*)xcalloc(1, sizeof(cgltf_buffer));
	m->buffer = &m->data->buffers[0];
	m->buffer->size = m->bin_size;
	m->data->buffers_count = 1;
	for (cgltf_size i = 0; i < m->view_count; ++i)
		m->data->buffer_views[i].buffer = m->buffer;

	m->data->bin = m->bin;
	m->data->bin_size = m->bin_size;
}

/* ------------------------------------------------------------------ */
/* Bandwidth model                                                     */
/*                                                                     */
/* index bytes  = index count * index component size                   */
/* vertex bytes = unique vertices fetched (post-transform cache miss   */
/*                count) * sum of per-vertex attribute stride          */
/* Total = index + vertex bytes, per draw.                             */
/* ------------------------------------------------------------------ */

typedef struct Bandwidth
{
	double index_bytes;
	double vertex_bytes;
	double total;
} Bandwidth;

static cgltf_size vertex_stride(OptResult* res, cgltf_size blob_count, int use_out)
{
	cgltf_size s = 0;
	AttrBlob* blobs = use_out ? res->out_blobs : res->in_blobs;
	for (cgltf_size b = 0; b < blob_count; ++b)
		s += blobs[b].elem_bytes;
	return s;
}

static Bandwidth estimate_bandwidth(OptResult* res, cgltf_size idx_bytes_before,
		cgltf_size idx_bytes_after)
{
	cgltf_size stride = vertex_stride(res, res->blob_count, 1);
	Bandwidth bw;
	bw.index_bytes = (double)idx_bytes_after;
	bw.vertex_bytes = (double)res->fetched_after * (double)stride;
	bw.total = bw.index_bytes + bw.vertex_bytes;
	(void)idx_bytes_before;
	return bw;
}

/* ------------------------------------------------------------------ */
/* Timing helper                                                       */
/* ------------------------------------------------------------------ */

static double now_seconds(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* ------------------------------------------------------------------ */
/* JSON stats printing                                                 */
/* ------------------------------------------------------------------ */

static void print_stats_header(void)
{
	printf("{\n  \"primitives\": [\n");
}

static void print_primitive_stats(cgltf_size idx, const char* mesh_name,
		cgltf_size prim_idx, OptResult* res,
		cgltf_size idx_bytes_before, cgltf_size idx_bytes_after,
		cgltf_size stride_bytes, Bandwidth before, Bandwidth after, int last)
{
	double saved = before.total > 0 ? 100.0 * (before.total - after.total) / before.total : 0.0;
	printf("    {\n");
	printf("      \"mesh\": \"%s\",\n", mesh_name ? mesh_name : "");
	printf("      \"primitive\": %zu,\n", prim_idx);
	printf("      \"triangles\": %zu,\n", res->tri_count);
	printf("      \"vertices_before\": %zu,\n", res->in_vertex_count);
	printf("      \"vertices_after\": %zu,\n", res->out_vertex_count);
	printf("      \"indices_before\": %zu,\n",
			res->in_index_count ? res->in_index_count : res->tri_count * 3);
	printf("      \"indices_after\": %zu,\n", res->tri_count * 3);
	printf("      \"was_indexed\": %s,\n", res->in_index_count ? "true" : "false");
	printf("      \"triangles_reordered\": %zu,\n", res->triangles_reordered);
	printf("      \"triangle_order_change_pct\": %.2f,\n",
			res->tri_count ? 100.0 * (double)res->triangles_reordered / (double)res->tri_count : 0.0);
	printf("      \"acmr_before\": %.4f,\n", res->acmr_before);
	printf("      \"acmr_after\": %.4f,\n", res->acmr_after);
	printf("      \"vertex_fetches_before\": %zu,\n", res->fetched_before);
	printf("      \"vertex_fetches_after\": %zu,\n", res->fetched_after);
	printf("      \"index_bytes_before\": %zu,\n", idx_bytes_before);
	printf("      \"index_bytes_after\": %zu,\n", idx_bytes_after);
	printf("      \"vertex_stride_bytes\": %zu,\n", stride_bytes);
	printf("      \"bandwidth_before_bytes\": %.0f,\n", before.total);
	printf("      \"bandwidth_after_bytes\": %.0f,\n", after.total);
	printf("      \"bandwidth_saving_pct\": %.2f\n", saved);
	printf("    }%s\n", last ? "" : ",");
	(void)idx;
}

static void print_stats_footer(Bandwidth grand_before, Bandwidth grand_after,
		cgltf_size verts_before, cgltf_size verts_after,
		cgltf_size idx_before, cgltf_size idx_after)
{
	double saved = grand_before.total > 0
			? 100.0 * (grand_before.total - grand_after.total) / grand_before.total : 0.0;
	printf("  ],\n");
	printf("  \"totals\": {\n");
	printf("    \"vertices_before\": %zu,\n", verts_before);
	printf("    \"vertices_after\": %zu,\n", verts_after);
	printf("    \"indices_before\": %zu,\n", idx_before);
	printf("    \"indices_after\": %zu,\n", idx_after);
	printf("    \"bandwidth_before_bytes\": %.0f,\n", grand_before.total);
	printf("    \"bandwidth_after_bytes\": %.0f,\n", grand_after.total);
	printf("    \"bandwidth_saving_pct\": %.2f\n", saved);
	printf("  }\n");
	printf("}\n");
}

/* ------------------------------------------------------------------ */
/* Synthetic 100k-vertex benchmark                                     */
/* ------------------------------------------------------------------ */

static void run_benchmark(void)
{
	/* Build a synthetic indexed grid with >100k vertices and ~200k
	 * triangles; the corner list duplicates vertices to exercise dedup. */
	cgltf_size grid = 317; /* 317*317 = 100489 vertices */
	cgltf_size vcount = grid * grid;
	cgltf_size tris = (grid - 1) * (grid - 1) * 2;
	cgltf_size icount = tris * 3;

	/* Raw vertex table: position (3f) + normal (3f) + uv (2f) = 32 B. */
	cgltf_size stride = 8 * sizeof(cgltf_float);
	cgltf_float* verts = (cgltf_float*)xmalloc(vcount * 8 * sizeof(cgltf_float));
	for (cgltf_size y = 0; y < grid; ++y)
	{
		for (cgltf_size x = 0; x < grid; ++x)
		{
			cgltf_size idx = y * grid + x;
			if (idx >= vcount) die("bench vertex index out of range");
			cgltf_float* v = verts + idx * 8;
			v[0] = (cgltf_float)x; v[1] = (cgltf_float)y; v[2] = 0.0f;
			v[3] = 0.0f; v[4] = 0.0f; v[5] = 1.0f;
			v[6] = (cgltf_float)x / (cgltf_float)grid;
			v[7] = (cgltf_float)y / (cgltf_float)grid;
		}
	}

	/* Expand into a non-indexed corner list with 3x duplicated vertices to
	 * mimic a high-duplication mesh, then run the same pipeline pieces. */
	cgltf_size corner_verts = icount;
	cgltf_float* cverts = (cgltf_float*)xmalloc(corner_verts * 8 * sizeof(cgltf_float));
	cgltf_uint32* corners = (cgltf_uint32*)xmalloc(icount * sizeof(cgltf_uint32));
	cgltf_size w = 0;
	for (cgltf_size y = 0; y + 1 < grid; ++y)
	{
		for (cgltf_size x = 0; x + 1 < grid; ++x)
		{
			cgltf_size q[4] = {
				y * grid + x, y * grid + x + 1,
				(y + 1) * grid + x, (y + 1) * grid + x + 1
			};
			cgltf_size t0[3] = { q[0], q[2], q[1] };
			cgltf_size t1[3] = { q[1], q[2], q[3] };
			for (cgltf_size k = 0; k < 3; ++k)
			{
				memcpy(cverts + w * 8, verts + t0[k] * 8, stride);
				corners[w] = (cgltf_uint32)w;
				++w;
			}
			for (cgltf_size k = 0; k < 3; ++k)
			{
				memcpy(cverts + w * 8, verts + t1[k] * 8, stride);
				corners[w] = (cgltf_uint32)w;
				++w;
			}
		}
	}

	/* Time the full dedup + reorder + compact pipeline on raw corners. */
	const int runs = 5;
	double best = 1e9, total = 0.0;
	double tdedup = 0, tforsyth = 0, tcompact = 0;
	for (int run = 0; run < runs; ++run)
	{
		double t0 = now_seconds();

		cgltf_size table_cap = 16;
		while (table_cap < corner_verts * 2u) table_cap <<= 1;
		cgltf_size table_mask = table_cap - 1;
		cgltf_uint* table = (cgltf_uint*)xmalloc(table_cap * sizeof(cgltf_uint));
		for (cgltf_size i = 0; i < table_cap; ++i) table[i] = HASH_EMPTY;
		cgltf_uint32* dedup = (cgltf_uint32*)xmalloc(corner_verts * sizeof(cgltf_uint32));
		cgltf_uint32* usrc = (cgltf_uint32*)xmalloc(corner_verts * sizeof(cgltf_uint32));
		cgltf_size uc = 0;
		for (cgltf_size v = 0; v < corner_verts; ++v)
		{
			cgltf_uint h = hash_fast((cgltf_ubyte*)(cverts + v * 8), stride, 2166136261u);
			cgltf_size slot = h & table_mask;
			while (table[slot] != HASH_EMPTY)
			{
				cgltf_uint32 cs = usrc[table[slot]];
				if (memcmp(cverts + v * 8, cverts + cs * 8, stride) == 0)
				{
					dedup[v] = table[slot];
					goto bfound;
				}
				slot = (slot + 1) & table_mask;
			}
			dedup[v] = (cgltf_uint32)uc;
			usrc[uc++] = (cgltf_uint32)v;
			table[slot] = (cgltf_uint)(uc - 1);
bfound:;
		}
		free(table);
		cgltf_uint32* idx = (cgltf_uint32*)xmalloc(icount * sizeof(cgltf_uint32));
		for (cgltf_size i = 0; i < icount; ++i) idx[i] = dedup[corners[i]];
		free(dedup);

		double t1 = now_seconds();

		cgltf_uint32* order = (cgltf_uint32*)xmalloc(tris * sizeof(cgltf_uint32));
		forsyth_reorder(idx, icount, uc, order);
		free(order);

		double t2 = now_seconds();

		cgltf_uint32* compact = (cgltf_uint32*)xmalloc(uc * sizeof(cgltf_uint32));
		cgltf_ubyte* seen = (cgltf_ubyte*)xcalloc(uc, 1);
		cgltf_size nc = 0;
		for (cgltf_size i = 0; i < icount; ++i)
			if (!seen[idx[i]]) { seen[idx[i]] = 1; compact[idx[i]] = (cgltf_uint32)nc++; }
		free(seen);
		for (cgltf_size i = 0; i < icount; ++i) idx[i] = compact[idx[i]];
		free(compact);
		free(idx);
		free(usrc);

		double t3 = now_seconds();
		tdedup += t1 - t0;
		tforsyth += t2 - t1;
		tcompact += t3 - t2;

		double dt = t3 - t0;
		if (dt < best) best = dt;
		total += dt;
	}

	fprintf(stderr,
		"[bench] unique_vertices=%zu input_corners=%zu triangles=%zu runs=%d "
		"best=%.3fms avg=%.3fms\n"
		"[bench] dedup=%.3fms forsyth=%.3fms compact=%.3fms (avg per run)\n",
		vcount, corner_verts, tris, runs, best * 1000.0, total / runs * 1000.0,
		tdedup / runs * 1000.0, tforsyth / runs * 1000.0,
		tcompact / runs * 1000.0);

	free(corners);
	free(cverts);
	free(verts);
}

/* ------------------------------------------------------------------ */
/* main                                                                */
/* ------------------------------------------------------------------ */

int main(int argc, char** argv)
{
	if (argc < 3)
		die("usage: %s <input.gltf|input.glb> <output.glb> [--bench]", argv[0]);

	int bench = 0;
	for (int i = 3; i < argc; ++i)
		if (strcmp(argv[i], "--bench") == 0) bench = 1;
	if (bench) run_benchmark();

	cgltf_options options = {0};
	cgltf_data* data = NULL;
	cgltf_result result = cgltf_parse_file(&options, argv[1], &data);
	if (result != cgltf_result_success)
		die("failed to parse %s (code %d)", argv[1], (int)result);
	result = cgltf_load_buffers(&options, data, argv[1]);
	if (result != cgltf_result_success)
		die("failed to load buffers for %s (code %d)", argv[1], (int)result);

	OutModel* out = out_model_create();
	cgltf_size cap_meshes = data->meshes_count;
	cgltf_mesh* out_meshes = (cgltf_mesh*)xcalloc(cap_meshes ? cap_meshes : 1,
			sizeof(cgltf_mesh));

	/* First pass: pre-size every array whose base address the writer
	 * relies on for pointer-difference index computation. Per optimised
	 * primitive: 1 index accessor/view + one accessor/view per regular
	 * attribute + one per morph target attribute. */
	cgltf_size total_view_cap = 0;
	cgltf_size total_prim_cap = 0;
	for (cgltf_size i = 0; i < data->meshes_count; ++i)
	{
		cgltf_mesh* mesh = &data->meshes[i];
		cgltf_mesh* omesh = &out_meshes[i];
		omesh->name = mesh->name ? xstrdup(mesh->name) : NULL;
		omesh->weights = mesh->weights;
		omesh->weights_count = mesh->weights_count;
		omesh->target_names = mesh->target_names;
		omesh->target_names_count = mesh->target_names_count;
		omesh->primitives = (cgltf_primitive*)xcalloc(
				mesh->primitives_count ? mesh->primitives_count : 1,
				sizeof(cgltf_primitive));

		for (cgltf_size j = 0; j < mesh->primitives_count; ++j)
		{
			cgltf_primitive* prim = &mesh->primitives[j];
			if (prim->type != cgltf_primitive_type_triangles
					|| prim->has_draco_mesh_compression
					|| prim->attributes_count == 0)
				continue;
			++total_prim_cap;
			cgltf_size per = 1 + prim->attributes_count; /* index + attrs */
			for (cgltf_size t = 0; t < prim->targets_count; ++t)
				per += prim->targets[t].attributes_count;
			total_view_cap += per; /* one view per accessor */
		}
	}
	out_model_reserve(out, total_view_cap, total_view_cap);

	/* Carry the source materials across to the output document. A
	 * structure-level shallow copy is sufficient: nested pointers keep
	 * referencing the parsed input document, which stays alive until after
	 * the GLB is written. Primitive material pointers are then remapped
	 * into this array so the writer's index arithmetic stays valid. */
	if (data->materials_count)
	{
		out->src_materials = data->materials;
		out->data->materials = (cgltf_material*)xcalloc(
				data->materials_count, sizeof(cgltf_material));
		memcpy(out->data->materials, data->materials,
				data->materials_count * sizeof(cgltf_material));
		out->data->materials_count = data->materials_count;
	}

	Bandwidth grand_before = {0, 0, 0};
	Bandwidth grand_after = {0, 0, 0};
	cgltf_size grand_vb = 0, grand_va = 0, grand_ib = 0, grand_ia = 0;

	print_stats_header();

	cgltf_size printed = 0;

	for (cgltf_size i = 0; i < data->meshes_count; ++i)
	{
		cgltf_mesh* mesh = &data->meshes[i];
		cgltf_mesh* omesh = &out_meshes[i];

		for (cgltf_size j = 0; j < mesh->primitives_count; ++j)
		{
			cgltf_primitive* prim = &mesh->primitives[j];
			OptResult* res = optimize_primitive(prim);
			if (!res)
			{
				fprintf(stderr, "warning: skipping non-triangle/DRACO primitive "
						"%s[%zu]\n", mesh->name ? mesh->name : "", j);
				continue;
			}

			cgltf_size idx_count = res->tri_count * 3;
			cgltf_size idx_comp_after = res->out_vertex_count > 0xFFFF ? 4 : 2;
			cgltf_size idx_bytes_after = idx_count * idx_comp_after;
			cgltf_size idx_comp_before;
			if (!prim->indices)
				idx_comp_before = 0; /* non-indexed: no index stream */
			else
				idx_comp_before = cgltf_component_size(prim->indices->component_type);
			cgltf_size idx_bytes_before = prim->indices
					? prim->indices->count * idx_comp_before : 0;

			cgltf_size stride = vertex_stride(res, res->blob_count, 1);
			Bandwidth bw_after = estimate_bandwidth(res, idx_bytes_before, idx_bytes_after);
			Bandwidth bw_before;
			bw_before.index_bytes = (double)idx_bytes_before;
			bw_before.vertex_bytes = (double)res->fetched_before * (double)stride;
			bw_before.total = bw_before.index_bytes + bw_before.vertex_bytes;

			++printed;
			print_primitive_stats(i, mesh->name, j, res,
					idx_bytes_before, idx_bytes_after, stride,
					bw_before, bw_after, printed == total_prim_cap);
			out_add_primitive(out, omesh, prim, res);

			grand_before.index_bytes += bw_before.index_bytes;
			grand_before.vertex_bytes += bw_before.vertex_bytes;
			grand_before.total += bw_before.total;
			grand_after.index_bytes += bw_after.index_bytes;
			grand_after.vertex_bytes += bw_after.vertex_bytes;
			grand_after.total += bw_after.total;
			grand_vb += res->in_vertex_count;
			grand_va += res->out_vertex_count;
			grand_ib += prim->indices ? prim->indices->count : res->tri_count * 3;
			grand_ia += idx_count;

			free_opt(res);
		}
	}

	print_stats_footer(grand_before, grand_after, grand_vb, grand_va, grand_ib, grand_ia);

	out_model_finalize(out, out_meshes, data->meshes_count);
	cgltf_options write_opts = {0};
	write_opts.type = cgltf_file_type_glb;
	result = cgltf_write_file(&write_opts, argv[2], out->data);
	if (result != cgltf_result_success)
		die("failed to write %s (code %d)", argv[2], (int)result);

	cgltf_free(data);
	return 0;
}
