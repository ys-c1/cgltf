/*
 * test_prim_group.c — primitive grouping for cgltf-loaded meshes.
 *
 * Merges each mesh's primitives by material (reordering indices and
 * concatenating vertex attributes / morph targets in their original order),
 * and can split the merged primitives back to the original grouping.
 *
 * Verification per model:
 *   - primitive / total-index / total-vertex counts before and after merge
 *   - merge -> split restores the original data byte for byte
 *   - draw-call (primitive) reduction and merge wall time
 *
 * Build (from repo root):  gcc -O2 -I. test/test_prim_group.c -o <out>
 * Run:                     <out> <dir-with-gltf-samples>
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#define CGLTF_IMPLEMENTATION
#include "cgltf.h"

/* ------------------------------------------------------------------ */
/* tool context                                                        */

typedef struct {
	cgltf_size vcount, icount;
	cgltf_size orig;           /* index of the source primitive in the original mesh */
} prim_slice;

typedef struct {
	cgltf_mesh* mesh;
	cgltf_primitive* merged;   /* points into mesh->primitives */
	prim_slice* slices;        /* original per-primitive vertex/index counts */
	cgltf_size slice_count;
} merge_record;

typedef struct {
	cgltf_data* data;

	void** ptrs;               /* every tool-owned allocation, freed at exit */
	size_t ptrs_n, ptrs_cap;

	cgltf_accessor* acc_blk;   /* chunked pools (pointers must stay stable) */
	size_t acc_left;
	cgltf_buffer_view* view_blk;
	size_t view_left;

	char* arena;               /* merged/split vertex+index data */
	size_t arena_used, arena_cap;
	cgltf_buffer arena_buf;    /* tool-owned buffer object for the arena */

	merge_record* recs;
	size_t recs_n, recs_cap;
} grouper;

static void* track(grouper* g, void* p) {
	if (g->ptrs_n == g->ptrs_cap) {
		g->ptrs_cap = g->ptrs_cap ? g->ptrs_cap * 2 : 64;
		g->ptrs = (void**)realloc(g->ptrs, g->ptrs_cap * sizeof(void*));
	}
	g->ptrs[g->ptrs_n++] = p;
	return p;
}

static void* galloc(grouper* g, size_t n) { return track(g, calloc(1, n)); }

static cgltf_accessor* new_accessor(grouper* g) {
	if (!g->acc_left) {
		g->acc_blk = (cgltf_accessor*)galloc(g, 128 * sizeof(cgltf_accessor));
		g->acc_left = 128;
	}
	g->acc_left--;
	return g->acc_blk++;
}

static cgltf_buffer_view* new_view(grouper* g) {
	if (!g->view_left) {
		g->view_blk = (cgltf_buffer_view*)galloc(g, 128 * sizeof(cgltf_buffer_view));
		g->view_left = 128;
	}
	g->view_left--;
	return g->view_blk++;
}

static size_t arena_alloc(grouper* g, size_t n) {
	size_t off = (g->arena_used + 3) & ~(size_t)3;
	if (off + n > g->arena_cap) {
		g->arena_cap = (off + n) * 2 + 1024;
		g->arena = (char*)realloc(g->arena, g->arena_cap);
	}
	g->arena_used = off + n;
	g->arena_buf.data = g->arena;
	g->arena_buf.size = g->arena_used;
	return off;
}

static void* mem_alloc(grouper* g, size_t n) {
	return g->data->memory.alloc_func(g->data->memory.user_data, n);
}
static void mem_free(grouper* g, void* p) {
	if (p) g->data->memory.free_func(g->data->memory.user_data, p);
}

static char* str_dup(grouper* g, const char* s) {
	size_t n = strlen(s) + 1;
	char* p = (char*)mem_alloc(g, n);
	memcpy(p, s, n);
	return p;
}

/* ------------------------------------------------------------------ */
/* accessor helpers                                                    */

static const unsigned char* acc_base(const cgltf_accessor* acc, size_t* stride, size_t* esize) {
	*esize = cgltf_calc_size(acc->type, acc->component_type);
	*stride = acc->stride ? acc->stride : *esize;
	return cgltf_buffer_view_data(acc->buffer_view) + acc->offset;
}

/* copy `count` elements of `acc` into tightly packed dst */
static void copy_acc_elements(const cgltf_accessor* acc, unsigned char* dst, cgltf_size count) {
	size_t stride, esize;
	const unsigned char* base = acc_base(acc, &stride, &esize);
	for (cgltf_size i = 0; i < count; ++i)
		memcpy(dst + i * esize, base + i * stride, esize);
}

static cgltf_accessor* emit_accessor(grouper* g, const cgltf_accessor* proto,
                                     cgltf_size count, size_t arena_off) {
	cgltf_buffer_view* v = new_view(g);
	v->buffer = &g->arena_buf;
	v->offset = arena_off;
	v->size = count * cgltf_calc_size(proto->type, proto->component_type);
	cgltf_accessor* a = new_accessor(g);
	a->component_type = proto->component_type;
	a->type = proto->type;
	a->normalized = proto->normalized;
	a->count = count;
	a->stride = cgltf_calc_size(proto->type, proto->component_type);
	a->buffer_view = v;
	return a;
}

/* ------------------------------------------------------------------ */
/* merge                                                               */

static int accessor_usable(const cgltf_accessor* a) {
	return a && !a->is_sparse && a->buffer_view && cgltf_buffer_view_data(a->buffer_view);
}

static int prim_mergeable(const cgltf_primitive* p) {
	if (p->type != cgltf_primitive_type_triangles) return 0;
	if (p->has_draco_mesh_compression) return 0;
	if (!accessor_usable(p->indices)) return 0;
	if (p->attributes_count == 0) return 0;
	for (cgltf_size j = 0; j < p->attributes_count; ++j)
		if (!accessor_usable(p->attributes[j].data)) return 0;
	for (cgltf_size t = 0; t < p->targets_count; ++t)
		for (cgltf_size j = 0; j < p->targets[t].attributes_count; ++j)
			if (!accessor_usable(p->targets[t].attributes[j].data)) return 0;
	return 1;
}

static int attr_sig_equal(const cgltf_attribute* a, const cgltf_attribute* b) {
	if (a->type != b->type || a->index != b->index) return 0;
	if (a->type == cgltf_attribute_type_custom &&
	    strcmp(a->name ? a->name : "", b->name ? b->name : "") != 0) return 0;
	return a->data->component_type == b->data->component_type &&
	       a->data->type == b->data->type &&
	       a->data->normalized == b->data->normalized;
}

static int prim_compatible(const cgltf_primitive* a, const cgltf_primitive* b) {
	if (a->attributes_count != b->attributes_count) return 0;
	if (a->targets_count != b->targets_count) return 0;
	if (a->indices->component_type != b->indices->component_type) return 0;
	for (cgltf_size j = 0; j < a->attributes_count; ++j)
		if (!attr_sig_equal(&a->attributes[j], &b->attributes[j])) return 0;
	for (cgltf_size t = 0; t < a->targets_count; ++t) {
		if (a->targets[t].attributes_count != b->targets[t].attributes_count) return 0;
		for (cgltf_size j = 0; j < a->targets[t].attributes_count; ++j)
			if (!attr_sig_equal(&a->targets[t].attributes[j], &b->targets[t].attributes[j])) return 0;
	}
	return 1;
}

static void write_index(unsigned char* dst, size_t isz, cgltf_size v) {
	uint32_t v32 = (uint32_t)v;
	memcpy(dst, &v32, isz); /* little-endian, component size 1/2/4 */
}

/* free the per-primitive sub-allocations exactly like cgltf_free does */
static void free_prim_contents(grouper* g, cgltf_primitive* p) {
	cgltf_data* data = g->data;
	for (cgltf_size k = 0; k < p->attributes_count; ++k)
		mem_free(g, p->attributes[k].name);
	mem_free(g, p->attributes);
	for (cgltf_size k = 0; k < p->targets_count; ++k) {
		for (cgltf_size m = 0; m < p->targets[k].attributes_count; ++m)
			mem_free(g, p->targets[k].attributes[m].name);
		mem_free(g, p->targets[k].attributes);
	}
	mem_free(g, p->targets);
	if (p->has_draco_mesh_compression) {
		for (cgltf_size k = 0; k < p->draco_mesh_compression.attributes_count; ++k)
			mem_free(g, p->draco_mesh_compression.attributes[k].name);
		mem_free(g, p->draco_mesh_compression.attributes);
	}
	for (cgltf_size k = 0; k < p->mappings_count; ++k)
		cgltf_free_extras(data, &p->mappings[k].extras);
	mem_free(g, p->mappings);
	cgltf_free_extensions(data, p->extensions, p->extensions_count);
	cgltf_free_extras(data, &p->extras);
}

typedef struct {
	cgltf_size* members;
	cgltf_size count, cap;
} prim_group;

static cgltf_primitive build_merged(grouper* g, const cgltf_primitive* prims,
                                    const prim_group* grp, merge_record* rec) {
	const cgltf_primitive* rep = &prims[grp->members[0]];
	cgltf_size nattr = rep->attributes_count, ntgt = rep->targets_count;
	cgltf_size total_v = 0, total_i = 0;
	for (cgltf_size s = 0; s < grp->count; ++s) {
		const cgltf_primitive* p = &prims[grp->members[s]];
		total_v += p->attributes[0].data->count;
		total_i += p->indices->count;
	}

	/* arena layout: indices, then attributes, then morph targets */
	size_t isz = cgltf_component_size(rep->indices->component_type);
	size_t ioff = arena_alloc(g, total_i * isz);
	size_t* aoff = (size_t*)galloc(g, nattr * sizeof(size_t));
	for (cgltf_size j = 0; j < nattr; ++j)
		aoff[j] = arena_alloc(g, total_v * cgltf_calc_size(rep->attributes[j].data->type,
		                                                   rep->attributes[j].data->component_type));
	size_t* toff = (size_t*)galloc(g, ntgt * nattr * sizeof(size_t));
	for (cgltf_size t = 0; t < ntgt; ++t)
		for (cgltf_size j = 0; j < rep->targets[t].attributes_count; ++j) {
			const cgltf_accessor* pa = rep->targets[t].attributes[j].data;
			toff[t * nattr + j] = arena_alloc(g, total_v * cgltf_calc_size(pa->type, pa->component_type));
		}

	rec->slices = (prim_slice*)galloc(g, grp->count * sizeof(prim_slice));
	rec->slice_count = grp->count;

	cgltf_size vb = 0, ib = 0;
	for (cgltf_size s = 0; s < grp->count; ++s) {
		const cgltf_primitive* p = &prims[grp->members[s]];
		cgltf_size vc = p->attributes[0].data->count, ic = p->indices->count;
		for (cgltf_size k = 0; k < ic; ++k) /* re-base indices by vertex offset */
			write_index((unsigned char*)g->arena + ioff + (ib + k) * isz, isz,
			            cgltf_accessor_read_index(p->indices, k) + vb);
		for (cgltf_size j = 0; j < nattr; ++j) {
			size_t esz = cgltf_calc_size(p->attributes[j].data->type, p->attributes[j].data->component_type);
			copy_acc_elements(p->attributes[j].data, (unsigned char*)g->arena + aoff[j] + vb * esz, vc);
		}
		for (cgltf_size t = 0; t < ntgt; ++t)
			for (cgltf_size j = 0; j < p->targets[t].attributes_count; ++j) {
				const cgltf_accessor* pa = p->targets[t].attributes[j].data;
				size_t esz = cgltf_calc_size(pa->type, pa->component_type);
				copy_acc_elements(pa, (unsigned char*)g->arena + toff[t * nattr + j] + vb * esz, vc);
			}
		rec->slices[s].vcount = vc;
		rec->slices[s].icount = ic;
		rec->slices[s].orig = grp->members[s];
		vb += vc;
		ib += ic;
	}

	cgltf_primitive mp;
	memset(&mp, 0, sizeof(mp));
	mp.type = rep->type;
	mp.material = rep->material;
	mp.indices = emit_accessor(g, rep->indices, total_i, ioff);
	mp.attributes_count = nattr;
	mp.attributes = (cgltf_attribute*)mem_alloc(g, nattr * sizeof(cgltf_attribute));
	memset(mp.attributes, 0, nattr * sizeof(cgltf_attribute));
	for (cgltf_size j = 0; j < nattr; ++j) {
		mp.attributes[j].name = str_dup(g, rep->attributes[j].name);
		mp.attributes[j].type = rep->attributes[j].type;
		mp.attributes[j].index = rep->attributes[j].index;
		mp.attributes[j].data = emit_accessor(g, rep->attributes[j].data, total_v, aoff[j]);
	}
	mp.targets_count = ntgt;
	if (ntgt) {
		mp.targets = (cgltf_morph_target*)mem_alloc(g, ntgt * sizeof(cgltf_morph_target));
		memset(mp.targets, 0, ntgt * sizeof(cgltf_morph_target));
		for (cgltf_size t = 0; t < ntgt; ++t) {
			cgltf_size tc = rep->targets[t].attributes_count;
			mp.targets[t].attributes_count = tc;
			mp.targets[t].attributes = (cgltf_attribute*)mem_alloc(g, tc * sizeof(cgltf_attribute));
			memset(mp.targets[t].attributes, 0, tc * sizeof(cgltf_attribute));
			for (cgltf_size j = 0; j < tc; ++j) {
				mp.targets[t].attributes[j].name = str_dup(g, rep->targets[t].attributes[j].name);
				mp.targets[t].attributes[j].type = rep->targets[t].attributes[j].type;
				mp.targets[t].attributes[j].index = rep->targets[t].attributes[j].index;
				mp.targets[t].attributes[j].data =
					emit_accessor(g, rep->targets[t].attributes[j].data, total_v, toff[t * nattr + j]);
			}
		}
	}
	return mp;
}

static void merge_mesh(grouper* g, cgltf_mesh* mesh) {
	cgltf_size n = mesh->primitives_count;
	cgltf_primitive* prims = mesh->primitives;

	/* group primitives by (material, attribute signature), order preserved */
	prim_group* groups = (prim_group*)calloc(n, sizeof(prim_group));
	cgltf_material** gmats = (cgltf_material**)calloc(n, sizeof(cgltf_material*));
	int* gid = (int*)malloc(n * sizeof(int));
	cgltf_size ngroups = 0;
	for (cgltf_size i = 0; i < n; ++i) {
		gid[i] = -1;
		if (!prim_mergeable(&prims[i])) continue;
		for (cgltf_size gi = 0; gi < ngroups; ++gi) {
			if (gmats[gi] == prims[i].material &&
			    prim_compatible(&prims[groups[gi].members[0]], &prims[i])) {
				prim_group* gr = &groups[gi];
				if (gr->count == gr->cap) {
					gr->cap = gr->cap ? gr->cap * 2 : 4;
					gr->members = (cgltf_size*)realloc(gr->members, gr->cap * sizeof(cgltf_size));
				}
				gr->members[gr->count++] = i;
				gid[i] = (int)gi;
				break;
			}
		}
		if (gid[i] < 0) {
			prim_group* gr = &groups[ngroups];
			gr->cap = 4;
			gr->members = (cgltf_size*)malloc(gr->cap * sizeof(cgltf_size));
			gr->members[0] = i;
			gr->count = 1;
			gmats[ngroups] = prims[i].material;
			gid[i] = (int)ngroups;
			ngroups++;
		}
	}

	/* groups with a single member are left untouched */
	cgltf_primitive* merged_prims = (cgltf_primitive*)calloc(ngroups ? ngroups : 1, sizeof(cgltf_primitive));
	int* group_merged = (int*)calloc(ngroups ? ngroups : 1, sizeof(int));
	for (cgltf_size gi = 0; gi < ngroups; ++gi) {
		if (groups[gi].count < 2) continue;
		if (g->recs_n == g->recs_cap) {
			g->recs_cap = g->recs_cap ? g->recs_cap * 2 : 16;
			g->recs = (merge_record*)realloc(g->recs, g->recs_cap * sizeof(merge_record));
		}
		merge_record* rec = &g->recs[g->recs_n++];
		memset(rec, 0, sizeof(*rec));
		rec->mesh = mesh;
		merged_prims[gi] = build_merged(g, prims, &groups[gi], rec);
		group_merged[gi] = 1;
	}

	/* new primitive list: merged groups at their first member's position */
	cgltf_size newn = 0;
	for (cgltf_size i = 0; i < n; ++i) {
		int gi = gid[i];
		if (gi >= 0 && group_merged[gi]) {
			if (groups[gi].members[0] == i) newn++;
		} else {
			newn++;
		}
	}
	cgltf_primitive* newp = (cgltf_primitive*)mem_alloc(g, newn * sizeof(cgltf_primitive));
	memset(newp, 0, newn * sizeof(cgltf_primitive));
	cgltf_size m = 0, rec_idx = g->recs_n;
	for (cgltf_size gi = 0; gi < ngroups; ++gi)
		if (group_merged[gi]) rec_idx--; /* records were appended in group order */
	for (cgltf_size i = 0; i < n; ++i) {
		int gi = gid[i];
		if (gi >= 0 && group_merged[gi]) {
			if (groups[gi].members[0] == i) {
				newp[m] = merged_prims[gi];
				g->recs[rec_idx++].merged = &newp[m]; /* fix up after array is final */
				m++;
			}
			free_prim_contents(g, &prims[i]);
		} else {
			newp[m++] = prims[i];
		}
	}
	mem_free(g, prims);
	mesh->primitives = newp;
	mesh->primitives_count = newn;

	for (cgltf_size gi = 0; gi < ngroups; ++gi) free(groups[gi].members);
	free(groups);
	free(gmats);
	free(gid);
	free(merged_prims);
	free(group_merged);
}

/* ------------------------------------------------------------------ */
/* split (reverse of merge)                                            */

/* carve one slice (vertex/index data + accessors) out of a merged primitive */
static void build_slice(grouper* g, const cgltf_primitive* mp, const prim_slice* slice,
                        cgltf_size vb, cgltf_size ib, cgltf_primitive* sp) {
	cgltf_size vc = slice->vcount, ic = slice->icount;
	cgltf_size nattr = mp->attributes_count, ntgt = mp->targets_count;
	size_t isz = cgltf_component_size(mp->indices->component_type);

	size_t ioff = arena_alloc(g, ic * isz);
	size_t* aoff = (size_t*)galloc(g, nattr * sizeof(size_t));
	for (cgltf_size j = 0; j < nattr; ++j)
		aoff[j] = arena_alloc(g, vc * cgltf_calc_size(mp->attributes[j].data->type,
		                                              mp->attributes[j].data->component_type));
	size_t* toff = (size_t*)galloc(g, (ntgt ? ntgt : 1) * nattr * sizeof(size_t));
	for (cgltf_size t = 0; t < ntgt; ++t)
		for (cgltf_size j = 0; j < mp->targets[t].attributes_count; ++j) {
			const cgltf_accessor* pa = mp->targets[t].attributes[j].data;
			toff[t * nattr + j] = arena_alloc(g, vc * cgltf_calc_size(pa->type, pa->component_type));
		}

	/* arena base is re-read after the allocations above (they may realloc) */
	size_t st, esz;
	/* undo the index re-base: merged indices were offset by the vertex base */
	for (cgltf_size k = 0; k < ic; ++k)
		write_index((unsigned char*)g->arena + ioff + k * isz, isz,
		            cgltf_accessor_read_index(mp->indices, ib + k) - vb);

	sp->type = mp->type;
	sp->material = mp->material;
	sp->indices = emit_accessor(g, mp->indices, ic, ioff);
	sp->attributes_count = nattr;
	sp->attributes = (cgltf_attribute*)mem_alloc(g, nattr * sizeof(cgltf_attribute));
	memset(sp->attributes, 0, nattr * sizeof(cgltf_attribute));
	for (cgltf_size j = 0; j < nattr; ++j) {
		const cgltf_accessor* ma = mp->attributes[j].data;
		size_t es = cgltf_calc_size(ma->type, ma->component_type);
		memcpy(g->arena + aoff[j], acc_base(ma, &st, &esz) + vb * es, vc * es);
		sp->attributes[j].name = str_dup(g, mp->attributes[j].name);
		sp->attributes[j].type = mp->attributes[j].type;
		sp->attributes[j].index = mp->attributes[j].index;
		sp->attributes[j].data = emit_accessor(g, ma, vc, aoff[j]);
	}
	sp->targets_count = ntgt;
	if (ntgt) {
		sp->targets = (cgltf_morph_target*)mem_alloc(g, ntgt * sizeof(cgltf_morph_target));
		memset(sp->targets, 0, ntgt * sizeof(cgltf_morph_target));
		for (cgltf_size t = 0; t < ntgt; ++t) {
			cgltf_size tc = mp->targets[t].attributes_count;
			sp->targets[t].attributes_count = tc;
			sp->targets[t].attributes = (cgltf_attribute*)mem_alloc(g, tc * sizeof(cgltf_attribute));
			memset(sp->targets[t].attributes, 0, tc * sizeof(cgltf_attribute));
			for (cgltf_size j = 0; j < tc; ++j) {
				const cgltf_accessor* ma = mp->targets[t].attributes[j].data;
				size_t es = cgltf_calc_size(ma->type, ma->component_type);
				memcpy(g->arena + toff[t * nattr + j], acc_base(ma, &st, &esz) + vb * es, vc * es);
				sp->targets[t].attributes[j].name = str_dup(g, mp->targets[t].attributes[j].name);
				sp->targets[t].attributes[j].type = mp->targets[t].attributes[j].type;
				sp->targets[t].attributes[j].index = mp->targets[t].attributes[j].index;
				sp->targets[t].attributes[j].data = emit_accessor(g, ma, vc, toff[t * nattr + j]);
			}
		}
	}
}

/* restore a mesh's original primitive grouping from the merge records */
static void split_mesh(grouper* g, cgltf_mesh* mesh, merge_record** recs, size_t nrecs) {
	cgltf_size oldn = mesh->primitives_count;
	cgltf_size total_slices = 0;
	for (size_t r = 0; r < nrecs; ++r) total_slices += recs[r]->slice_count;
	cgltf_size newn = oldn - nrecs + total_slices;

	cgltf_primitive* newp = (cgltf_primitive*)mem_alloc(g, newn * sizeof(cgltf_primitive));
	memset(newp, 0, newn * sizeof(cgltf_primitive));

	/* original positions covered by slices of merged groups */
	unsigned char* covered = (unsigned char*)calloc(newn ? newn : 1, 1);
	for (size_t r = 0; r < nrecs; ++r)
		for (cgltf_size s = 0; s < recs[r]->slice_count; ++s)
			covered[recs[r]->slices[s].orig] = 1;

	cgltf_size next_orig = 0;
	for (cgltf_size i = 0; i < oldn; ++i) {
		cgltf_primitive* cur = &mesh->primitives[i];
		merge_record* rec = NULL;
		for (size_t r = 0; r < nrecs; ++r)
			if (recs[r]->merged == cur) { rec = recs[r]; break; }
		if (rec) {
			cgltf_size vb = 0, ib = 0;
			for (cgltf_size s = 0; s < rec->slice_count; ++s) {
				build_slice(g, cur, &rec->slices[s], vb, ib, &newp[rec->slices[s].orig]);
				vb += rec->slices[s].vcount;
				ib += rec->slices[s].icount;
			}
			free_prim_contents(g, cur);
		} else {
			while (next_orig < newn && covered[next_orig]) next_orig++;
			if (next_orig == newn) {
				fprintf(stderr, "split: inconsistent record layout\n");
				exit(1);
			}
			newp[next_orig++] = *cur;
		}
	}
	free(covered);
	mem_free(g, mesh->primitives);
	mesh->primitives = newp;
	mesh->primitives_count = newn;
}

/* ------------------------------------------------------------------ */
/* snapshot / byte-exact verification                                  */

typedef struct {
	unsigned char* p;
	size_t n, cap;
} blob;

static void blob_put(blob* b, const void* src, size_t n) {
	if (b->n + n > b->cap) {
		b->cap = (b->n + n) * 2 + 64;
		b->p = (unsigned char*)realloc(b->p, b->cap);
	}
	memcpy(b->p + b->n, src, n);
	b->n += n;
}
static void blob_u32(blob* b, uint32_t v) { blob_put(b, &v, sizeof(v)); }
static void blob_u64(blob* b, uint64_t v) { blob_put(b, &v, sizeof(v)); }

static void blob_accessor(blob* b, const cgltf_accessor* a) {
	blob_u32(b, (uint32_t)a->component_type);
	blob_u32(b, (uint32_t)a->type);
	blob_u32(b, (uint32_t)a->normalized);
	blob_u64(b, a->count);
	size_t stride, esize;
	const unsigned char* base = acc_base(a, &stride, &esize);
	for (cgltf_size i = 0; i < a->count; ++i)
		blob_put(b, base + i * stride, esize);
}

static void snapshot_prim(blob* b, const cgltf_primitive* p) {
	blob_u32(b, (uint32_t)p->type);
	if (p->indices) {
		blob_u32(b, 1);
		blob_accessor(b, p->indices);
	} else {
		blob_u32(b, 0);
	}
	blob_u32(b, (uint32_t)p->attributes_count);
	for (cgltf_size j = 0; j < p->attributes_count; ++j) {
		blob_u32(b, (uint32_t)p->attributes[j].type);
		blob_u32(b, (uint32_t)p->attributes[j].index);
		blob_accessor(b, p->attributes[j].data);
	}
	blob_u32(b, (uint32_t)p->targets_count);
	for (cgltf_size t = 0; t < p->targets_count; ++t) {
		blob_u32(b, (uint32_t)p->targets[t].attributes_count);
		for (cgltf_size j = 0; j < p->targets[t].attributes_count; ++j) {
			blob_u32(b, (uint32_t)p->targets[t].attributes[j].type);
			blob_u32(b, (uint32_t)p->targets[t].attributes[j].index);
			blob_accessor(b, p->targets[t].attributes[j].data);
		}
	}
}

typedef struct {
	cgltf_size mesh_count;
	cgltf_size* prim_counts;     /* per mesh */
	cgltf_material*** mats;      /* per mesh per prim */
	blob** prims;                /* per mesh per prim */
} snapshot;

static snapshot take_snapshot(cgltf_data* d) {
	snapshot s;
	memset(&s, 0, sizeof(s));
	s.mesh_count = d->meshes_count;
	s.prim_counts = (cgltf_size*)calloc(s.mesh_count, sizeof(cgltf_size));
	s.mats = (cgltf_material***)calloc(s.mesh_count, sizeof(cgltf_material**));
	s.prims = (blob**)calloc(s.mesh_count, sizeof(blob*));
	for (cgltf_size i = 0; i < d->meshes_count; ++i) {
		cgltf_size n = d->meshes[i].primitives_count;
		s.prim_counts[i] = n;
		s.mats[i] = (cgltf_material**)calloc(n, sizeof(cgltf_material*));
		s.prims[i] = (blob*)calloc(n, sizeof(blob));
		for (cgltf_size j = 0; j < n; ++j) {
			s.mats[i][j] = d->meshes[i].primitives[j].material;
			snapshot_prim(&s.prims[i][j], &d->meshes[i].primitives[j]);
		}
	}
	return s;
}

static void free_snapshot(snapshot* s) {
	for (cgltf_size i = 0; i < s->mesh_count; ++i) {
		for (cgltf_size j = 0; j < s->prim_counts[i]; ++j) free(s->prims[i][j].p);
		free(s->prims[i]);
		free(s->mats[i]);
	}
	free(s->prim_counts);
	free(s->mats);
	free(s->prims);
}

/* compare current data against snapshot; returns number of mismatches */
static int verify_against(cgltf_data* d, const snapshot* s, const char* tag) {
	int fails = 0;
	if (d->meshes_count != s->mesh_count) {
		printf("  [%s] FAIL: mesh count changed\n", tag);
		return 1;
	}
	for (cgltf_size i = 0; i < d->meshes_count; ++i) {
		cgltf_mesh* m = &d->meshes[i];
		if (m->primitives_count != s->prim_counts[i]) {
			printf("  [%s] FAIL: mesh %zu primitive count %zu != %zu\n", tag, i,
			       (size_t)m->primitives_count, (size_t)s->prim_counts[i]);
			fails++;
			continue;
		}
		for (cgltf_size j = 0; j < m->primitives_count; ++j) {
			if (m->primitives[j].material != s->mats[i][j]) {
				printf("  [%s] FAIL: mesh %zu prim %zu material mismatch\n", tag, i, j);
				fails++;
			}
			blob cur;
			memset(&cur, 0, sizeof(cur));
			snapshot_prim(&cur, &m->primitives[j]);
			if (cur.n != s->prims[i][j].n || memcmp(cur.p, s->prims[i][j].p, cur.n) != 0) {
				size_t fn = cur.n < s->prims[i][j].n ? cur.n : s->prims[i][j].n;
				size_t diff = 0;
				while (diff < fn && cur.p[diff] == s->prims[i][j].p[diff]) diff++;
				printf("  [%s] FAIL: mesh %zu prim %zu data differs (len %zu vs %zu, first diff @%zu)\n",
				       tag, i, j, cur.n, s->prims[i][j].n, diff);
				fails++;
			}
			free(cur.p);
		}
	}
	return fails;
}

/* ------------------------------------------------------------------ */

typedef struct {
	cgltf_size prims, indices, vertices;
} stats;

static stats compute_stats(cgltf_data* d) {
	stats st;
	memset(&st, 0, sizeof(st));
	for (cgltf_size i = 0; i < d->meshes_count; ++i) {
		cgltf_mesh* m = &d->meshes[i];
		st.prims += m->primitives_count;
		for (cgltf_size j = 0; j < m->primitives_count; ++j) {
			cgltf_primitive* p = &m->primitives[j];
			if (p->indices) st.indices += p->indices->count;
			if (p->attributes_count) st.vertices += p->attributes[0].data->count;
		}
	}
	return st;
}

static double now_ms(void) {
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

int main(int argc, char** argv) {
	const char* dir = argc > 1 ? argv[1] : "artifacts/primgroup";
	struct { const char* name; cgltf_size expect_prims; } models[] = {
		{"single_material", 1},
		{"multi_material", 4},
		{"shared_indices", 2},
		{"morph", 2},
	};
	int total_fails = 0;
	cgltf_size draws_before = 0, draws_after = 0;

	printf("%-16s %6s %6s | %9s %9s | %9s %9s | %8s | %s\n",
	       "model", "prims", "merged", "indices", "idx(aft)", "verts", "vert(aft)", "merge ms", "split");
	for (size_t mi = 0; mi < sizeof(models) / sizeof(models[0]); ++mi) {
		char path[512];
		snprintf(path, sizeof(path), "%s/%s.gltf", dir, models[mi].name);

		cgltf_options options;
		memset(&options, 0, sizeof(options));
		cgltf_data* data = NULL;
		cgltf_result r = cgltf_parse_file(&options, path, &data);
		if (r == cgltf_result_success) r = cgltf_load_buffers(&options, data, path);
		if (r != cgltf_result_success) {
			printf("%-16s LOAD FAILED (%d)\n", models[mi].name, r);
			total_fails++;
			continue;
		}

		snapshot before = take_snapshot(data);
		stats s0 = compute_stats(data);

		grouper g;
		memset(&g, 0, sizeof(g));
		g.data = data;

		double t0 = now_ms();
		for (cgltf_size i = 0; i < data->meshes_count; ++i)
			merge_mesh(&g, &data->meshes[i]);
		double merge_ms = now_ms() - t0;

		stats s1 = compute_stats(data);

		int fails = 0;
		if (s1.prims != models[mi].expect_prims) {
			printf("  FAIL: expected %zu merged primitives, got %zu\n",
			       (size_t)models[mi].expect_prims, (size_t)s1.prims);
			fails++;
		}
		if (s1.indices != s0.indices || s1.vertices != s0.vertices) {
			printf("  FAIL: index/vertex totals changed by merge\n");
			fails++;
		}

		/* split back and verify byte-exact restore */
		for (cgltf_size i = 0; i < data->meshes_count; ++i) {
			merge_record** mrecs = (merge_record**)malloc((g.recs_n ? g.recs_n : 1) * sizeof(merge_record*));
			size_t nm = 0;
			for (size_t ri = 0; ri < g.recs_n; ++ri)
				if (g.recs[ri].mesh == &data->meshes[i]) mrecs[nm++] = &g.recs[ri];
			if (nm) split_mesh(&g, &data->meshes[i], mrecs, nm);
			free(mrecs);
		}
		fails += verify_against(data, &before, "split");

		stats s2 = compute_stats(data);
		if (s2.indices != s0.indices || s2.vertices != s0.vertices || s2.prims != s0.prims) {
			printf("  FAIL: totals changed after split\n");
			fails++;
		}
		if (merge_ms > 100.0) {
			printf("  FAIL: merge took %.3f ms (> 100 ms)\n", merge_ms);
			fails++;
		}

		draws_before += s0.prims;
		draws_after += s1.prims;
		printf("%-16s %6zu %6zu | %9zu %9zu | %9zu %9zu | %8.3f | %s\n",
		       models[mi].name, (size_t)s0.prims, (size_t)s1.prims,
		       (size_t)s0.indices, (size_t)s1.indices,
		       (size_t)s0.vertices, (size_t)s1.vertices,
		       merge_ms, fails ? "FAIL" : "byte-identical");
		total_fails += fails;

		/* cleanup: tool-owned memory first (nothing in `data` points into
		 * the pools after split except live accessors — cgltf_free only
		 * walks arrays, so order is safe either way) */
		cgltf_free(data);
		for (size_t k = 0; k < g.ptrs_n; ++k) free(g.ptrs[k]);
		free(g.ptrs);
		free(g.arena);
		free(g.recs);
		free_snapshot(&before);
	}

	if (draws_before) {
		double red = 100.0 * (double)(draws_before - draws_after) / (double)draws_before;
		printf("\ndraw calls: %zu -> %zu (%.1f%% reduction, need > 50%%)\n",
		       (size_t)draws_before, (size_t)draws_after, red);
		if (red <= 50.0) {
			printf("FAIL: draw-call reduction below 50%%\n");
			total_fails++;
		}
	}
	printf(total_fails ? "RESULT: FAIL (%d)\n" : "RESULT: PASS\n", total_fails);
	return total_fails ? 1 : 0;
}
