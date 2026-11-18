#ifndef CGLTF_CACHE_H
#define CGLTF_CACHE_H

/*
 * cgltf_cache.h - optional unpack cache for cgltf accessors.
 *
 * Every vertex fetch through cgltf_accessor_unpack_floats() re-decodes the
 * underlying accessor: integer components are converted one by one, normalized
 * values are divided, sparse overlays are re-applied. When the same accessor is
 * read repeatedly this work is paid over and over again.
 *
 * This file provides an opt-in cache layered strictly on top of the public
 * cgltf API. It caches, per accessor, the full float array produced by
 * cgltf_accessor_unpack_floats(), so repeated reads collapse into a memcpy.
 *
 * The cache is a separate object and does not change the behavior, signature or
 * semantics of any existing cgltf accessor function. Not including this file
 * leaves cgltf exactly as it was; with the cache disabled (max_bytes == 0 is NOT
 * a disable - see cgltf_unpack_cache_options::disabled) the wrapper returns
 * byte-for-byte identical results to calling cgltf_accessor_unpack_floats()
 * directly.
 *
 * Define CGLTF_CACHE_IMPLEMENTATION in exactly one translation unit before
 * including this file to emit the implementation.
 */

/*
 * cgltf.h places its CGLTF_IMPLEMENTATION section outside its own include
 * guard, so it must never be expanded twice in one translation unit. When the
 * user has already included cgltf.h (the normal usage pattern) its API
 * declarations are visible and we must not include it again.
 */
#ifndef CGLTF_H_INCLUDED__
#include "cgltf.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cgltf_unpack_cache_options
{
	/*
	 * Upper bound on the total number of bytes the cache is allowed to
	 * allocate for cached float payloads (entry bookkeeping is counted as
	 * well, so the real usage stays at or below this value).
	 * 0 means unlimited.
	 */
	cgltf_size max_bytes;

	/*
	 * When nonzero, the cache never stores anything: every call falls through
	 * to cgltf_accessor_unpack_floats(). Results are identical to reading
	 * without a cache; no statistics are recorded.
	 */
	int disabled;

	/* Allocator hooks; zeroed entries fall back to malloc/free. */
	void* (*alloc_func)(void* user, cgltf_size size);
	void (*free_func)(void* user, void* ptr);
	void* user_data;
} cgltf_unpack_cache_options;

typedef struct cgltf_unpack_cache_stats
{
	cgltf_size hits;        /* calls served from a cached float array */
	cgltf_size misses;      /* calls that had to run a full unpack */
	cgltf_size evictions;   /* entries removed to honor the byte limit */
	cgltf_size used_bytes;  /* bytes currently held by the cache */
	cgltf_size entry_count; /* number of entries currently held */
} cgltf_unpack_cache_stats;

/*
 * Fingerprint of the byte stream an accessor decodes from. A cached entry is
 * valid only while the accessor pointer and every field below stay unchanged.
 * See cgltf_unpack_cache_private.h-level comments in the implementation for the
 * exact invalidation rules.
 */
typedef struct cgltf_unpack_fingerprint
{
	/* Base accessor data (accessor->buffer_view, may be NULL). */
	const void* base_data;       /* resolved buffer_view data pointer */
	cgltf_size base_view_offset;
	cgltf_size base_view_size;
	cgltf_size base_view_stride;

	/* Sparse index/value data (valid when is_sparse != 0). */
	const void* sparse_index_data;
	cgltf_size sparse_index_view_offset;
	cgltf_size sparse_index_view_size;
	cgltf_size sparse_index_byte_offset;
	const void* sparse_value_data;
	cgltf_size sparse_value_view_offset;
	cgltf_size sparse_value_view_size;
	cgltf_size sparse_value_byte_offset;
	cgltf_size sparse_count;

	/* Accessor fields that participate in decoding. */
	cgltf_component_type component_type;
	cgltf_bool normalized;
	cgltf_type type;
	cgltf_size accessor_offset;
	cgltf_size accessor_count;
	cgltf_size accessor_stride;
	cgltf_bool is_sparse;
} cgltf_unpack_fingerprint;

typedef struct cgltf_unpack_cache_entry
{
	const cgltf_accessor* accessor;
	cgltf_unpack_fingerprint fingerprint;
	cgltf_float* floats;          /* full unpacked array, always available_floats long */
	cgltf_size float_count;       /* number of floats in the array */
	cgltf_size payload_bytes;     /* float_count * sizeof(cgltf_float) */
	cgltf_size last_used;         /* LRU timestamp (larger == more recent) */
	struct cgltf_unpack_cache_entry* next;
} cgltf_unpack_cache_entry;

typedef struct cgltf_unpack_cache
{
	cgltf_unpack_cache_options options;
	cgltf_unpack_cache_stats stats;
	cgltf_unpack_cache_entry* entries;
	cgltf_size clock; /* monotonic LRU tick */
} cgltf_unpack_cache;

void cgltf_unpack_cache_init(cgltf_unpack_cache* cache, const cgltf_unpack_cache_options* options);
void cgltf_unpack_cache_free(cgltf_unpack_cache* cache);

/* Drop every cached entry (statistics are preserved unless reset separately). */
void cgltf_unpack_cache_clear(cgltf_unpack_cache* cache);

/*
 * Drop the cached entry of a single accessor, if any. Call this after mutating
 * accessor contents in place without changing any data pointer or range - the
 * fingerprint cannot observe in-place byte edits.
 */
void cgltf_unpack_cache_invalidate(cgltf_unpack_cache* cache, const cgltf_accessor* accessor);

void cgltf_unpack_cache_reset_stats(cgltf_unpack_cache* cache);
void cgltf_unpack_cache_get_stats(const cgltf_unpack_cache* cache, cgltf_unpack_cache_stats* out_stats);

/*
 * Cached counterpart of cgltf_accessor_unpack_floats(). Has the same contract:
 * returns the number of available floats when out == NULL, otherwise decodes
 * up to float_count floats (clamped to the accessor's available count) into out
 * and returns the number written; returns 0 when the underlying data is not
 * available, exactly like the uncached function.
 */
cgltf_size cgltf_unpack_cache_unpack_floats(cgltf_unpack_cache* cache, const cgltf_accessor* accessor, cgltf_float* out, cgltf_size float_count);

#ifdef __cplusplus
}
#endif

#endif /* CGLTF_CACHE_H */

#ifdef CGLTF_CACHE_IMPLEMENTATION

#ifndef CGLTF_CACHE_IMPLEMENTATION_ONCE
#define CGLTF_CACHE_IMPLEMENTATION_ONCE

#include <string.h>
#include <stdlib.h>

static void* cgltf_unpack_cache_alloc(cgltf_unpack_cache* cache, cgltf_size size)
{
	if (cache->options.alloc_func)
		return cache->options.alloc_func(cache->options.user_data, size);
	return malloc(size);
}

static void cgltf_unpack_cache_dealloc(cgltf_unpack_cache* cache, void* ptr)
{
	if (cache->options.free_func)
	{
		cache->options.free_func(cache->options.user_data, ptr);
		return;
	}
	free(ptr);
}

void cgltf_unpack_cache_init(cgltf_unpack_cache* cache, const cgltf_unpack_cache_options* options)
{
	memset(cache, 0, sizeof(*cache));
	if (options)
		cache->options = *options;
}

static void cgltf_unpack_cache_remove_entry(cgltf_unpack_cache* cache, cgltf_unpack_cache_entry* entry, cgltf_unpack_cache_entry* prev)
{
	if (prev)
		prev->next = entry->next;
	else
		cache->entries = entry->next;

	cache->stats.used_bytes -= entry->payload_bytes + sizeof(cgltf_unpack_cache_entry);
	cache->stats.entry_count--;

	cgltf_unpack_cache_dealloc(cache, entry->floats);
	cgltf_unpack_cache_dealloc(cache, entry);
}

void cgltf_unpack_cache_clear(cgltf_unpack_cache* cache)
{
	cgltf_unpack_cache_entry* entry = cache->entries;
	while (entry)
	{
		cgltf_unpack_cache_entry* next = entry->next;
		cgltf_unpack_cache_dealloc(cache, entry->floats);
		cgltf_unpack_cache_dealloc(cache, entry);
		entry = next;
	}
	cache->entries = NULL;
	cache->stats.used_bytes = 0;
	cache->stats.entry_count = 0;
}

void cgltf_unpack_cache_free(cgltf_unpack_cache* cache)
{
	cgltf_unpack_cache_clear(cache);
}

void cgltf_unpack_cache_invalidate(cgltf_unpack_cache* cache, const cgltf_accessor* accessor)
{
	cgltf_unpack_cache_entry* prev = NULL;
	cgltf_unpack_cache_entry* entry = cache->entries;
	while (entry)
	{
		if (entry->accessor == accessor)
		{
			cgltf_unpack_cache_remove_entry(cache, entry, prev);
			return;
		}
		prev = entry;
		entry = entry->next;
	}
}

void cgltf_unpack_cache_reset_stats(cgltf_unpack_cache* cache)
{
	cgltf_size used_bytes = cache->stats.used_bytes;
	cgltf_size entry_count = cache->stats.entry_count;
	memset(&cache->stats, 0, sizeof(cache->stats));
	cache->stats.used_bytes = used_bytes;
	cache->stats.entry_count = entry_count;
}

void cgltf_unpack_cache_get_stats(const cgltf_unpack_cache* cache, cgltf_unpack_cache_stats* out_stats)
{
	*out_stats = cache->stats;
}

/*
 * Build the fingerprint of everything that determines the decoded float stream
 * of an accessor.
 *
 * CACHE INVALIDATION RULES (a cached entry is reused only while all of these
 * stay identical; any change makes the next read a miss and replaces the entry):
 *
 *  1. Data pointer change. The resolved byte pointer of the accessor's base
 *     buffer view (view->data override, or buffer->data + view->offset) and,
 *     for sparse accessors, the resolved pointers of the sparse index and value
 *     buffer views. A reload, realloc or re-pointing of any of these changes
 *     the decoded contents even when every range stays the same.
 *
 *  2. Range change. The offset and size of each involved buffer view plus the
 *     accessor offset and the sparse byte offsets: growing, shrinking or
 *     sliding the window the accessor reads from invalidates the entry. The
 *     accessor's element count, stride, component type, normalization flag,
 *     element type and sparse count are part of the fingerprint for the same
 *     reason - they define how the byte range is interpreted.
 *
 * In-place edits that keep every pointer and range identical cannot be detected
 * from the fingerprint; callers performing such edits must call
 * cgltf_unpack_cache_invalidate() (or clear) explicitly.
 */
static void cgltf_unpack_cache_make_fingerprint(const cgltf_accessor* accessor, cgltf_unpack_fingerprint* fp)
{
	memset(fp, 0, sizeof(*fp));

	fp->component_type = accessor->component_type;
	fp->normalized = accessor->normalized;
	fp->type = accessor->type;
	fp->accessor_offset = accessor->offset;
	fp->accessor_count = accessor->count;
	fp->accessor_stride = accessor->stride;
	fp->is_sparse = accessor->is_sparse;

	if (accessor->buffer_view)
	{
		const cgltf_buffer_view* view = accessor->buffer_view;
		fp->base_data = (const void*)cgltf_buffer_view_data(view);
		fp->base_view_offset = view->offset;
		fp->base_view_size = view->size;
		fp->base_view_stride = view->stride;
	}

	if (accessor->is_sparse)
	{
		const cgltf_accessor_sparse* sparse = &accessor->sparse;

		fp->sparse_count = sparse->count;
		fp->sparse_index_byte_offset = sparse->indices_byte_offset;
		fp->sparse_value_byte_offset = sparse->values_byte_offset;

		if (sparse->indices_buffer_view)
		{
			const cgltf_buffer_view* view = sparse->indices_buffer_view;
			fp->sparse_index_data = (const void*)cgltf_buffer_view_data(view);
			fp->sparse_index_view_offset = view->offset;
			fp->sparse_index_view_size = view->size;
		}
		if (sparse->values_buffer_view)
		{
			const cgltf_buffer_view* view = sparse->values_buffer_view;
			fp->sparse_value_data = (const void*)cgltf_buffer_view_data(view);
			fp->sparse_value_view_offset = view->offset;
			fp->sparse_value_view_size = view->size;
		}
	}
}

static cgltf_bool cgltf_unpack_cache_fingerprints_equal(const cgltf_unpack_fingerprint* a, const cgltf_unpack_fingerprint* b)
{
	return memcmp(a, b, sizeof(*a)) == 0;
}

/* Evict least-recently-used entries until needed_bytes fit under the limit. */
static void cgltf_unpack_cache_evict(cgltf_unpack_cache* cache, cgltf_size needed_bytes)
{
	while (cache->options.max_bytes != 0 && cache->entries != NULL &&
		cache->stats.used_bytes + needed_bytes > cache->options.max_bytes)
	{
		cgltf_unpack_cache_entry* lru = cache->entries;
		cgltf_unpack_cache_entry* lru_prev = NULL;
		cgltf_unpack_cache_entry* prev = NULL;
		cgltf_unpack_cache_entry* entry = cache->entries;

		while (entry)
		{
			if (entry->last_used < lru->last_used)
			{
				lru = entry;
				lru_prev = prev;
			}
			prev = entry;
			entry = entry->next;
		}

		cgltf_unpack_cache_remove_entry(cache, lru, lru_prev);
		cache->stats.evictions++;
	}
}

cgltf_size cgltf_unpack_cache_unpack_floats(cgltf_unpack_cache* cache, const cgltf_accessor* accessor, cgltf_float* out, cgltf_size float_count)
{
	cgltf_size available_floats = cgltf_accessor_unpack_floats(accessor, NULL, 0);

	/* NULL out keeps the exact cgltf_accessor_unpack_floats() contract. */
	if (out == NULL)
		return available_floats;

	/* Clamp exactly like cgltf_accessor_unpack_floats() does. */
	cgltf_size requested = available_floats < float_count ? available_floats : float_count;

	/* Disabled cache: pure pass-through, results identical to direct reads. */
	if (cache->options.disabled)
		return cgltf_accessor_unpack_floats(accessor, out, float_count);

	cgltf_unpack_fingerprint fingerprint;
	cgltf_unpack_cache_make_fingerprint(accessor, &fingerprint);

	cgltf_unpack_cache_entry* prev = NULL;
	cgltf_unpack_cache_entry* entry = cache->entries;
	while (entry)
	{
		if (entry->accessor == accessor)
		{
			if (cgltf_unpack_cache_fingerprints_equal(&entry->fingerprint, &fingerprint))
			{
				/* Cache hit: copy the requested prefix of the decoded array. */
				cache->stats.hits++;
				entry->last_used = ++cache->clock;
				memcpy(out, entry->floats, requested * sizeof(cgltf_float));
				return requested;
			}

			/*
			 * Same accessor object but its data pointer/range/decoding
			 * parameters changed: the cached copy is stale. Drop it and fall
			 * through to rebuild below.
			 */
			cgltf_unpack_cache_remove_entry(cache, entry, prev);
			break;
		}
		prev = entry;
		entry = entry->next;
	}

	cache->stats.misses++;

	cgltf_size payload_bytes = available_floats * sizeof(cgltf_float);
	cgltf_size entry_cost = payload_bytes + sizeof(cgltf_unpack_cache_entry);

	/*
	 * Entries larger than the whole configured limit are never cached - the
	 * limit must never be exceeded - but the read itself still succeeds
	 * directly.
	 */
	cgltf_bool cacheable = cache->options.max_bytes == 0 || entry_cost <= cache->options.max_bytes;
	if (cacheable)
		cgltf_unpack_cache_evict(cache, entry_cost);

	cgltf_float* floats = NULL;
	if (cacheable)
	{
		floats = (cgltf_float*)cgltf_unpack_cache_alloc(cache, payload_bytes);
		if (floats == NULL)
			cacheable = 0; /* allocation failure: degrade to an uncached read */
	}

	if (!cacheable)
		return cgltf_accessor_unpack_floats(accessor, out, float_count);

	cgltf_size decoded = cgltf_accessor_unpack_floats(accessor, floats, available_floats);
	if (decoded == 0)
	{
		/* Underlying data unavailable: mirror the uncached failure, do not cache. */
		cgltf_unpack_cache_dealloc(cache, floats);
		return 0;
	}

	cgltf_unpack_cache_entry* new_entry = (cgltf_unpack_cache_entry*)cgltf_unpack_cache_alloc(cache, sizeof(cgltf_unpack_cache_entry));
	if (new_entry == NULL)
	{
		cgltf_unpack_cache_dealloc(cache, floats);
		return cgltf_accessor_unpack_floats(accessor, out, float_count);
	}

	new_entry->accessor = accessor;
	new_entry->fingerprint = fingerprint;
	new_entry->floats = floats;
	new_entry->float_count = available_floats;
	new_entry->payload_bytes = payload_bytes;
	new_entry->last_used = ++cache->clock;
	new_entry->next = cache->entries;
	cache->entries = new_entry;

	cache->stats.used_bytes += entry_cost;
	cache->stats.entry_count++;

	memcpy(out, floats, requested * sizeof(cgltf_float));
	return requested;
}

#endif /* CGLTF_CACHE_IMPLEMENTATION_ONCE */
#endif /* CGLTF_CACHE_IMPLEMENTATION */
