#define CGLTF_IMPLEMENTATION
#include "../cgltf.h"

#define CGLTF_CACHE_IMPLEMENTATION
#include "../cgltf_cache.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

/*
 * Synthetic, in-memory glTF-like data. No JSON parsing is needed: the cache
 * only consumes cgltf_accessor / cgltf_buffer_view / cgltf_buffer structs.
 *
 * Four model variants exercise the required cases:
 *   small       - tiny normalized int16 vec3 accessor
 *   large       - big normalized int16 vec3 accessor (expensive to decode)
 *   sparse      - int16 vec3 base with a float vec3 sparse overlay
 *   interleaved - vec3 + vec2 packed into one shared strided buffer view
 */

#define SMALL_COUNT 64u
#define LARGE_COUNT 200000u
#define SPARSE_COUNT 5000u
#define SPARSE_OVERRIDES 37u
#define INTERLEAVED_COUNT 30000u
#define ITERATIONS 100

typedef struct test_alloc_header
{
	size_t size;
	struct test_alloc_header* next;
} test_alloc_header;

typedef struct
{
	test_alloc_header* head;
	size_t live_bytes;
	size_t peak_bytes;
	size_t alloc_count;
} test_alloc_ctx;

static void* test_alloc(void* user, cgltf_size size)
{
	test_alloc_ctx* ctx = (test_alloc_ctx*)user;
	test_alloc_header* hdr = (test_alloc_header*)malloc(sizeof(test_alloc_header) + size);
	if (!hdr)
		return NULL;
	hdr->size = size;
	hdr->next = ctx->head;
	ctx->head = hdr;
	ctx->live_bytes += size;
	if (ctx->live_bytes > ctx->peak_bytes)
		ctx->peak_bytes = ctx->live_bytes;
	ctx->alloc_count++;
	return (char*)hdr + sizeof(test_alloc_header);
}

static void test_free(void* user, void* ptr)
{
	if (!ptr)
		return;
	test_alloc_ctx* ctx = (test_alloc_ctx*)user;
	test_alloc_header* hdr = (test_alloc_header*)((char*)ptr - sizeof(test_alloc_header));

	test_alloc_header** link = &ctx->head;
	while (*link && *link != hdr)
		link = &(*link)->next;
	if (*link == hdr)
		*link = hdr->next;

	ctx->live_bytes -= hdr->size;
	free(hdr);
}

typedef struct
{
	cgltf_buffer buffer;
	cgltf_buffer_view small_view;
	cgltf_buffer_view large_view;
	cgltf_buffer_view interleaved_view;
	cgltf_buffer_view sparse_base_view;
	cgltf_buffer_view sparse_index_view;
	cgltf_buffer_view sparse_value_view;

	cgltf_accessor small_acc;
	cgltf_accessor large_acc;
	cgltf_accessor sparse_acc;
	cgltf_accessor interleaved_pos;
	cgltf_accessor interleaved_uv;

	/* Extra accessors for the memory-limit tests. */
	cgltf_buffer_view cap_a_view;
	cgltf_buffer_view cap_b_view;
	cgltf_accessor cap_a_acc;
	cgltf_accessor cap_b_acc;

	size_t total_size;
} test_model;

static void fill_i16_vec3(int16_t* dst, cgltf_size count, cgltf_size salt)
{
	for (cgltf_size i = 0; i < count; ++i)
		for (cgltf_size c = 0; c < 3; ++c)
			dst[i * 3 + c] = (int16_t)((i * 7 + c * 13 + salt * 101) % 30001 - 15000);
}

static void build_model(test_model* m)
{
	memset(m, 0, sizeof(*m));

	const cgltf_size small_bytes = SMALL_COUNT * 3 * sizeof(int16_t);
	const cgltf_size large_bytes = LARGE_COUNT * 3 * sizeof(int16_t);
	const cgltf_size interleaved_stride = 3 * sizeof(float) + 2 * sizeof(float);
	const cgltf_size interleaved_bytes = INTERLEAVED_COUNT * interleaved_stride;
	const cgltf_size sparse_base_bytes = SPARSE_COUNT * 3 * sizeof(int16_t);
	const cgltf_size sparse_index_bytes = SPARSE_OVERRIDES * sizeof(uint16_t);
	const cgltf_size sparse_value_bytes = SPARSE_OVERRIDES * 3 * sizeof(float);
	const cgltf_size cap_bytes = 10000u * 3 * sizeof(float);

	cgltf_size offsets[7];
	offsets[0] = 0;
	offsets[1] = (offsets[0] + small_bytes + 15) & ~(cgltf_size)15;
	offsets[2] = (offsets[1] + large_bytes + 15) & ~(cgltf_size)15;
	offsets[3] = (offsets[2] + interleaved_bytes + 15) & ~(cgltf_size)15;
	offsets[4] = (offsets[3] + sparse_base_bytes + 15) & ~(cgltf_size)15;
	offsets[5] = (offsets[4] + sparse_index_bytes + 15) & ~(cgltf_size)15;
	offsets[6] = (offsets[5] + sparse_value_bytes + 15) & ~(cgltf_size)15;
	m->total_size = offsets[6] + 2 * cap_bytes;

	uint8_t* mem = (uint8_t*)calloc(1, m->total_size);
	m->buffer.data = mem;
	m->buffer.size = m->total_size;

	/* small: normalized int16 vec3, tightly packed. */
	m->small_view.buffer = &m->buffer;
	m->small_view.offset = offsets[0];
	m->small_view.size = small_bytes;
	fill_i16_vec3((int16_t*)(mem + offsets[0]), SMALL_COUNT, 1);
	m->small_acc.buffer_view = &m->small_view;
	m->small_acc.component_type = cgltf_component_type_r_16;
	m->small_acc.normalized = 1;
	m->small_acc.type = cgltf_type_vec3;
	m->small_acc.count = SMALL_COUNT;
	m->small_acc.offset = 0;
	m->small_acc.stride = 3 * sizeof(int16_t);

	/* large: normalized int16 vec3, tightly packed. */
	m->large_view.buffer = &m->buffer;
	m->large_view.offset = offsets[1];
	m->large_view.size = large_bytes;
	fill_i16_vec3((int16_t*)(mem + offsets[1]), LARGE_COUNT, 2);
	m->large_acc.buffer_view = &m->large_view;
	m->large_acc.component_type = cgltf_component_type_r_16;
	m->large_acc.normalized = 1;
	m->large_acc.type = cgltf_type_vec3;
	m->large_acc.count = LARGE_COUNT;
	m->large_acc.offset = 0;
	m->large_acc.stride = 3 * sizeof(int16_t);

	/* interleaved: float vec3 position + float vec2 uv sharing one view. */
	m->interleaved_view.buffer = &m->buffer;
	m->interleaved_view.offset = offsets[2];
	m->interleaved_view.size = interleaved_bytes;
	m->interleaved_view.stride = interleaved_stride;
	{
		uint8_t* base = mem + offsets[2];
		for (cgltf_size i = 0; i < INTERLEAVED_COUNT; ++i)
		{
			float* v = (float*)(base + i * interleaved_stride);
			v[0] = (float)i * 0.5f;
			v[1] = (float)i * 1.5f;
			v[2] = -(float)i;
			v[3] = (float)((i * 3) % 97) / 97.0f;
			v[4] = (float)((i * 5 + 1) % 89) / 89.0f;
		}
	}
	m->interleaved_pos.buffer_view = &m->interleaved_view;
	m->interleaved_pos.component_type = cgltf_component_type_r_32f;
	m->interleaved_pos.type = cgltf_type_vec3;
	m->interleaved_pos.count = INTERLEAVED_COUNT;
	m->interleaved_pos.offset = 0;
	m->interleaved_pos.stride = interleaved_stride;

	m->interleaved_uv.buffer_view = &m->interleaved_view;
	m->interleaved_uv.component_type = cgltf_component_type_r_32f;
	m->interleaved_uv.type = cgltf_type_vec2;
	m->interleaved_uv.count = INTERLEAVED_COUNT;
	m->interleaved_uv.offset = 3 * sizeof(float);
	m->interleaved_uv.stride = interleaved_stride;

	/* sparse: int16 vec3 base, overridden at scattered indices by float vec3. */
	m->sparse_base_view.buffer = &m->buffer;
	m->sparse_base_view.offset = offsets[3];
	m->sparse_base_view.size = sparse_base_bytes;
	fill_i16_vec3((int16_t*)(mem + offsets[3]), SPARSE_COUNT, 3);

	m->sparse_index_view.buffer = &m->buffer;
	m->sparse_index_view.offset = offsets[4];
	m->sparse_index_view.size = sparse_index_bytes;
	m->sparse_value_view.buffer = &m->buffer;
	m->sparse_value_view.offset = offsets[5];
	m->sparse_value_view.size = sparse_value_bytes;

	{
		uint16_t* indices = (uint16_t*)(mem + offsets[4]);
		float* values = (float*)(mem + offsets[5]);
		for (cgltf_size i = 0; i < SPARSE_OVERRIDES; ++i)
		{
			indices[i] = (uint16_t)((i * 131 + 7) % SPARSE_COUNT);
			values[i * 3 + 0] = 1000.0f + (float)i;
			values[i * 3 + 1] = 2000.0f + (float)i;
			values[i * 3 + 2] = 3000.0f + (float)i;
		}
	}
	m->sparse_acc.buffer_view = &m->sparse_base_view;
	m->sparse_acc.component_type = cgltf_component_type_r_16;
	m->sparse_acc.normalized = 1;
	m->sparse_acc.type = cgltf_type_vec3;
	m->sparse_acc.count = SPARSE_COUNT;
	m->sparse_acc.offset = 0;
	m->sparse_acc.stride = 3 * sizeof(int16_t);
	m->sparse_acc.is_sparse = 1;
	m->sparse_acc.sparse.count = SPARSE_OVERRIDES;
	m->sparse_acc.sparse.indices_buffer_view = &m->sparse_index_view;
	m->sparse_acc.sparse.indices_component_type = cgltf_component_type_r_16u;
	m->sparse_acc.sparse.indices_byte_offset = 0;
	m->sparse_acc.sparse.values_buffer_view = &m->sparse_value_view;
	m->sparse_acc.sparse.values_byte_offset = 0;

	/* cap test accessors: two float vec3 arrays of 10k elements each. */
	for (int which = 0; which < 2; ++which)
	{
		cgltf_buffer_view* view = which == 0 ? &m->cap_a_view : &m->cap_b_view;
		cgltf_accessor* acc = which == 0 ? &m->cap_a_acc : &m->cap_b_acc;
		cgltf_size off = offsets[6] + (cgltf_size)which * cap_bytes;
		view->buffer = &m->buffer;
		view->offset = off;
		view->size = cap_bytes;
		float* v = (float*)(mem + off);
		for (cgltf_size i = 0; i < 10000u * 3u; ++i)
			v[i] = (float)i + which * 0.25f;
		acc->buffer_view = view;
		acc->component_type = cgltf_component_type_r_32f;
		acc->type = cgltf_type_vec3;
		acc->count = 10000u;
		acc->offset = 0;
		acc->stride = 3 * sizeof(float);
	}
}

static void free_model(test_model* m)
{
	free(m->buffer.data);
	memset(m, 0, sizeof(*m));
}

static double seconds_now(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int check_equal(const cgltf_float* a, const cgltf_float* b, cgltf_size count, const char* what)
{
	for (cgltf_size i = 0; i < count; ++i)
	{
		if (a[i] != b[i])
		{
			printf("FAIL: %s mismatch at float %zu: %f vs %f\n", what, (size_t)i, a[i], b[i]);
			return 1;
		}
	}
	return 0;
}

static cgltf_size floats_of(const cgltf_accessor* acc)
{
	return cgltf_accessor_unpack_floats(acc, NULL, 0);
}

/* 1. Cached results must equal direct unpack results for every model variant. */
static int test_correctness(const test_model* m, test_alloc_ctx* alloc)
{
	cgltf_unpack_cache_options opts;
	memset(&opts, 0, sizeof(opts));
	opts.alloc_func = test_alloc;
	opts.free_func = test_free;
	opts.user_data = alloc;

	cgltf_unpack_cache cache;
	cgltf_unpack_cache_init(&cache, &opts);

	const cgltf_accessor* accs[] = { &m->small_acc, &m->large_acc, &m->sparse_acc, &m->interleaved_pos, &m->interleaved_uv };
	const char* names[] = { "small", "large", "sparse", "interleaved-pos", "interleaved-uv" };
	int failures = 0;

	for (int round = 0; round < 3; ++round)
	{
		for (int i = 0; i < 5; ++i)
		{
			cgltf_size n = floats_of(accs[i]);
			cgltf_float* direct = (cgltf_float*)malloc(n * sizeof(cgltf_float));
			cgltf_float* cached = (cgltf_float*)malloc(n * sizeof(cgltf_float));

			cgltf_size rd = cgltf_accessor_unpack_floats(accs[i], direct, n);
			cgltf_size rc = cgltf_unpack_cache_unpack_floats(&cache, accs[i], cached, n);
			if (rd != rc || check_equal(direct, cached, n, names[i]))
			{
				printf("FAIL: correctness round %d %s (rd=%zu rc=%zu)\n", round, names[i], (size_t)rd, (size_t)rc);
				failures++;
			}

			/* Partial reads must also match and stay in cache. */
			cgltf_size part = n / 2;
			memset(cached, 0xCC, n * sizeof(cgltf_float));
			rc = cgltf_unpack_cache_unpack_floats(&cache, accs[i], cached, part);
			if (rc != part || check_equal(direct, cached, part, names[i]))
			{
				printf("FAIL: partial read %s\n", names[i]);
				failures++;
			}

			free(direct);
			free(cached);
		}
	}

	cgltf_unpack_cache_stats st;
	cgltf_unpack_cache_get_stats(&cache, &st);
	if (st.misses != 5 || st.hits == 0)
	{
		printf("FAIL: correctness stats hits=%zu misses=%zu (expect 5 misses, >0 hits)\n", (size_t)st.hits, (size_t)st.misses);
		failures++;
	}

	cgltf_unpack_cache_free(&cache);
	if (alloc->live_bytes != 0)
	{
		printf("FAIL: cache leaked %zu bytes\n", (size_t)alloc->live_bytes);
		failures++;
	}
	return failures;
}

/* 2. Disabled cache: identical output, no allocations, no statistics. */
static int test_disabled(const test_model* m, test_alloc_ctx* alloc)
{
	cgltf_unpack_cache_options opts;
	memset(&opts, 0, sizeof(opts));
	opts.disabled = 1;
	opts.alloc_func = test_alloc;
	opts.free_func = test_free;
	opts.user_data = alloc;

	cgltf_unpack_cache cache;
	cgltf_unpack_cache_init(&cache, &opts);

	const cgltf_accessor* accs[] = { &m->small_acc, &m->large_acc, &m->sparse_acc, &m->interleaved_pos, &m->interleaved_uv };
	int failures = 0;

	for (int i = 0; i < 5; ++i)
	{
		cgltf_size n = floats_of(accs[i]);
		cgltf_float* direct = (cgltf_float*)malloc(n * sizeof(cgltf_float));
		cgltf_float* cached = (cgltf_float*)malloc(n * sizeof(cgltf_float));
		cgltf_size rd = cgltf_accessor_unpack_floats(accs[i], direct, n);
		cgltf_size rc = cgltf_unpack_cache_unpack_floats(&cache, accs[i], cached, n);
		if (rd != rc || check_equal(direct, cached, n, "disabled"))
			failures++;
		free(direct);
		free(cached);
	}

	cgltf_unpack_cache_stats st;
	cgltf_unpack_cache_get_stats(&cache, &st);
	if (st.hits != 0 || st.misses != 0 || st.entry_count != 0 || alloc->alloc_count != 0)
	{
		printf("FAIL: disabled cache recorded state: hits=%zu misses=%zu entries=%zu allocs=%zu\n",
			(size_t)st.hits, (size_t)st.misses, (size_t)st.entry_count, (size_t)alloc->alloc_count);
		failures++;
	}

	cgltf_unpack_cache_free(&cache);
	return failures;
}

/*
 * 3. Invalidation on data pointer / range change.
 *    Reallocating the buffer changes the resolved data pointer; shrinking the
 *    view changes its range. Both must force a miss and refresh the entry.
 */
static int test_invalidation(test_model* m, test_alloc_ctx* alloc)
{
	cgltf_unpack_cache_options opts;
	memset(&opts, 0, sizeof(opts));
	opts.alloc_func = test_alloc;
	opts.free_func = test_free;
	opts.user_data = alloc;

	cgltf_unpack_cache cache;
	cgltf_unpack_cache_init(&cache, &opts);

	const cgltf_accessor* acc = &m->large_acc;
	cgltf_size n = floats_of(acc);
	cgltf_float* out = (cgltf_float*)malloc(n * sizeof(cgltf_float));
	cgltf_float* ref = (cgltf_float*)malloc(n * sizeof(cgltf_float));
	int failures = 0;

	cgltf_unpack_cache_unpack_floats(&cache, acc, out, n); /* miss */
	cgltf_unpack_cache_unpack_floats(&cache, acc, out, n); /* hit */
	cgltf_unpack_cache_stats st;
	cgltf_unpack_cache_get_stats(&cache, &st);
	if (st.hits != 1 || st.misses != 1)
	{
		printf("FAIL: invalidation baseline hits=%zu misses=%zu\n", (size_t)st.hits, (size_t)st.misses);
		failures++;
	}

	/* Pointer change: rebuild the backing storage at a new address. */
	void* old_data = m->buffer.data;
	uint8_t* new_mem = (uint8_t*)malloc(m->total_size);
	memcpy(new_mem, old_data, m->total_size);
	m->buffer.data = new_mem;

	cgltf_accessor_unpack_floats(acc, ref, n);
	cgltf_size rc = cgltf_unpack_cache_unpack_floats(&cache, acc, out, n);
	cgltf_unpack_cache_get_stats(&cache, &st);
	if (rc != n || check_equal(ref, out, n, "pointer-change") || st.misses != 2 || st.hits != 1 || st.entry_count != 1)
	{
		printf("FAIL: pointer change not invalidated (misses=%zu hits=%zu entries=%zu)\n",
			(size_t)st.misses, (size_t)st.hits, (size_t)st.entry_count);
		failures++;
	}

	/* Range change: shrink the view; entry must be rebuilt again. */
	cgltf_size old_size = m->large_view.size;
	m->large_view.size = old_size / 2;
	cgltf_unpack_cache_reset_stats(&cache);
	cgltf_unpack_cache_unpack_floats(&cache, acc, out, n);
	cgltf_unpack_cache_get_stats(&cache, &st);
	if (st.misses != 1 || st.hits != 0)
	{
		printf("FAIL: range change not invalidated (misses=%zu hits=%zu)\n", (size_t)st.misses, (size_t)st.hits);
		failures++;
	}
	m->large_view.size = old_size;

	/* Rebuild the entry against the restored range before editing contents. */
	cgltf_unpack_cache_reset_stats(&cache);
	cgltf_unpack_cache_unpack_floats(&cache, acc, out, n);

	/* In-place content edit with unchanged pointers/ranges needs explicit invalidation. */
	((int16_t*)((uint8_t*)m->buffer.data + m->large_view.offset))[0] = (int16_t)-12345;
	cgltf_accessor_unpack_floats(acc, ref, n);
	cgltf_unpack_cache_unpack_floats(&cache, acc, out, n); /* stale hit expected */
	if (memcmp(ref, out, 3 * sizeof(cgltf_float)) == 0)
	{
		/* Equal means the cache silently refreshed, i.e. the stale case was not exercised. */
		printf("FAIL: stale cache unexpectedly matched mutated data\n");
		failures++;
	}
	else
	{
		/* The expected mismatch proves the stale entry was served; now refresh explicitly. */
		cgltf_unpack_cache_invalidate(&cache, acc);
		cgltf_unpack_cache_unpack_floats(&cache, acc, out, n);
		if (check_equal(ref, out, n, "after-explicit-invalidate"))
		{
			printf("FAIL: explicit invalidation did not refresh entry\n");
			failures++;
		}
	}

	free(out);
	free(ref);
	free(new_mem);
	m->buffer.data = old_data;
	cgltf_unpack_cache_free(&cache);
	if (alloc->live_bytes != 0)
	{
		printf("FAIL: invalidation leaked %zu bytes\n", (size_t)alloc->live_bytes);
		failures++;
	}
	return failures;
}

/* 4. Configurable memory limit: eviction keeps usage at or below the cap. */
static int test_memory_cap(const test_model* m, test_alloc_ctx* alloc)
{
	cgltf_unpack_cache_options opts;
	memset(&opts, 0, sizeof(opts));
	opts.alloc_func = test_alloc;
	opts.free_func = test_free;
	opts.user_data = alloc;

	int failures = 0;

	/* Each cap accessor payload is 10000*3*4 = 120000 bytes plus bookkeeping. */
	cgltf_size entry_cost = 10000u * 3u * sizeof(float) + sizeof(cgltf_unpack_cache_entry);
	opts.max_bytes = entry_cost + 1; /* room for exactly one entry */

	cgltf_unpack_cache cache;
	cgltf_unpack_cache_init(&cache, &opts);

	cgltf_size n = floats_of(&m->cap_a_acc);
	cgltf_float* out = (cgltf_float*)malloc(n * sizeof(cgltf_float));

	cgltf_unpack_cache_unpack_floats(&cache, &m->cap_a_acc, out, n);
	cgltf_unpack_cache_unpack_floats(&cache, &m->cap_b_acc, out, n); /* evicts A */

	cgltf_unpack_cache_stats st;
	cgltf_unpack_cache_get_stats(&cache, &st);
	if (st.entry_count != 1 || st.evictions != 1)
	{
		printf("FAIL: cap eviction entries=%zu evictions=%zu\n", (size_t)st.entry_count, (size_t)st.evictions);
		failures++;
	}
	if (st.used_bytes > opts.max_bytes || alloc->live_bytes > opts.max_bytes)
	{
		printf("FAIL: cap exceeded used=%zu live=%zu cap=%zu\n",
			(size_t)st.used_bytes, (size_t)alloc->live_bytes, (size_t)opts.max_bytes);
		failures++;
	}

	/* LRU order: touch B, then A must evict B (not the other way around). */
	cgltf_unpack_cache_reset_stats(&cache);
	cgltf_unpack_cache_unpack_floats(&cache, &m->cap_b_acc, out, n); /* hit */
	cgltf_unpack_cache_unpack_floats(&cache, &m->cap_a_acc, out, n); /* miss, evicts B */
	cgltf_unpack_cache_get_stats(&cache, &st);
	if (st.hits != 1 || st.misses != 1 || st.evictions != 1 || st.entry_count != 1)
	{
		printf("FAIL: LRU order hits=%zu misses=%zu evictions=%zu entries=%zu\n",
			(size_t)st.hits, (size_t)st.misses, (size_t)st.evictions, (size_t)st.entry_count);
		failures++;
	}
	if (cache.entries->accessor != &m->cap_a_acc)
	{
		printf("FAIL: LRU kept the wrong entry\n");
		failures++;
	}

	/* An entry larger than the cap is never cached, but reads still work. */
	cgltf_unpack_cache_clear(&cache);
	cgltf_unpack_cache_reset_stats(&cache);
	opts.max_bytes = 4096; /* large accessor is far bigger than this */
	cgltf_size big_n = floats_of(&m->large_acc);
	cgltf_float* big_out = (cgltf_float*)malloc(big_n * sizeof(cgltf_float));
	cgltf_float* big_ref = (cgltf_float*)malloc(big_n * sizeof(cgltf_float));
	cgltf_accessor_unpack_floats(&m->large_acc, big_ref, big_n);
	cgltf_size r1 = cgltf_unpack_cache_unpack_floats(&cache, &m->large_acc, big_out, big_n);
	cgltf_size r2 = cgltf_unpack_cache_unpack_floats(&cache, &m->large_acc, big_out, big_n);
	cgltf_unpack_cache_get_stats(&cache, &st);
	if (r1 != big_n || r2 != big_n || st.entry_count != 0 || st.misses != 2 ||
		check_equal(big_ref, big_out, big_n, "oversized"))
	{
		printf("FAIL: oversized entry handling r1=%zu r2=%zu entries=%zu misses=%zu\n",
			(size_t)r1, (size_t)r2, (size_t)st.entry_count, (size_t)st.misses);
		failures++;
	}
	if (alloc->live_bytes > 4096)
	{
		printf("FAIL: oversized read exceeded cap, live=%zu\n", (size_t)alloc->live_bytes);
		failures++;
	}

	free(big_out);
	free(big_ref);
	free(out);
	cgltf_unpack_cache_free(&cache);
	if (alloc->live_bytes != 0)
	{
		printf("FAIL: cap test leaked %zu bytes\n", (size_t)alloc->live_bytes);
		failures++;
	}
	return failures;
}

/* 5. Benchmark: 100 repeated full reads per variant, direct vs cached. */
typedef struct
{
	const char* name;
	const cgltf_accessor* acc;
} bench_case;

static int benchmark(const test_model* m, test_alloc_ctx* alloc)
{
	cgltf_unpack_cache_options opts;
	memset(&opts, 0, sizeof(opts));
	opts.alloc_func = test_alloc;
	opts.free_func = test_free;
	opts.user_data = alloc;

	bench_case cases[] = {
		{ "small",       &m->small_acc },
		{ "large",       &m->large_acc },
		{ "sparse",      &m->sparse_acc },
		{ "interleaved", &m->interleaved_pos },
	};
	const int case_count = (int)(sizeof(cases) / sizeof(cases[0]));

	printf("\n%-12s %12s %12s %10s %10s %10s %10s %8s\n",
		"model", "direct(ms)", "cached(ms)", "hits", "misses", "hit-rate", "speedup", "floats");
	printf("--------------------------------------------------------------------------------------------------------\n");

	int failures = 0;

	for (int i = 0; i < case_count; ++i)
	{
		const cgltf_accessor* acc = cases[i].acc;
		cgltf_size n = floats_of(acc);
		cgltf_float* out = (cgltf_float*)malloc(n * sizeof(cgltf_float));

		/* Warm up: faults, page-in and branch predictors shared by both runs. */
		cgltf_accessor_unpack_floats(acc, out, n);

		double t0 = seconds_now();
		for (int it = 0; it < ITERATIONS; ++it)
			cgltf_accessor_unpack_floats(acc, out, n);
		double direct_secs = seconds_now() - t0;

		cgltf_unpack_cache cache;
		cgltf_unpack_cache_init(&cache, &opts);
		cgltf_unpack_cache_unpack_floats(&cache, acc, out, n); /* populate, excluded from timing */

		double t1 = seconds_now();
		for (int it = 0; it < ITERATIONS; ++it)
			cgltf_unpack_cache_unpack_floats(&cache, acc, out, n);
		double cached_secs = seconds_now() - t1;

		cgltf_unpack_cache_stats st;
		cgltf_unpack_cache_get_stats(&cache, &st);
		double hit_rate = (double)st.hits / (double)(st.hits + st.misses);
		double speedup = cached_secs > 0 ? direct_secs / cached_secs : 0.0;

		printf("%-12s %12.4f %12.4f %10zu %10zu %9.2f%% %9.2fx %8zu\n",
			cases[i].name,
			direct_secs * 1000.0, cached_secs * 1000.0,
			(size_t)st.hits, (size_t)st.misses, hit_rate * 100.0, speedup, (size_t)n);

		if (st.hits != ITERATIONS || st.misses != 1 || hit_rate < 0.99)
		{
			printf("FAIL: benchmark hit rate for %s\n", cases[i].name);
			failures++;
		}
#ifdef CGLTF_CACHE_CHECK_SPEEDUP
		/* Timing threshold is only meaningful in the sanitizer-free build. */
		if (speedup < 5.0)
		{
			printf("FAIL: speedup for %s is %.2fx, acceptance threshold is 5x\n", cases[i].name, speedup);
			failures++;
		}
#endif

		cgltf_unpack_cache_free(&cache);
		free(out);
	}

	if (alloc->live_bytes != 0)
	{
		printf("FAIL: benchmark leaked %zu bytes (peak %zu)\n", (size_t)alloc->live_bytes, (size_t)alloc->peak_bytes);
		failures++;
	}
	return failures;
}

int main(void)
{
	test_model model;
	build_model(&model);

	int failures = 0;
	test_alloc_ctx alloc_a = { 0 };
	test_alloc_ctx alloc_b = { 0 };
	test_alloc_ctx alloc_c = { 0 };
	test_alloc_ctx alloc_d = { 0 };

	failures += test_correctness(&model, &alloc_a);
	failures += test_disabled(&model, &alloc_b);
	failures += test_invalidation(&model, &alloc_c);
	failures += test_memory_cap(&model, &alloc_d);
	failures += benchmark(&model, &alloc_a);

	free_model(&model);

	if (failures == 0)
		printf("\nAll cache tests passed.\n");
	else
		printf("\n%d test failure(s).\n", failures);
	return failures == 0 ? 0 : 1;
}
