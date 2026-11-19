#!/usr/bin/env python3
"""Independent verifier for mesh optimizer results.

Parses the original and optimized GLB files with the Python standard
library only (run with -I) and checks:

  1. Every output primitive reconstructs the exact same triangle soup as
     its input: same multiset of triangles, and every corner carries the
     exact same per-vertex attribute values (position, normal, uv and any
     morph target deltas). Winding order within each triangle is
     preserved.
  2. Output vertices are fully deduplicated (no two output vertices have
     identical attribute tuples).
  3. The optimizer's reported vertex/index counts and bandwidth numbers
     match an independent recomputation, and total bandwidth dropped by
     at least 20%%.
  4. The optimized GLB is structurally valid (4-byte alignment, sane
     accessor/bufferView references, legal index component types).

Usage: python3 -I verify.py <samples_dir> <out_dir> <stats.json>
"""

import json
import struct
import sys


F32, U16, U32 = 5126, 5123, 5125
COMP_SIZE = {5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4}
TYPE_COMPS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4,
              "MAT4": 16}


def read_glb(path):
    with open(path, "rb") as f:
        blob = f.read()
    magic, version, total = struct.unpack_from("<III", blob, 0)
    assert magic == 0x46546C67, "bad GLB magic in %s" % path
    assert version == 2
    assert total == len(blob), "truncated GLB %s" % path
    off = 12
    chunks = {}
    while off < len(blob):
        length, ctype = struct.unpack_from("<II", blob, off)
        off += 8
        chunks[ctype] = blob[off:off + length]
        off += length
    gltf = json.loads(chunks[0x4E4F534A].decode("utf-8"))
    bin_data = chunks.get(0x004E4942, b"")
    return gltf, bin_data


def read_accessor(gltf, bin_data, acc_idx):
    """Return a list of tuples, one tuple per accessor element."""
    acc = gltf["accessors"][acc_idx]
    view = gltf["bufferViews"][acc["bufferView"]]
    ctype = acc["componentType"]
    comps = TYPE_COMPS[acc["type"]]
    count = acc["count"]
    base = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
    stride = view.get("byteStride", comps * COMP_SIZE[ctype])
    csize = COMP_SIZE[ctype]

    fmt = {5120: "b", 5121: "B", 5122: "h", 5123: "H",
           5125: "I", 5126: "f"}[ctype]
    out = []
    for i in range(count):
        row = struct.unpack_from("<%d%s" % (comps, fmt), bin_data,
                                 base + i * stride)
        out.append(row)
    return out


def primitive_signatures(gltf, bin_data, prim):
    """Return (corners, indices_or_None, attr_names) where corners is a
    list of per-corner attribute tuples covering all attributes and morph
    targets in stable order."""
    attr_names = sorted(prim["attributes"].keys())
    target_tuples = []
    for target in prim.get("targets", []):
        for name in sorted(target.keys()):
            target_tuples.append(("target", name, target[name]))
    tables = []
    for name in attr_names:
        tables.append(read_accessor(gltf, bin_data, prim["attributes"][name]))
    for kind, name, acc_idx in target_tuples:
        tables.append(read_accessor(gltf, bin_data, acc_idx))

    vcount = len(tables[0])
    corners = [tuple(t[v] for t in tables) for v in range(vcount)]

    indices = None
    if "indices" in prim:
        indices = [v[0] for v in read_accessor(
            gltf, bin_data, prim["indices"])]
    return corners, indices, attr_names


def triangles_of(corners, indices):
    if indices is not None:
        ids = indices
    else:
        ids = list(range(len(corners)))
    assert len(ids) % 3 == 0
    tris = []
    for i in range(0, len(ids), 3):
        tris.append((corners[ids[i]], corners[ids[i + 1]],
                     corners[ids[i + 2]]))
    return tris


def cache_misses(ids, cache_size=16):
    """FIFO post-transform cache simulation over vertex ids, mirroring
    the C tool. For a non-indexed draw every corner is its own id, so
    every corner is a physical fetch with no cache reuse."""
    cache = []
    misses = 0
    for v in ids:
        if v not in cache:
            misses += 1
            cache.append(v)
            if len(cache) > cache_size:
                cache.pop(0)
    return misses


def primitive_stride(gltf, prim):
    total = 0
    for acc_idx in prim["attributes"].values():
        acc = gltf["accessors"][acc_idx]
        total += TYPE_COMPS[acc["type"]] * COMP_SIZE[acc["componentType"]]
    for target in prim.get("targets", []):
        for acc_idx in target.values():
            acc = gltf["accessors"][acc_idx]
            total += TYPE_COMPS[acc["type"]] * COMP_SIZE[acc["componentType"]]
    return total


def check_model(name, in_gltf, in_bin, out_gltf, out_bin, stats):
    errors = []
    in_meshes = in_gltf["meshes"]
    out_meshes = out_gltf["meshes"]
    assert len(in_meshes) == len(out_meshes), "mesh count changed"

    stat_prims = {(p["mesh"], p["primitive"]): p
                  for p in stats["primitives"]}
    total_before = total_after = 0.0

    for mi, (im, om) in enumerate(zip(in_meshes, out_meshes)):
        assert len(im["primitives"]) == len(om["primitives"]), \
            "primitive count changed in mesh %d" % mi
        for pi, (ip, op) in enumerate(zip(im["primitives"], om["primitives"])):
            in_corners, in_idx, attr_names = primitive_signatures(
                in_gltf, in_bin, ip)
            out_corners, out_idx, out_attr_names = primitive_signatures(
                out_gltf, out_bin, op)
            assert attr_names == out_attr_names, "attribute set changed"
            assert out_idx is not None, "output primitive not indexed"

            in_tris = triangles_of(in_corners, in_idx)
            out_tris = triangles_of(out_corners, out_idx)

            # 1. Triangle soup identity (order-independent) with winding.
            if sorted(map(repr, in_tris)) != sorted(map(repr, out_tris)):
                errors.append("%s[%d]: triangle soups differ" % (name, pi))
                continue

            # Winding must be preserved per matched triangle.
            in_set = {}
            for t in in_tris:
                in_set.setdefault(repr(t), []).append(t)
            for t in out_tris:
                bucket = in_set[repr(t)]
                match = bucket.pop()
                if t != match:
                    errors.append("%s[%d]: triangle winding changed"
                                  % (name, pi))
                    break

            # 2. Output vertices are unique.
            if len(set(out_corners)) != len(out_corners):
                errors.append("%s[%d]: output still has duplicate vertices"
                              % (name, pi))

            # 3. Every output vertex exists in the input vertex set.
            in_set_all = set(in_corners)
            for sig in out_corners:
                if sig not in in_set_all:
                    errors.append("%s[%d]: output vertex not present in input"
                                  % (name, pi))
                    break

            # 4. Independent bandwidth recomputation.
            stride = primitive_stride(out_gltf, op)
            in_ids = in_idx if in_idx is not None else list(range(len(in_corners)))
            out_ids = out_idx
            in_misses = cache_misses(in_ids)
            out_misses = cache_misses(out_ids)
            in_index_bytes = 0
            if in_idx is not None:
                iacc = in_gltf["accessors"][ip["indices"]]
                in_index_bytes = len(in_idx) * COMP_SIZE[iacc["componentType"]]
            oacc = out_gltf["accessors"][op["indices"]]
            out_index_bytes = len(out_idx) * COMP_SIZE[oacc["componentType"]]
            bw_before = in_misses * stride + in_index_bytes
            bw_after = out_misses * stride + out_index_bytes
            total_before += bw_before
            total_after += bw_after

            # 5. Cross-check reported stats.
            st = stat_prims.get((im.get("name", ""), pi))
            assert st is not None, "missing stats for %s[%d]" % (name, pi)
            if st["vertices_before"] != len(in_corners):
                errors.append("%s[%d]: vertices_before mismatch" % (name, pi))
            if st["vertices_after"] != len(out_corners):
                errors.append("%s[%d]: vertices_after mismatch" % (name, pi))
            exp_idx_before = len(in_idx) if in_idx is not None else len(in_tris) * 3
            if st["indices_before"] != exp_idx_before:
                errors.append("%s[%d]: indices_before mismatch" % (name, pi))
            if st["indices_after"] != len(out_idx):
                errors.append("%s[%d]: indices_after mismatch" % (name, pi))
            if abs(st["bandwidth_before_bytes"] - bw_before) > 1.0:
                errors.append("%s[%d]: bandwidth_before mismatch "
                              "(%s vs %d)" % (name, pi,
                                              st["bandwidth_before_bytes"],
                                              bw_before))
            if abs(st["bandwidth_after_bytes"] - bw_after) > 1.0:
                errors.append("%s[%d]: bandwidth_after mismatch "
                              "(%s vs %d)" % (name, pi,
                                              st["bandwidth_after_bytes"],
                                              bw_after))

            # 6. Structural checks on the output.
            if oacc["componentType"] not in (U16, U32):
                errors.append("%s[%d]: illegal index component type"
                              % (name, pi))
            if oacc["componentType"] == U16 and len(out_corners) > 65536:
                errors.append("%s[%d]: uint16 indices overflow" % (name, pi))
            if out_misses > in_misses:
                errors.append("%s[%d]: optimization increased cache misses"
                              % (name, pi))

            print("  %s prim[%d]: verts %d->%d  idx %s->%d  tris %d  "
                  "reordered %d (%.1f%%)  misses %d->%d  BW %d->%d B"
                  % (name, pi, len(in_corners), len(out_corners),
                     exp_idx_before, len(out_idx), len(in_tris),
                     st["triangles_reordered"],
                     st["triangle_order_change_pct"],
                     in_misses, out_misses, bw_before, bw_after))

    # Per-model aggregate; global totals are cross-checked in main().
    saving = 100.0 * (total_before - total_after) / max(total_before, 1)
    return errors, saving, total_before, total_after


def check_glb_structure(path, gltf, bin_data):
    errors = []
    if len(bin_data) % 4:
        errors.append("%s: BIN chunk not 4-byte aligned" % path)
    for i, view in enumerate(gltf["bufferViews"]):
        off = view.get("byteOffset", 0)
        if off % 4:
            errors.append("%s: bufferView %d offset unaligned" % (path, i))
        if view["buffer"] != 0:
            errors.append("%s: bufferView %d references unknown buffer"
                          % (path, i))
        if off + view["byteLength"] > len(bin_data):
            errors.append("%s: bufferView %d out of bounds" % (path, i))
    for i, acc in enumerate(gltf["accessors"]):
        if acc["bufferView"] >= len(gltf["bufferViews"]):
            errors.append("%s: accessor %d dangling bufferView" % (path, i))
    return errors


def main():
    if len(sys.argv) != 4:
        sys.exit("usage: python3 -I verify.py <samples_dir> <out_dir> "
                 "<stats.json>")
    samples_dir, out_dir, stats_path = sys.argv[1:]
    with open(stats_path, "r") as f:
        stats = json.load(f)

    models = ["high_dup", "low_dup", "morph", "multi_prim"]
    all_errors = []
    sum_before = sum_after = 0
    for name in models:
        print("[%s]" % name)
        in_gltf, in_bin = read_glb("%s/%s.glb" % (samples_dir, name))
        out_gltf, out_bin = read_glb("%s/%s.opt.glb" % (out_dir, name))
        all_errors += check_glb_structure(name, out_gltf, out_bin)
        errs, saving, before, after = check_model(
            name, in_gltf, in_bin, out_gltf, out_bin, stats)
        all_errors += errs
        sum_before += before
        sum_after += after
        print("    bandwidth saving: %.2f%%\n" % saving)

    total_saving = 100.0 * (sum_before - sum_after) / max(sum_before, 1)
    print("=" * 68)
    print("aggregate bandwidth: %d B -> %d B, saving %.2f%%"
          % (sum_before, sum_after, total_saving))
    print("reported aggregate:   %s B -> %s B, saving %s%%"
          % (stats["totals"]["bandwidth_before_bytes"],
             stats["totals"]["bandwidth_after_bytes"],
             stats["totals"]["bandwidth_saving_pct"]))

    rep_before = stats["totals"]["bandwidth_before_bytes"]
    rep_after = stats["totals"]["bandwidth_after_bytes"]
    if abs(sum_before - rep_before) > 1.0:
        all_errors.append("aggregate bandwidth_before mismatch: %d vs %s"
                          % (sum_before, rep_before))
    if abs(sum_after - rep_after) > 1.0:
        all_errors.append("aggregate bandwidth_after mismatch: %d vs %s"
                          % (sum_after, rep_after))

    if all_errors:
        print("\nFAILURES:")
        for e in all_errors:
            print("  - " + e)
        sys.exit(1)
    if total_saving < 20.0:
        print("\nFAIL: aggregate bandwidth saving %.2f%% < 20%%"
              % total_saving)
        sys.exit(1)
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
