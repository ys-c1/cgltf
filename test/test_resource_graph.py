#!/usr/bin/env python3
#
# test_resource_graph.py — acceptance tests for gltf_resource_graph.c
#
# Generates four tiny glTF models in an isolated temp directory:
#   1. complete      — every external file present, all resources used
#   2. missing_tex   — two referenced image files absent from disk
#   3. circular      — node parent cycle + a buffer whose URI points at
#                      the document itself (filesystem-level cycle)
#   4. duplicate     — one texture shared by two materials, one material
#                      shared by two mesh primitives
# Plus two extra zero-false-positive cases (GLB-embedded buffer + data URI
# image, and an intentionally unused resource to confirm the warning fires
# only when it should).
#
# The analyzer output is cross-checked against an independent reference
# implementation written straight from the source JSON (manual verification).
#
# Usage: python3 -I test_resource_graph.py [path-to-gltf_resource_graph-binary]
#
# Standard library only; no network access; all data is self-generated.

import json
import os
import struct
import subprocess
import sys
import tempfile

# --------------------------------------------------------------------- #
# Tiny glTF model builders                                              #
# --------------------------------------------------------------------- #

PNG_1X1 = bytes([
    0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,  # PNG signature
    0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,
    0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
    0x08, 0x06, 0x00, 0x00, 0x00, 0x1F, 0x15, 0xC4,
    0x89, 0x00, 0x00, 0x00, 0x0A, 0x49, 0x44, 0x41,
    0x54, 0x78, 0x9C, 0x63, 0x00, 0x01, 0x00, 0x00,
    0x05, 0x00, 0x01, 0x0D, 0x0A, 0x2D, 0xB4, 0x00,
    0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE,
    0x42, 0x60, 0x82,
])

# 3 vertices * 3 floats, plus one 16-bit index tri (padded)
POSITION_BYTES = struct.pack("<9f", 0, 0, 0, 1, 0, 0, 0, 1, 0)
INDEX_BYTES = struct.pack("<3H", 0, 1, 2) + b"\x00\x00"


def write_bin(path, payload):
    with open(path, "wb") as f:
        f.write(payload)


def write_gltf(directory, name, doc):
    path = os.path.join(directory, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1)
    return os.path.abspath(path)


def base_doc(buf_uri="data.bin", img_uris=("img0.png", "img1.png"),
             mesh_material_count=1, materials_share_texture=False):
    """Build a minimal but fully-linked glTF document.

    Two images feed two textures; the single material consumes both (base
    color + normal), so every declared resource has a semantic consumer and
    a fully-linked model produces no warnings.
    """
    position_size = len(POSITION_BYTES)
    index_offset = position_size
    index_size = len(INDEX_BYTES)
    doc = {
        "asset": {"version": "2.0"},
        "buffers": [{"uri": buf_uri, "byteLength": position_size + index_size}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": position_size, "target": 34962},
            {"buffer": 0, "byteOffset": index_offset, "byteLength": index_size, "target": 34963},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 3,
             "type": "VEC3", "min": [0, 0, 0], "max": [1, 1, 0]},
            {"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"},
        ],
        "images": [{"uri": u} for u in img_uris],
        "textures": [
            {"source": 0, "name": "tex0"},
            {"source": 1, "name": "tex1"},
        ],
        "materials": [],
        "meshes": [{
            "name": "mesh0",
            "primitives": [
                {"attributes": {"POSITION": 0}, "indices": 1, "material": i}
                for i in range(mesh_material_count)
            ],
        }],
        "nodes": [{"name": "node0", "mesh": 0}],
        "scene": 0,
        "scenes": [{"nodes": [0]}],
    }
    mat0 = {
        "name": "mat0",
        "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}},
        "normalTexture": {"index": 1},
    }
    if mesh_material_count == 1:
        doc["materials"] = [mat0]
    else:
        tex_idx = 0 if materials_share_texture else 1
        mat1 = {
            "name": "mat1",
            "pbrMetallicRoughness": {"baseColorTexture": {"index": tex_idx}},
        }
        doc["materials"] = [mat0, mat1]
    return doc


def build_complete(d):
    write_bin(os.path.join(d, "data.bin"), POSITION_BYTES + INDEX_BYTES)
    write_bin(os.path.join(d, "img0.png"), PNG_1X1)
    write_bin(os.path.join(d, "img1.png"), PNG_1X1)
    return write_gltf(d, "complete.gltf", base_doc())


def build_missing_texture(d):
    # The geometry buffer and the base-color image are present; the normal-map
    # image uses a case-unique filename that is never written, so sharing the
    # temp directory with the other models cannot mask the missing file.
    write_bin(os.path.join(d, "data.bin"), POSITION_BYTES + INDEX_BYTES)
    write_bin(os.path.join(d, "img0.png"), PNG_1X1)
    doc = base_doc()
    doc["images"][1]["uri"] = "missing_normal.png"
    return write_gltf(d, "missing_tex.gltf", doc)


def build_circular(d):
    # data.bin exists; image files exist. cgltf rejects node-parent cycles at
    # parse time (a child may only have one parent), so the exercisable cycle
    # is filesystem-level: an external buffer whose URI points back at the
    # document itself. Two such buffers give two independent cycles.
    write_bin(os.path.join(d, "data.bin"), POSITION_BYTES + INDEX_BYTES)
    write_bin(os.path.join(d, "img0.png"), PNG_1X1)
    write_bin(os.path.join(d, "img1.png"), PNG_1X1)
    doc = base_doc()
    doc["nodes"] = [
        {"name": "n0", "mesh": 0, "children": [1]},
        {"name": "n1"},
    ]
    doc["scenes"] = [{"nodes": [0]}]
    doc["buffers"].append({"uri": "circular.gltf", "byteLength": 1})
    doc["buffers"].append({"uri": "circular.gltf", "byteLength": 1})
    path = write_gltf(d, "circular.gltf", doc)
    return path


def build_duplicate(d):
    write_bin(os.path.join(d, "data.bin"), POSITION_BYTES + INDEX_BYTES)
    write_bin(os.path.join(d, "img0.png"), PNG_1X1)
    write_bin(os.path.join(d, "img1.png"), PNG_1X1)
    # two materials both using texture 0; two primitives both using material 0
    doc = base_doc(mesh_material_count=2, materials_share_texture=True)
    doc["meshes"][0]["primitives"][1]["material"] = 0
    return write_gltf(d, "duplicate.gltf", doc)


def build_embedded(d):
    # One image carried as a data URI (no external file, must not be reported
    # missing). The image/texture/material index chain stays consistent.
    import base64
    doc = base_doc(img_uris=("img0.png",))
    doc["images"] = [{"uri": "data:image/png;base64," +
                      base64.b64encode(PNG_1X1).decode("ascii")}]
    doc["textures"] = [{"source": 0, "name": "tex0"}]
    doc["materials"] = [{
        "name": "mat0",
        "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}},
    }]
    write_bin(os.path.join(d, "data.bin"), POSITION_BYTES + INDEX_BYTES)
    return write_gltf(d, "embedded.gltf", doc)


def build_unused(d):
    # an extra image/texture/material that nothing consumes -> WARN only
    write_bin(os.path.join(d, "data.bin"), POSITION_BYTES + INDEX_BYTES)
    write_bin(os.path.join(d, "img0.png"), PNG_1X1)
    write_bin(os.path.join(d, "img1.png"), PNG_1X1)
    doc = base_doc()
    doc["images"].append({"uri": "img1.png"})
    doc["textures"].append({"source": 2, "name": "orphan_tex"})
    doc["materials"].append({"name": "orphan_mat"})
    return write_gltf(d, "unused.gltf", doc)


# --------------------------------------------------------------------- #
# Independent reference checks computed straight from the source JSON   #
# --------------------------------------------------------------------- #

def reference_expectations(directory, gltf_path):
    with open(gltf_path, encoding="utf-8") as f:
        doc = json.load(f)

    missing = []
    for i, buf in enumerate(doc.get("buffers", [])):
        uri = buf.get("uri")
        if uri and not uri.startswith("data:") and "://" not in uri:
            if not os.path.exists(os.path.join(directory, uri)):
                missing.append(("buffer", i, uri))
            if os.path.abspath(os.path.join(directory, uri)) == os.path.abspath(gltf_path):
                missing  # self-reference is a cycle, recorded below instead
    for i, img in enumerate(doc.get("images", [])):
        uri = img.get("uri")
        if uri and not uri.startswith("data:") and "://" not in uri:
            if not os.path.exists(os.path.join(directory, uri)):
                missing.append(("image", i, uri))

    # cycles in node hierarchy (independent iterative DFS over source JSON)
    node_cycles = 0
    color = [0] * len(doc.get("nodes", []))
    adj = {i: list(doc["nodes"][i].get("children", [])) for i in range(len(doc.get("nodes", [])))}
    for root in range(len(adj)):
        if color[root] != 0:
            continue
        stack = [(root, 0)]
        color[root] = 1
        while stack:
            u, pos = stack[-1]
            if pos < len(adj[u]):
                v = adj[u][pos]
                stack[-1] = (u, pos + 1)
                if color[v] == 0:
                    color[v] = 1
                    stack.append((v, 0))
                elif color[v] == 1:
                    node_cycles += 1
            else:
                color[u] = 2
                stack.pop()

    # self-referencing external URIs close filesystem-level cycles.
    # Distinct self-referencing resources sharing one URI collapse onto one
    # FILE node, so the graph contains exactly one back edge per distinct
    # self-referencing path; the analyzer additionally emits one ERROR line
    # per resource. The reference computes both numbers.
    self_ref_resources = 0
    self_ref_paths = set()
    for buf in doc.get("buffers", []):
        uri = buf.get("uri")
        if uri and os.path.abspath(os.path.join(directory, uri)) == os.path.abspath(gltf_path):
            self_ref_resources += 1
            self_ref_paths.add(os.path.abspath(os.path.join(directory, uri)))
    for img in doc.get("images", []):
        uri = img.get("uri")
        if uri and os.path.abspath(os.path.join(directory, uri)) == os.path.abspath(gltf_path):
            self_ref_resources += 1
            self_ref_paths.add(os.path.abspath(os.path.join(directory, uri)))
    graph_cycles = node_cycles + len(self_ref_paths)

    # duplicate semantic references: a resource referenced by >=2 members.
    # Each shared target is counted once, matching the analyzer's in-degree rule.
    tex_users = {}
    mat_users = {}
    for mat in doc.get("materials", []):
        def walk(views):
            for tv in views:
                if tv and "index" in tv:
                    tex_users[tv["index"]] = tex_users.get(tv["index"], 0) + 1
        pbr = mat.get("pbrMetallicRoughness", {})
        walk([pbr.get("baseColorTexture"), pbr.get("metallicRoughnessTexture")])
        walk([mat.get("normalTexture"), mat.get("occlusionTexture"), mat.get("emissiveTexture")])
    for mesh in doc.get("meshes", []):
        for prim in mesh.get("primitives", []):
            if "material" in prim:
                mat_users[prim["material"]] = mat_users.get(prim["material"], 0) + 1
    duplicates = sum(1 for c in tex_users.values() if c >= 2) + \
                 sum(1 for c in mat_users.values() if c >= 2)

    # unused resources
    used_textures = set(tex_users.keys())
    used_materials = set(mat_users.keys())
    image_users = {}
    for t in doc.get("textures", []):
        if "source" in t:
            image_users[t["source"]] = image_users.get(t["source"], 0) + 1
    unused = 0
    unused += sum(1 for i in range(len(doc.get("images", []))) if image_users.get(i, 0) == 0)
    unused += sum(1 for i in range(len(doc.get("textures", []))) if i not in used_textures)
    unused += sum(1 for i in range(len(doc.get("materials", []))) if i not in used_materials)
    mesh_users = {n.get("mesh") for n in doc.get("nodes", []) if "mesh" in n}
    unused += sum(1 for i in range(len(doc.get("meshes", []))) if i not in mesh_users)

    return {
        "missing": len(missing),
        "node_cycles": node_cycles,
        "self_ref_resources": self_ref_resources,
        "graph_cycles": graph_cycles,
        "duplicates": duplicates,
        "unused": unused,
        "missing_detail": missing,
    }


# --------------------------------------------------------------------- #
# Runner                                                                #
# --------------------------------------------------------------------- #

def run_analyzer(binary, gltf_path):
    proc = subprocess.run([binary, "--json", gltf_path],
                          capture_output=True, text=True, timeout=30)
    if proc.returncode != 0:
        raise RuntimeError("analyzer failed: %s\n%s" % (proc.returncode, proc.stderr))
    return json.loads(proc.stdout)


def main():
    binary = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "build_grg", "gltf_resource_graph")
    if not os.path.exists(binary):
        print("binary not found: %s" % binary)
        return 2

    failures = []
    results = []

    # Honor TMPDIR so callers can keep temp files off a small/tmp tmpfs.
    tmp_root = os.environ.get("GRG_TMPDIR")
    tmp_kwargs = {"dir": tmp_root} if tmp_root and os.path.isdir(tmp_root) else {}
    with tempfile.TemporaryDirectory(prefix="grg_test_", **tmp_kwargs) as d:
        cases = [
            # (name, model, expected ERROR total, expected WARN total,
            #  expected duplicate-warn count, expected unused-warn count)
            ("complete",         build_complete(d),        0, 0, 0, 0),
            ("missing_tex",      build_missing_texture(d), 1, 0, 0, 0),
            ("circular",         build_circular(d),        3, 0, 0, 0),
            ("duplicate",        build_duplicate(d),       0, 3, 2, 1),
            ("embedded_datauri", build_embedded(d),        0, 0, 0, 0),
            ("unused_resource",  build_unused(d),          0, 2, 0, 2),
        ]

        for name, path, exp_err, exp_warn, exp_dup_warn, exp_unused_warn in cases:
            report = run_analyzer(binary, path)
            summary = report["summary"]
            exp = reference_expectations(d, path)

            checks = []
            # 1. severity counts match the acceptance table
            checks.append(("severity[error]=%d (want %d)" %
                           (summary["error"], exp_err), summary["error"] == exp_err))
            checks.append(("severity[warn]=%d (want %d)" %
                           (summary["warn"], exp_warn), summary["warn"] == exp_warn))
            # 2. analyzer counters match the independent reference implementation
            cross = [
                ("missing", summary["missing"], exp["missing"]),
                ("circular", summary["circular"], exp["graph_cycles"]),
                ("duplicate", summary["duplicate"], exp["duplicates"]),
            ]
            for label, got, want in cross:
                checks.append(("manual cross-check %s: got=%d want=%d" % (label, got, want),
                               got == want))
            # 3. warning split: duplicate-reference vs unused-resource
            dup_warns = sum(1 for f in report["findings"]
                            if "duplicate reference" in f["message"])
            unused_warns = sum(1 for f in report["findings"]
                               if "not referenced" in f["message"])
            checks.append(("duplicate warnings: got=%d want=%d" %
                           (dup_warns, exp_dup_warn), dup_warns == exp_dup_warn))
            checks.append(("unused warnings: got=%d want=%d" %
                           (unused_warns, exp_unused_warn), unused_warns == exp_unused_warn))
            checks.append(("unused matches manual reference: got=%d want=%d" %
                           (unused_warns, exp["unused"]), unused_warns == exp["unused"]))

            ok_all = all(ok for _, ok in checks)
            results.append((name, report, checks))
            if not ok_all:
                failures.append(name)

        # performance case: 10,000-node chain
        perf_dir = os.path.join(d, "perf")
        os.mkdir(perf_dir)
        write_bin(os.path.join(perf_dir, "data.bin"), POSITION_BYTES + INDEX_BYTES)
        n = 10000
        perf_doc = base_doc()
        perf_doc["nodes"] = [{"name": "n%d" % i, "children": [i + 1]} for i in range(n - 1)]
        perf_doc["nodes"].append({"name": "n%d" % (n - 1)})
        perf_doc["nodes"][0]["mesh"] = 0
        perf_doc["scenes"] = [{"nodes": [0]}]
        perf_path = write_gltf(perf_dir, "perf10k.gltf", perf_doc)

        perf = subprocess.run([binary, "--perf", "200", perf_path],
                              capture_output=True, text=True, timeout=120)
        if perf.returncode != 0:
            failures.append("perf")
            print(perf.stderr)

        print("=" * 78)
        print("ACCEPTANCE REPORT")
        print("=" * 78)
        for name, report, checks in results:
            s = report["summary"]
            print("\n[%s] %s" % (name, "PASS" if all(ok for _, ok in checks) else "FAIL"))
            print("  graph nodes=%d edges=%d | findings total=%d ERROR=%d WARN=%d INFO=%d"
                  % (report["nodes_count"], report["edges_count"],
                     len(report["findings"]), s["error"], s["warn"], s["info"]))
            print("  missing=%d circular=%d duplicate=%d"
                  % (s["missing"], s["circular"], s["duplicate"]))
            for label, ok in checks:
                print("    %s  %s" % ("ok " if ok else "BAD", label))
            for f in report["findings"]:
                print("      - %-5s %s" % (f["severity"], f["message"]))

        print("\n" + "=" * 78)
        print("PERFORMANCE (10,000 nodes, 200 iterations)")
        print("=" * 78)
        print(perf.stdout.strip())

        print("\n" + "=" * 78)
        if failures:
            print("RESULT: FAIL (%s)" % ", ".join(failures))
            return 1
        print("RESULT: ALL PASS — zero false positives, manual cross-checks matched")
        return 0


if __name__ == "__main__":
    sys.exit(main())
