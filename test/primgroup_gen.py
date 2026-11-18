#!/usr/bin/env python3
"""Generate four small glTF samples for the primitive-grouping test.

Models (all single-mesh, embedded base64 buffer, tightly packed accessors):
  single_material.gltf  - 8 primitives, 1 material, uint16 indices
  multi_material.gltf   - 120 primitives, 4 materials, uint32 indices
  shared_indices.gltf   - 6 primitives, 2 materials, prims in a group share one index accessor
  morph.gltf            - 8 primitives, 2 materials, 2 morph targets per prim

Usage: primgroup_gen.py <out_dir>
"""
import base64
import json
import os
import random
import struct
import sys

CT = {"i8": 5120, "u8": 5121, "i16": 5122, "u16": 5123, "u32": 5125, "f32": 5126}


class Builder:
    def __init__(self):
        self.blob = bytearray()
        self.views = []
        self.accessors = []

    def add_accessor(self, data, comp, count, acc_type):
        while len(self.blob) % 4:
            self.blob.append(0)
        off = len(self.blob)
        self.blob += data
        self.views.append({"buffer": 0, "byteOffset": off, "byteLength": len(data)})
        self.accessors.append(
            {"bufferView": len(self.views) - 1, "componentType": CT[comp],
             "count": count, "type": acc_type})
        return len(self.accessors) - 1


def rand_vertices(rng, vcount):
    pos = b"".join(struct.pack("<3f", rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1))
                   for _ in range(vcount))
    nrm = b"".join(struct.pack("<3f", 0.0, 1.0, 0.0) for _ in range(vcount))
    uv = b"".join(struct.pack("<2f", rng.random(), rng.random()) for _ in range(vcount))
    return pos, nrm, uv


def rand_indices(rng, icount, vcount, comp):
    fmt = "<%dH" % icount if comp == "u16" else "<%dI" % icount
    return struct.pack(fmt, *[rng.randrange(vcount) for _ in range(icount)])


def make_prim(builder, rng, vcount, icount, index_comp, morph_targets=0, shared_index_acc=None):
    pos, nrm, uv = rand_vertices(rng, vcount)
    prim = {"attributes": {
        "POSITION": builder.add_accessor(pos, "f32", vcount, "VEC3"),
        "NORMAL": builder.add_accessor(nrm, "f32", vcount, "VEC3"),
        "TEXCOORD_0": builder.add_accessor(uv, "f32", vcount, "VEC2"),
    }}
    if shared_index_acc is not None:
        prim["indices"] = shared_index_acc
    else:
        prim["indices"] = builder.add_accessor(
            rand_indices(rng, icount, vcount, index_comp), index_comp, icount, "SCALAR")
    if morph_targets:
        targets = []
        for _ in range(morph_targets):
            dpos = b"".join(struct.pack("<3f", rng.uniform(-0.1, 0.1), rng.uniform(-0.1, 0.1),
                                        rng.uniform(-0.1, 0.1)) for _ in range(vcount))
            dnrm = b"".join(struct.pack("<3f", 0.0, 0.0, 0.0) for _ in range(vcount))
            targets.append({
                "POSITION": builder.add_accessor(dpos, "f32", vcount, "VEC3"),
                "NORMAL": builder.add_accessor(dnrm, "f32", vcount, "VEC3"),
            })
        prim["targets"] = targets
    prim["mode"] = 4
    return prim


def emit(builder, prims, materials, path):
    doc = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": prims}],
        "materials": materials,
        "accessors": builder.accessors,
        "bufferViews": builder.views,
        "buffers": [{
            "byteLength": len(builder.blob),
            "uri": "data:application/octet-stream;base64," + base64.b64encode(bytes(builder.blob)).decode(),
        }],
    }
    with open(path, "w") as f:
        json.dump(doc, f)


def make_materials(n):
    return [{"pbrMetallicRoughness": {"baseColorFactor": [i / max(n - 1, 1), 0.5, 0.5, 1.0]}}
            for i in range(n)]


def gen_single_material(out):
    rng, b = random.Random(101), Builder()
    prims = [make_prim(b, rng, vcount=4 + i, icount=6 + 2 * i, index_comp="u16") for i in range(8)]
    emit(b, prims, make_materials(1), os.path.join(out, "single_material.gltf"))


def gen_multi_material(out):
    rng, b = random.Random(202), Builder()
    prims = []
    for i in range(120):
        p = make_prim(b, rng, vcount=4 + i % 6, icount=6 + 2 * (i % 5), index_comp="u32")
        p["material"] = i % 4
        prims.append(p)
    emit(b, prims, make_materials(4), os.path.join(out, "multi_material.gltf"))


def gen_shared_indices(out):
    rng, b = random.Random(303), Builder()
    prims = []
    for mat in range(2):
        # three primitives within one material share a single index accessor
        shared = b.add_accessor(rand_indices(rng, 12, 6, "u16"), "u16", 12, "SCALAR")
        for j in range(3):
            p = make_prim(b, rng, vcount=6 + j, icount=12, index_comp="u16",
                          shared_index_acc=shared)
            p["material"] = mat
            prims.append(p)
    emit(b, prims, make_materials(2), os.path.join(out, "shared_indices.gltf"))


def gen_morph(out):
    rng, b = random.Random(404), Builder()
    prims = []
    for i in range(8):
        p = make_prim(b, rng, vcount=5 + i % 4, icount=9, index_comp="u16", morph_targets=2)
        p["material"] = i % 2
        prims.append(p)
    emit(b, prims, make_materials(2), os.path.join(out, "morph.gltf"))


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    gen_single_material(out)
    gen_multi_material(out)
    gen_shared_indices(out)
    gen_morph(out)
    for name in ("single_material", "multi_material", "shared_indices", "morph"):
        print("generated %s.gltf (%d bytes)" % (name, os.path.getsize(os.path.join(out, name + ".gltf"))))


if __name__ == "__main__":
    main()
