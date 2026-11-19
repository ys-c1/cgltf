#!/usr/bin/env python3
"""Generate four small GLB test models for the mesh optimizer.

Pure standard library (struct/json/os) - no third-party packages, no
network access. Run with: python3 -I make_samples.py <out_dir>

Models:
  high_dup.glb  - one non-indexed mesh: a cube built from 24 split
                  vertices but every position is duplicated across 3
                  corners (36 corners, 12 triangles). High duplicate rate.
  low_dup.glb   - one indexed mesh of a jagged surface where nearly all
                  vertices are unique; little to dedup.
  morph.glb     - one indexed mesh with POSITION/NORMAL/TEXCOORD_0 plus
                  two morph targets (POSITION + NORMAL deltas).
  multi_prim.glb- one mesh with two primitives (different materials are
                  emulated by primitive grouping), one non-indexed and
                  one indexed.
"""

import json
import os
import struct
import sys


class GlbBuilder:
    """Minimal interleaved-attribute GLB writer.

    Each primitive stores one interleaved vertex buffer view plus, when
    indexed, a separate index buffer view. Component types are float32
    for vertex data and uint16 for indices.
    """

    def __init__(self):
        self.buffer = bytearray()
        self.buffer_views = []
        self.accessors = []
        self.meshes = []
        self.materials = []

    def _align(self):
        while len(self.buffer) % 4:
            self.buffer.append(0)

    def add_view(self, raw, target=None, stride=None):
        self._align()
        offset = len(self.buffer)
        self.buffer += raw
        view = {"buffer": 0, "byteOffset": offset, "byteLength": len(raw)}
        if stride is not None:
            view["byteStride"] = stride
        if target is not None:
            view["target"] = target
        self.buffer_views.append(view)
        return len(self.buffer_views) - 1

    def add_interleaved(self, rows, attr_specs, target=None):
        """Write one interleaved view.

        rows: list of tuples holding every attribute's components back to
        back, e.g. (px,py,pz, nx,ny,nz, u,v).
        attr_specs: list of (comp_type, acc_type, comps, byte_offset).
        Returns the list of accessor indices, one per attribute.
        """
        comps_total = sum(s[2] for s in attr_specs)
        stride = comps_total * 4  # all float32 in these samples
        assert all(len(r) == comps_total for r in rows), "row width mismatch"
        raw = struct.pack("<%df" % (len(rows) * comps_total),
                          *[v for r in rows for v in r])
        view = self.add_view(raw, target=target, stride=stride)
        out = []
        for comp_type, acc_type, comps, byte_offset in attr_specs:
            out.append(self.add_accessor(view, comp_type, acc_type,
                                         len(rows), byte_offset))
        return out

    def add_accessor(self, view, comp_type, acc_type, count, offset=0,
                     sparse_min=None, sparse_max=None):
        acc = {
            "bufferView": view,
            "componentType": comp_type,
            "type": acc_type,
            "count": count,
            "byteOffset": offset,
        }
        if sparse_min is not None:
            acc["min"] = sparse_min
        if sparse_max is not None:
            acc["max"] = sparse_max
        self.accessors.append(acc)
        return len(self.accessors) - 1

    def add_material(self, name):
        self.materials.append({"name": name})
        return len(self.materials) - 1

    def add_mesh(self, name, primitives):
        self.meshes.append({"name": name, "primitives": primitives})

    def write(self, path):
        self._align()
        gltf = {
            "asset": {"version": "2.0", "generator": "make_samples.py"},
            "buffers": [{"byteLength": len(self.buffer)}],
            "bufferViews": self.buffer_views,
            "accessors": self.accessors,
            "meshes": self.meshes,
            "materials": self.materials,
        }
        js = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
        while len(js) % 4:
            js += b" "
        bin_data = bytes(self.buffer)
        while len(bin_data) % 4:
            bin_data += b"\x00"
        total = 12 + 8 + len(js) + 8 + len(bin_data)
        out = bytearray()
        out += struct.pack("<III", 0x46546C67, 2, total)
        out += struct.pack("<II", len(js), 0x4E4F534A)
        out += js
        out += struct.pack("<II", len(bin_data), 0x004E4942)
        out += bin_data
        with open(path, "wb") as f:
            f.write(out)


F32 = 5126
U16 = 5123
ARRAY_BUFFER = 34962
ELEMENT_ARRAY_BUFFER = 34963


def pack_vertices(rows, stride_comps):
    """rows: list of tuples with stride_comps floats each."""
    assert all(len(r) == stride_comps for r in rows)
    return struct.pack("<%df" % (len(rows) * stride_comps),
                       *[v for r in rows for v in r])


def pack_u16(values):
    return struct.pack("<%dH" % len(values), *values)


def pn_uv(p, n, uv):
    return (p[0], p[1], p[2], n[0], n[1], n[2], uv[0], uv[1])


STRIDE = 8  # pos3 + normal3 + uv2


def build_high_dup(b):
    """Cube, NON-indexed: 12 triangles * 3 corners = 36 corners.

    Only 8 distinct (position, normal, uv) combinations exist so the
    dedup pass collapses 36 corners to 8 vertices.
    """
    corners = []
    # 8 unique corner signatures keyed by (position, normal-ish, uv).
    sig = []
    for x in (-1.0, 1.0):
        for y in (-1.0, 1.0):
            for z in (-1.0, 1.0):
                sig.append(pn_uv((x, y, z), (x, y, z),
                                 (0.0 if x < 0 else 1.0,
                                  0.0 if y < 0 else 1.0)))
    # 12 triangles each referencing three of the 8 signatures.
    faces = [
        (0, 1, 3), (0, 3, 2),
        (4, 6, 7), (4, 7, 5),
        (0, 4, 5), (0, 5, 1),
        (2, 3, 7), (2, 7, 6),
        (0, 2, 6), (0, 6, 4),
        (1, 5, 7), (1, 7, 3),
    ]
    for f in faces:
        for idx in f:
            corners.append(sig[idx])
    pos_v, nrm_v, uv_v = b.add_interleaved(
        corners,
        [(F32, "VEC3", 3, 0),
         (F32, "VEC3", 3, 12),
         (F32, "VEC2", 2, 24)],
        target=ARRAY_BUFFER)
    b.add_mesh("high_dup", [{
        "attributes": {"POSITION": pos_v, "NORMAL": nrm_v, "TEXCOORD_0": uv_v},
        "mode": 4,
    }])


def build_low_dup(b):
    """Indexed jagged grid: 5x5 = 25 unique vertices, 32 triangles.

    Heights are distinct per vertex so nothing collapses; duplicate rate
    is essentially zero. Indices are deliberately shuffled to defeat
    cache locality.
    """
    verts = []
    for y in range(5):
        for x in range(5):
            h = float(x * 7 + y * 13) * 0.01
            verts.append(pn_uv((float(x), float(y), h),
                               (0.0, 0.0, 1.0),
                               (x / 4.0, y / 4.0)))
    tris = []
    for y in range(4):
        for x in range(4):
            a = y * 5 + x
            tris.append((a, a + 5, a + 1))
            tris.append((a + 1, a + 5, a + 6))
    # Deterministic shuffle of triangle order.
    order = list(range(len(tris)))
    order = order[::3] + order[1::3] + order[2::3]
    tris = [tris[i] for i in order]
    indices = [v for t in tris for v in t]

    pos_v, nrm_v, uv_v = b.add_interleaved(
        verts,
        [(F32, "VEC3", 3, 0),
         (F32, "VEC3", 3, 12),
         (F32, "VEC2", 2, 24)],
        target=ARRAY_BUFFER)
    iview = b.add_view(pack_u16(indices), ELEMENT_ARRAY_BUFFER)
    idx_v = b.add_accessor(iview, U16, "SCALAR", len(indices))
    b.add_mesh("low_dup", [{
        "attributes": {"POSITION": pos_v, "NORMAL": nrm_v, "TEXCOORD_0": uv_v},
        "indices": idx_v,
        "mode": 4,
    }])


def build_morph(b):
    """Indexed quad split into 2 triangles with duplicated corners that
    dedup must collapse, plus two morph targets (POSITION/NORMAL)."""
    # 6 corners, 4 unique signatures.
    base = [
        pn_uv((0, 0, 0), (0, 0, 1), (0, 0)),  # 0
        pn_uv((0, 1, 0), (0, 0, 1), (0, 1)),  # 1
        pn_uv((1, 0, 0), (0, 0, 1), (1, 0)),  # 2
        pn_uv((1, 1, 0), (0, 0, 1), (1, 1)),  # 3
    ]
    corners = [base[0], base[2], base[1], base[1], base[2], base[3]]
    indices = [0, 1, 2, 2, 1, 3]

    # Morph target deltas, one entry per corner (matching POSITION count).
    t1_pos = [(0.0, 0.0, 0.5)] * 6
    t1_nrm = [(1.0, 0.0, 0.0)] * 6
    t2_pos = [(0.0, 0.5, 0.0)] * 6
    t2_nrm = [(0.0, 1.0, 0.0)] * 6

    pos_v, nrm_v, uv_v = b.add_interleaved(
        corners,
        [(F32, "VEC3", 3, 0),
         (F32, "VEC3", 3, 12),
         (F32, "VEC2", 2, 24)],
        target=ARRAY_BUFFER)

    t1p_view = b.add_view(pack_vertices(t1_pos, 3), ARRAY_BUFFER)
    t1p = b.add_accessor(t1p_view, F32, "VEC3", 6)
    t1n_view = b.add_view(pack_vertices(t1_nrm, 3), ARRAY_BUFFER)
    t1n = b.add_accessor(t1n_view, F32, "VEC3", 6)
    t2p_view = b.add_view(pack_vertices(t2_pos, 3), ARRAY_BUFFER)
    t2p = b.add_accessor(t2p_view, F32, "VEC3", 6)
    t2n_view = b.add_view(pack_vertices(t2_nrm, 3), ARRAY_BUFFER)
    t2n = b.add_accessor(t2n_view, F32, "VEC3", 6)

    iview = b.add_view(pack_u16(indices), ELEMENT_ARRAY_BUFFER)
    idx_v = b.add_accessor(iview, U16, "SCALAR", 6)

    b.add_mesh("morph", [{
        "attributes": {"POSITION": pos_v, "NORMAL": nrm_v, "TEXCOORD_0": uv_v},
        "indices": idx_v,
        "mode": 4,
        "targets": [
            {"POSITION": t1p, "NORMAL": t1n},
            {"POSITION": t2p, "NORMAL": t2n},
        ],
    }])
    # Mesh-level morph weights.
    b.meshes[-1]["weights"] = [0.5, 0.5]


def build_multi_prim(b):
    """One mesh, two primitives:
      prim 0: non-indexed triangle pair with duplicated corners (6 -> 3)
      prim 1: indexed small quad (4 verts, 2 tris)
    """
    mat0 = b.add_material("mat_a")
    mat1 = b.add_material("mat_b")

    # Prim 0: 6 corners, 3 unique signatures.
    a = pn_uv((0, 0, 0), (0, 0, 1), (0, 0))
    c = pn_uv((1, 0, 0), (0, 0, 1), (1, 0))
    d = pn_uv((0, 1, 0), (0, 0, 1), (0, 1))
    corners = [a, c, d, a, d, c]
    p0p, p0n, p0u = b.add_interleaved(
        corners,
        [(F32, "VEC3", 3, 0),
         (F32, "VEC3", 3, 12),
         (F32, "VEC2", 2, 24)],
        target=ARRAY_BUFFER)

    # Prim 1: 4 unique vertices, indexed.
    q = [
        pn_uv((2, 0, 0), (0, 0, 1), (0, 0)),
        pn_uv((3, 0, 0), (0, 0, 1), (1, 0)),
        pn_uv((2, 1, 0), (0, 0, 1), (0, 1)),
        pn_uv((3, 1, 0), (0, 0, 1), (1, 1)),
    ]
    idx = [0, 2, 1, 1, 2, 3]
    p1p, p1n, p1u = b.add_interleaved(
        q,
        [(F32, "VEC3", 3, 0),
         (F32, "VEC3", 3, 12),
         (F32, "VEC2", 2, 24)],
        target=ARRAY_BUFFER)
    iview = b.add_view(pack_u16(idx), ELEMENT_ARRAY_BUFFER)
    p1i = b.add_accessor(iview, U16, "SCALAR", 6)

    b.add_mesh("multi_prim", [
        {
            "attributes": {"POSITION": p0p, "NORMAL": p0n, "TEXCOORD_0": p0u},
            "mode": 4,
            "material": mat0,
        },
        {
            "attributes": {"POSITION": p1p, "NORMAL": p1n, "TEXCOORD_0": p1u},
            "indices": p1i,
            "mode": 4,
            "material": mat1,
        },
    ])


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: python3 -I make_samples.py <out_dir>")
    out_dir = sys.argv[1]
    os.makedirs(out_dir, exist_ok=True)

    builders = {
        "high_dup.glb": build_high_dup,
        "low_dup.glb": build_low_dup,
        "morph.glb": build_morph,
        "multi_prim.glb": build_multi_prim,
    }
    for name, fn in builders.items():
        b = GlbBuilder()
        fn(b)
        path = os.path.join(out_dir, name)
        b.write(path)
        print("wrote %s (%d bytes)" % (path, os.path.getsize(path)))


if __name__ == "__main__":
    main()
