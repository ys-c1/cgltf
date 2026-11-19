#!/usr/bin/env bash
# End-to-end mesh optimizer test: build, generate 4 small GLB samples,
# optimize each, verify reconstruction correctness and bandwidth savings,
# then run the 100k-vertex performance benchmark.
#
# Uses only the repository's bundled cgltf sources, gcc and python3 stdlib.
# No network access, no third-party packages.
set -euo pipefail

cd "$(dirname "$0")"
ROOT="$(pwd)"
SAMPLES="$ROOT/samples"
OUT="$ROOT/out"
mkdir -p "$SAMPLES" "$OUT"

echo "== building meshopt =="
gcc -std=c99 -O3 -Wall -Wextra -pedantic "$ROOT/meshopt.c" -o "$ROOT/meshopt" -lm

echo "== generating sample models =="
python3 -I "$ROOT/make_samples.py" "$SAMPLES"

echo
echo "== optimizing four models =="
: > "$OUT/stats.ndjson"
for name in high_dup low_dup morph multi_prim; do
    echo "-- $name"
    "$ROOT/meshopt" "$SAMPLES/$name.glb" "$OUT/$name.opt.glb" > "$OUT/$name.stats.json"
done

# Merge per-file stats into one document for the verifier.
python3 -I - "$OUT" <<'PY'
import json, os, sys
out = sys.argv[1]
prims, tb, ta, vb, va, ib, ia = [], 0.0, 0.0, 0, 0, 0, 0
for name in ("high_dup", "low_dup", "morph", "multi_prim"):
    with open(os.path.join(out, name + ".stats.json")) as f:
        doc = json.load(f)
    prims += doc["primitives"]
    t = doc["totals"]
    tb += t["bandwidth_before_bytes"]
    ta += t["bandwidth_after_bytes"]
    vb += t["vertices_before"]; va += t["vertices_after"]
    ib += t["indices_before"];  ia += t["indices_after"]
merged = {
    "primitives": prims,
    "totals": {
        "vertices_before": vb, "vertices_after": va,
        "indices_before": ib, "indices_after": ia,
        "bandwidth_before_bytes": tb, "bandwidth_after_bytes": ta,
        "bandwidth_saving_pct": 100.0 * (tb - ta) / tb,
    },
}
with open(os.path.join(out, "stats.json"), "w") as f:
    json.dump(merged, f, indent=2)
PY

echo
echo "== verifying correctness =="
python3 -I "$ROOT/verify.py" "$SAMPLES" "$OUT" "$OUT/stats.json"

echo
echo "== cgltf roundtrip parse of every output GLB =="
gcc -std=c99 -O2 -Wall -Wextra -I"$ROOT/.." "$ROOT/roundtrip.c" -o "$ROOT/roundtrip" -lm
for name in high_dup low_dup morph multi_prim; do
    "$ROOT/roundtrip" "$OUT/$name.opt.glb" | tail -1
done

echo
echo "== performance benchmark (100k vertices) =="
"$ROOT/meshopt" "$SAMPLES/high_dup.glb" "$OUT/bench.glb" --bench > /dev/null

echo
echo "== per-model stats =="
cat "$OUT/stats.json"
