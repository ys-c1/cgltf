#!/bin/sh
# One-shot primitive-grouping test: generate samples, build, run.
# Usage: test/run_prim_group.sh
set -e
cd "$(dirname "$0")/.."
OUT=artifacts/primgroup
mkdir -p "$OUT"
python3 -I test/primgroup_gen.py "$OUT"
gcc -O2 -Wall -Wextra -I. test/test_prim_group.c -o "$OUT/test_prim_group"
"$OUT/test_prim_group" "$OUT"
