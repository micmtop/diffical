#!/usr/bin/env bash
# Flag parity matrix: Python reference vs Rust, one flag at a time.
#
# Usage:
#   tools/flag_parity.sh <request.json> <assembler.onnx> <vocoder.onnx> \
#       [hnsep.onnx] [long-note-request.json]
#
# The Python goldens are generated on first run (flag_golden/<case>/); delete a
# case directory to force a re-render. The long-note request optionally exercises
# `He` on a sustained note (the main request has none, where He is a no-op).
set -euo pipefail
# Compare against the Python reference, which separates whole files.
export DIFFICAL_HNSEP_WHOLE=1

REQ=${1:?usage: flag_parity.sh <request> <assembler> <vocoder> [hnsep] [long-request]}
ASM=${2:?missing assembler}
VOC=${3:?missing vocoder}
HNSEP=${4:-}
LONG_REQ=${5:-}

ROOT=$(cd "$(dirname "$0")/.." && pwd)
CLI="$ROOT/target/release/diffical-cli"
GOLD="$ROOT/flag_golden"
mkdir -p "$GOLD"
[ -x "$CLI" ] || { echo "build first: cargo build --release"; exit 1; }

hnsep_arg=()
if [ -n "$HNSEP" ]; then
  hnsep_arg=(--hnsep "$HNSEP")
fi

dump() { # dump <request> <outdir> <flags>
  if [ -f "$2/samples.f32" ] && [ "$(cat "$2/.flags" 2>/dev/null)" = "$3" ]; then
    return 0
  fi
  rm -rf "$2"
  mkdir -p "$2"
  printf '%s' "$3" > "$2/.flags"
  python "$ROOT/tools/dump_flag_golden.py" "$1" "$2" --inject-flags "$3" >/dev/null
}

case_run() { # case_run <request> <case> <flags> [extra flag-parity args...]
  local req=$1 name=$2 flags=$3
  shift 3
  echo "=== $name ($flags) ==="
  dump "$req" "$GOLD/$name" "$flags"
  "$CLI" flag-parity "$req" "$GOLD/$name" "$ASM" "$VOC" "${hnsep_arg[@]}" \
    --inject-flags "$flags" "$@" | tail -4
}

# Python implements g/Hb/Hv/HG; these must match mel and wav.
case_run "$REQ" g100 "g100"
case_run "$REQ" Hb200 "Hb200"
case_run "$REQ" Hv120 "Hv120"
case_run "$REQ" HG50 "HG50"
# t / A only exist in Rust; the mel must be identical, the wav deliberately not.
case_run "$REQ" t-50 "t-50" --allow-wav-diff
case_run "$REQ" A100 "A100" --allow-wav-diff
# Ht is a real tilt in Rust; the Python filter factor clips to zero, so the
# reference render is unchanged and only self-consistency can be checked.
case_run "$REQ" Ht30 "Ht30" --allow-mel-diff --allow-wav-diff
# He is inert on the short-note request (no hold path) -> full parity expected.
case_run "$REQ" He "He"
# Combined Python-implemented flags must still match both mel and wav.
case_run "$REQ" combo "Hb250Hv90g100"
# Combined Rust-only extensions (pitch + growl): mel must match, wav differs.
case_run "$REQ" combo_rust "t20HG40" --allow-wav-diff

if [ -n "$LONG_REQ" ]; then
  # Long note: He replaces the frozen hold. Python ignores He, so both the mel
  # and the wav intentionally differ; the printed diff is the He effect.
  echo "=== He-long ($LONG_REQ) ==="
  dump "$LONG_REQ" "$GOLD/He_long" "He"
  "$CLI" flag-parity "$LONG_REQ" "$GOLD/He_long" "$ASM" "$VOC" "${hnsep_arg[@]}" \
    --inject-flags He --allow-mel-diff --allow-wav-diff | tail -4
fi

echo "flag parity matrix PASS"
