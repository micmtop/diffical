#!/bin/bash
# Launch the installed OpenUtau app with the Diffical MVP backend explicitly selected.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
APP_PATH="${OPENUTAU_APP:-$ROOT/../diffical_new/mvp-openutau-osx-arm64/OpenUtau.app}"
PYTHON="${DIFFICAL_PYTHON:-/opt/anaconda3/bin/python}"

if [[ ! -f "$ROOT/backend/server.py" ]]; then
  echo "Diffical backend not found under $ROOT" >&2
  exit 1
fi
if [[ ! -x "$PYTHON" ]]; then
  echo "Python executable missing: $PYTHON (set DIFFICAL_PYTHON)" >&2
  exit 1
fi
if [[ -x "$APP_PATH/Contents/MacOS/OpenUtau" ]]; then
  APP_EXEC="$APP_PATH/Contents/MacOS/OpenUtau"
  CORE_DLL="$APP_PATH/Contents/MacOS/OpenUtau.Core.dll"
  GUI_DLL="$APP_PATH/Contents/MacOS/OpenUtau.dll"
elif [[ -x "$APP_PATH/OpenUtau" ]]; then
  APP_EXEC="$APP_PATH/OpenUtau"
  CORE_DLL="$APP_PATH/OpenUtau.Core.dll"
  GUI_DLL="$APP_PATH/OpenUtau.dll"
else
  echo "OpenUtau executable not found under: $APP_PATH" >&2
  echo "Set OPENUTAU_APP to a patched .app bundle or publish directory" >&2
  exit 1
fi
if [[ ! -f "$CORE_DLL" ]] || ! grep -a -q 'DifficalRenderer' "$CORE_DLL"; then
  echo "The selected OpenUtau app has no compiled Diffical renderer: $CORE_DLL" >&2
  echo "Build the patched OpenUtau source first (see docs/MVP.md)." >&2
  exit 1
fi

# AutoPitch has UI resources as well as Core code. A Core-only DLL swap is insufficient.
if [[ ! -f "$GUI_DLL" ]] || ! grep -a -q 'AutoPitchDialog' "$GUI_DLL" || \
   ! grep -a -q 'PrettyPitchOnnxModel' "$CORE_DLL"; then
  echo "The selected MVP lacks compiled AutoPitch GUI/Core: $APP_PATH" >&2
  echo "Rebuild the complete app (see docs/MVP.md), not just OpenUtau.Core.dll." >&2
  exit 1
fi

# Do not silently launch a package that can replay pre-handover audio caches.
"$PYTHON" - "$CORE_DLL" <<'PY'
from pathlib import Path
import sys
version = '20261018handover2'
if version.encode('utf-16le') not in Path(sys.argv[1]).read_bytes():
    raise SystemExit('MVP cache version mismatch: rebuild the complete app for ' + version)
PY

# The verified base stays external to the app; bank training remains optional.
export PRETTYPITCH_BASE_MODEL="${PRETTYPITCH_BASE_MODEL:-$ROOT/../diffical_new/models/prettypitch-base/model.onnx}"
export PRETTYPITCH_PYTHON="${PRETTYPITCH_PYTHON:-$PYTHON}"
# Optional local trainable initializer; inference ONNX is not a training checkpoint.
# Keep private weights external and never download them automatically.
export PRETTYPITCH_BASE_CHECKPOINT="${PRETTYPITCH_BASE_CHECKPOINT:-$ROOT/../workspaces/openutau-auto-pitch/artifacts/PrettyPitch-upstream/checkpoints/f0_3singer.pt}"
if [[ "$PRETTYPITCH_BASE_MODEL" != /* || ! -f "$PRETTYPITCH_BASE_MODEL" ]]; then
  echo "PrettyPitch base missing or not absolute: $PRETTYPITCH_BASE_MODEL" >&2
  echo "Choose a compatible external ONNX in the GUI, or set PRETTYPITCH_BASE_MODEL." >&2
  if [[ "${1:-}" == --check ]]; then exit 1; fi
fi

export DIFFICAL_BACKEND_PATH="$ROOT/backend"
export DIFFICAL_ENGINE="model"
export DIFFICAL_POINTER_CKPT="$ROOT/train/ckpt_prod_pilot/last.pt"
export DIFFICAL_SEAM_RAMP="${DIFFICAL_SEAM_RAMP:-0}"
export DIFFICAL_PYTHON="$PYTHON"
export DIFFICAL_CONFIG="$ROOT/docs/config.yaml"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/diffical_numba_cache}"
export KMP_DUPLICATE_LIB_OK="TRUE"
mkdir -p "$NUMBA_CACHE_DIR"

# Read-only preflight for automation; it must not start the GUI/backend.
if [[ "${1:-}" == --check ]]; then
  echo "MVP preflight PASS"
  echo "App: $APP_PATH"
  echo "PrettyPitch base: $PRETTYPITCH_BASE_MODEL"
  if [[ -f "$PRETTYPITCH_BASE_CHECKPOINT" ]]; then
    echo "Training checkpoint: $PRETTYPITCH_BASE_CHECKPOINT"
  else
    echo "Training checkpoint unavailable (optional): choose a local .pt in the training window."
  fi
  echo "Backend: $DIFFICAL_BACKEND_PATH"
  echo "Engine: ${DIFFICAL_ENGINE:-legacy}  SeamRamp: ${DIFFICAL_SEAM_RAMP}"
  echo "Audio cache: 20261018handover2 (old caches retained)"
  echo "Python: $DIFFICAL_PYTHON"
  exit 0
fi

exec "$APP_EXEC" "$@"
