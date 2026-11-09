#!/usr/bin/env bash
# Build an edit. Needs ffmpeg and: pip install numpy scipy opencv-python-headless soundfile
# An edit is a folder holding a config.py (edit/ itself is the first one; new ones live in edits/<name>/).
#   bash edit/build.sh                        # final render of edit/        -> edit/out/edit.mp4
#   bash edit/build.sh edits/hold_on          # final render of that edit    -> edits/hold_on/out/edit.mp4
#   bash edit/build.sh edits/hold_on --draft  # quick 30 fps review copy     -> .../out/edit_draft.mp4
# Fetch stored scene packs first if media/ is empty: bash scenepacks/fetch.sh
set -euo pipefail
ENGINE="$(cd "$(dirname "$0")" && pwd)"
DIR="$ENGINE"
if [ $# -gt 0 ] && [ -d "$1" ]; then DIR="$(cd "$1" && pwd)"; shift; fi
cd "$DIR"
python "$ENGINE/extract.py"             # graded, face-tracked clips (parallel)
python "$ENGINE/render.py" video "$@"   # soundtrack + render + encode in one pass
