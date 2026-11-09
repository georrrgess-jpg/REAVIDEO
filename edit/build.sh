#!/usr/bin/env bash
# Rebuild the edit from config.py. Needs ffmpeg and: pip install numpy scipy opencv-python-headless soundfile
# Media paths are set in config.py (SCENEPACK, SONG); fetch stored packs with: bash ../scenepacks/fetch.sh
#   bash build.sh          # final 1080x1920 60 fps -> out/edit.mp4
#   bash build.sh --draft  # quick 30 fps review     -> out/edit_draft.mp4
set -euo pipefail
cd "$(dirname "$0")"
python extract.py                       # graded, face-tracked clips (parallel)
python render.py video "$@"             # soundtrack + render + encode in one pass
