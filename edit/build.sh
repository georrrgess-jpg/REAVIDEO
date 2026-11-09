#!/usr/bin/env bash
# Rebuild the edit. Needs ffmpeg and: pip install numpy scipy opencv-python-headless soundfile
# Put the scene pack at ../media/scenepack.mp4 and the song at ../media/song.mp3 first.
set -euo pipefail
cd "$(dirname "$0")"
python extract.py            # face-tracked 9:16 clips -> clips/
python valid.py              # usable frame range per clip -> clips/meta.json
python render.py frames 0 1359 4
python audio.py              # song section + transition SFX -> out/audio.wav
ffmpeg -v error -y -framerate 60 -i out/frames/%05d.jpg -i out/audio.wav \
  -c:v libx264 -preset slow -crf 19 -maxrate 26M -bufsize 52M -pix_fmt yuv420p \
  -color_primaries bt709 -color_trc bt709 -colorspace bt709 \
  -c:a aac -b:a 256k -shortest -movflags +faststart out/dexter_doomsday_edit_v1.mp4
