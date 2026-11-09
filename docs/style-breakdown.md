# Reference edit: style breakdown

Reverse-engineered from `doomsday_voidfx_9x16 (3).mp4` (22.65 s, 1080×1920, 60 fps) with
[rea-video](https://github.com/JuneYaooo/rea-video) plus a frame-by-frame visual review.

## What it is

- A fan edit built entirely from TV footage (the same season-1 scene pack supplied for the remake). No AI-generated or 3D shots; every "graphic" is a composited overlay.
- 22 clips in 22.65 s, about one cut per second.
- A seamless loop: the last transition lands back on the first clip.
- No captions or titles. The OCR hits in the automatic report are false positives from set dressing.

## Music and timing

- Audio is MF DOOM "Doomsday" from 65.64 s to 88.29 s of the track, at original speed and pitch (waveform correlation 0.87 against the supplied MP3).
- Tempo is 95.4 BPM: one beat is 0.629 s, a half-beat 0.314 s.
- Every cut sits on that half-beat grid. Clip lengths are 2, 3 or 4 half-beats (0.63 / 0.94 / 1.26 s), with one 1.7 s hold near the end.
- Hard cuts and blur cuts land exactly on the grid (within 20 ms). Wipes are centred up to 0.13 s late, so the wipe starts on the beat and finishes just after it.
- Short whooshes and a liquid sound sit under the wipes, mixed well below the music.

| # | Cut (s) | Transition out of the clip |
|---|---|---|
| 1 | 0.94 | zoom + palm-leaf wipe |
| 2 | 1.88 | large fern-frond wipe |
| 3 | 2.83 | defocus (blur) dissolve |
| 4 | 3.90 | glass light streaks, then a diagonal glass-edge wipe |
| 5 | 5.03 | orange smoke / embers rising from the bottom |
| 6 | 5.66 | quick dip to dark |
| 7 | 7.00 | a floating glass blood slide flies at camera and becomes the next shot |
| 8 | 7.8–8.0 | red liquid pours down the frame, then drains to reveal the next shot |
| 9 | 8.80 | blur dissolve |
| 10 | 10.06 | spin with rotational blur (lands on an upside-down shot) |
| 11 | 11.32 | the frame freezes into a tilted Polaroid that drops away |
| 12 | 12.58 | ink splatter droplets |
| 13 | 13.20 | horizontal whip blur |
| 14 | 13.83 | blur dissolve |
| 15 | 15.09 | hard cut |
| 16 | 16.35 | red flash with concentric rings |
| 17 | 17.29 | hard cut |
| 18 | 17.92 | hard cut |
| 19 | 18.55 | whip blur |
| 20 | 20.25 | diagonal line wipe |
| 21 | 21.38 | water line rises with bubbles and a green tint, then drains |
| 22 | 22.4 | a blood drop falls, swells into a glossy red lens showing clip 1, and fills the frame (loop) |

## Look

- Framing: 9:16 crops held on the face, mostly close-ups and extreme close-ups.
- Motion: a slow, constant push-in on every clip; no shake.
- Grade: muted and cool with low saturation, but reds stay vivid. A few clips are tinted fully red or green. Soft bloom, vignette and fine grain.
- Transitions are themed to the content: foliage on outdoor shots, glass slides and blood on the forensic shots, water on the boat shot.

## How the new edit applies it

`edit/` rebuilds the style procedurally (Python + OpenCV + ffmpeg) on a different selection and order of clips from the scene pack:

- Same song section and the same 22 cut times.
- Same transition vocabulary in the same order, re-themed where the new clips suggest it: a camera flash triggers the Polaroid, the ink splatter becomes blood spatter into the spatter-wall shot, and the flame shot feeds the smoke wipe.
- All graphics (fronds, glass slide, blood, smoke, Polaroid, water, blood-drop lens) are generated in code; nothing is lifted from the reference.
- Transition sound effects are synthesized and mixed about 23 dB under the music.

Source footage and music are not stored in this repository. To rebuild, place them at `media/scenepack.mp4` and `media/song.mp3` and run `edit/build.sh`.
