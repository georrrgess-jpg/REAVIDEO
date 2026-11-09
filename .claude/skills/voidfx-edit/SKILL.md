---
name: voidfx-edit
description: The user's preferred editing style for short vertical fan edits ("voidfx" style, from the Dexter × Doomsday reference). Use for ANY request to make, remake, recut or restyle an edit, TikTok/Reels/Shorts edit, fan edit, scene-pack edit or music edit, even if the style isn't named. Covers the style rules (beat-grid timing, themed transitions, grade, framing, loop ending, SFX level), the build workflow with the tools in edit/, and the QA checklist.
---

# voidfx edit style (the user's default)

Every edit made in this repo uses this style unless the user asks otherwise. It was
reverse-engineered from the reference `doomsday_voidfx_9x16`. The full analysis is in
`docs/style-breakdown.md`, and the first edit built with it is the code in `edit/`.

## The style, as rules

**Format**
- 1080×1920, 60 fps, about 20–25 s (the reference is 22.65 s), H.264 at about 26 Mbit/s, AAC 256k.
- Music only. No captions, titles, watermarks or voice-over.
- It must loop: the last transition reveals the first clip.

**Timing (the most important part)**
- Get the song's beat grid with `edit/tools/beatgrid.py` and cut **only on half-beats**.
- Clip lengths come from {2, 3, 4} half-beats. At ~95 BPM that is 0.63 / 0.94 / 1.26 s.
  - Allow at most one longer hold (5–6 half-beats) in the back third, as a breather before the ending.
  - Never use two 4-half-beat clips in a row; vary 2-3-4 so the rhythm reads as edited.
- Aim for about one cut per second: 20–24 clips in 22 s.
- Hard cuts, blur dissolves, whips and punch-ins land **exactly** on the grid point.
- Wipes start on the grid point and are centred up to 0.13 s after it, so the motion "lands" just after the hit.
- Put the heaviest transition (red flash with rings, or a punch-in with a low hit) on the strongest drum hit. `beats_strength` in the beatgrid output shows which hits are strongest.
- Default section: if the user gives a reference edit that uses the same song, use `--ref` so the exact section is reused. Otherwise use the suggested section, and tell the user which time range was picked.

**Clip selection**
- One character as the sole subject. Mostly close-ups and extreme close-ups of the face; a medium or wide shot only every 3–5 clips, for air.
- Prefer moments with a change inside the clip: a head turn, a look to camera, glasses going on, a smirk forming. Set the in-point so that change happens inside the slot.
- Alternate light and dark, and warm and cool, from clip to clip. Don't put two clips from the same scene next to each other.
- Open on a strong, well-lit face. End on a still, looking-off shot that the loop transition can grow out of.

**Framing and motion**
- Use 9:16 crops that track the face (`extract.py` uses the YuNet face detector).
- Use `z` > 1 to tighten wide source shots.
- Every clip has a slow push-in of 5–8% over its length (`push=(1.00, 1.06)`), or a slow pull-out (`(1.06, 1.00)`) at most once in a row. Never a static frame, and never camera shake.

**Grade**
- Muted and cool: saturation 0.3–0.6, contrast 1.10–1.2.
- Reds are protected and stay vivid (`rk` up to 1.0). This is the signature look: grey world, red blood.
- Lift dark shots with `gain` instead of leaving them muddy (1.2–1.45 on night shots).
- Finishing is always on: soft bloom, vignette and fine grain (`fx.finish`). Teal goes into the shadows.
- Use full-frame colour (red or green tint) only as a transition beat, never as the base grade.

**Transitions: themed to the footage**
Almost every cut is hidden. Pick transitions by what is in the clips: foliage → leaf wipe, lab or forensics → glass slide or blood, camera → Polaroid flash, water or boat → water rise, fire → smoke and embers. Keep about 3–4 plain hard cuts in the second half so the big transitions hit harder.

| Transition (`edit/timeline.py`) | Use for | Length |
|---|---|---|
| `t_leaf` palm or fern frond wipe | outdoor and green shots | 0.30 s |
| `t_blur` defocus dissolve with zoom | the neutral glue between big moves; use 3–4 times | 0.30 s |
| `streaks` + `t_glass` light streaks, then a diagonal glass-edge wipe | sunlit or reflective shots | 0.19 s |
| `t_smoke` rising smoke and embers | fire, heat, anger | 0.45 s |
| `t_dip` quick dip to dark | into macro or extreme close-ups | 0.27 s |
| `draw_slide` + `t_slidewipe` glass slide graphics, then the slide flies at camera | forensic or lab scenes | 1.3 s overlay + 0.47 s wipe |
| `t_blood` blood pours down, then drains to reveal | the mid-edit climax | 0.6 s |
| `t_spin` spin with rotational blur | disorientation, into an upside-down shot | 0.34 s |
| `t_polaroid` flash, then the frame becomes a Polaroid that drops away | camera or photo moments | 0.66 s |
| `t_drops` blood spatter blots | violence, the spatter wall | 0.46 s |
| `t_whip` horizontal or vertical whip blur | fast energy; alternate the direction | 0.14–0.20 s |
| `t_punch` hard cut with zoom-in and slight flash | a strong hit on a cut | 0.18–0.20 s |
| `t_red` red flash with concentric rings | the single strongest hit | 0.5 s |
| `t_water` water line rises with bubbles and tint, then drains | water and boats, near the end | 0.7 s |
| `t_iris` blood drop falls and swells into a lens showing clip 1 | **always the last transition: the loop** | 0.7 s |

Follow the reference's arc:
- **Opening:** two organic wipes.
- **First third:** effects build up (glass, smoke, slides).
- **Middle:** the blood climax.
- **Second half:** quicker spins, flash, spatter, whips and hard cuts.
- **End:** one breather hold, then water and the loop.

Don't reuse the same big transition twice in one edit. Blur, whip and punch can repeat.

**Sound**
- Use the song section untouched: original speed and pitch, peak normalised to about −1.5 dBFS.
- Add synthesized SFX (`audio.py`) only under transitions: whooshes on wipes, a shutter on the Polaroid, splats on spatter, a low hit on the red flash and punch-ins, bubbles on water. Keep them about 20–25 dB under the music. They should be felt, not noticed.

## Workflow for a new edit

The user's stored footage is listed in `scenepacks/README.md` (Dexter S1, S2, S4 and mixes; The Flash S1, S2, S3, S5, all seasons, Blitzkrieg; plus a second voidfx reference edit). If `media/scenepacks/` is empty, run `bash scenepacks/fetch.sh [name-filter]` first. Shot lists are already in `scenepacks/catalog/<name>.json`. Choose the pack that matches the character the user asks for, and mix packs of the same show when one pack is too short. Put other inputs (song, new footage) in `media/`, which is gitignored and never committed. Tools need ffmpeg and `pip install numpy scipy opencv-python-headless soundfile`. The face model comes from rea-video's cache, `~/.cache/rea-video/models/face_detection_yunet_2023mar.onnx`. If that file is missing, get it from `https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx`; its sha256 starts with `8f2383e4`.

1. **Beat grid:**
   ```bash
   python edit/tools/beatgrid.py media/song.mp3 [--ref media/reference.mp4] -o media/beats.json
   ```
2. **Catalog the footage:**
   ```bash
   python edit/tools/catalog.py media/scenepack.mp4 -o media/catalog
   ```
   Look at every `sheet_N.jpg`, shortlist about 35 candidates, then run `--strip <ids>` and look at the strips to choose in-points.
3. **Plan:** write the cut list (half-beat grid points) and a shot table: clip, in-point, transition out, with reasons. Show the user this plan before rendering a long edit, unless they asked for a finished video straight away.
4. **Configure:** start a new edit as a copy of `edit/` (for example `edits/<name>/`), keeping `fx.py` and `render.py`, and update:
   - `config.py`: `SCENEPACK`, `SONG`, `SONG_OFFSET`, `CUTS`, `SHOTS` (src in-point, `x` face hint, `z`, grade and push per clip).
   - `timeline.py`: the `TRANS` list and the shot-specific overlays in `layer()`. These are tied to shot indexes and times, so rewrite them for the new cut list.
   - `audio.py`: the SFX `put(...)` calls, so they follow the new transition times.
5. **Extract and check framing:** run `python extract.py` and `python valid.py`, then look at `out/sheets/clips.jpg`. Replace any clip that is mostly a back of a head, a blank wall, or crosses a hard cut inside its slot. `valid.py` prints `CUTS` when that happens.
6. **Preview every transition** before the full render:
   ```bash
   python render.py sheet NAME t0 t1 0.05 10
   ```
   Look at each sheet. Check for edges or lines left by mattes, graphics that look like flat clip-art, and shots that are too dark or too bright.
7. **Full render:** `python render.py frames 0 <N> 4`, then `python audio.py`, then the ffmpeg encode line in `build.sh`. This takes about 1.5 min for 22 s on 4 cores.
8. **QA:**
   - Re-measure the cuts against the grid; every hard cut should be within ±20 ms.
   - Look at a 4 fps overview sheet of the encoded file.
   - Check the last frame flows into the first.
   - Check the file size (about 75 MB for 22 s).
9. Deliver the MP4 and list anything unverified. You cannot listen to the mix, so say that the SFX level should be checked by ear.

## Known limits to tell the user about

- 720p scene packs get upscaled about 2.7× in the 9:16 crop and look soft. Ask for a 1080p or 4K pack when quality matters.
- The graphics (fronds, smoke, blood, water) are procedural approximations of After Effects-style assets. The leaf wipe and blood spatter are the most stylized.

## When the user gives feedback

Update this file so the preference sticks. Change the rule and add a dated line to the log below. If a change only applies to one edit, put it in that edit's config instead.

### Preference log
- 2026-11-09: Style adopted as the default for all future edits, based on the Dexter × Doomsday voidfx reference (first build: `edit/`, `dexter_doomsday_edit_v1.mp4`).
