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
- Default section: if the user gives a reference edit that uses the same song, use `beatgrid.py --ref` so the exact section is reused. Otherwise run `tools/sections.py`. It picks the **last chorus with full drums, 8 bars long**, and makes the video exactly 64 half-beats so it loops musically: the closing fill lands back on the chorus downbeat. Tell the user which time range was picked and why.
- Before planning, list the loudest half-beats (`sections.py` prints them). Put the big transitions (glass, blood, red, punch-ins) on those, and keep cuts on beats, using lengths of 2 and 4, unless a 3+3 pair is wanted for syncopation.

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

| `out=` name in config.py (overlay `fx=`) | Use for | Length |
|---|---|---|
| `leaf` (`out_opts` ang, seed) palm or fern frond wipe | outdoor and green shots | 0.30 s |
| `blur` defocus dissolve with zoom | the neutral glue between big moves; use 3–4 times | 0.30 s |
| `shatter` (`out_opts` cx, cy = impact point) cracks race out from the impact, then the frame breaks into glass shards flying at camera; impact flash | the loudest snare; windows, mirrors, a violent beat | 0.46 s |
| ~~`glass` + `fx=['streaks']`~~ light streaks and glass-edge wipe. **The user disliked the glow; don't use it unless asked.** | | |
| `smoke` rising smoke and embers | fire, heat, anger | 0.45 s |
| `dip` (`fx=['fringe']` on macro shots) quick dip to dark | into macro or extreme close-ups | 0.27 s |
| `scope` (`out_opts` px, py = the blood/evidence position in the outgoing shot) dive into it, microscope eyepiece with drifting blood cells and a 40x reticle, pull back out into the next shot | forensic or lab shots, a blood slide, evidence | 0.78 s |
| ~~`slidewipe` + `fx=['slides']`~~ floating slide graphics. **The user disliked it; use `scope` for forensic moments.** | | |
| `blood` blood pours down, then drains to reveal | the mid-edit climax | 0.6 s |
| `spin` spin with rotational blur | disorientation, into an upside-down shot | 0.34 s |
| `polaroid` flash, then the frame becomes a Polaroid that drops away | camera or photo moments | 0.66 s |
| `drops` blood spatter blots | violence, the spatter wall | 0.46 s |
| `whip` (`out_opts` d=±1, vertical, dur, amount) horizontal or vertical whip blur | fast energy; alternate the direction | 0.14–0.20 s |
| `punch={}` on the incoming shot (z, flash, dur) hard cut with zoom-in and slight flash | a strong hit on a cut | 0.18–0.20 s |
| `confetti` a curtain of red/blue/white/gold parade confetti falls and reveals the next shot behind it; pieces flutter and settle | parades, celebrations, crowds, any joyful beat before a dark shot | 0.6 s |
| `red` red flash with concentric rings | the single strongest hit | 0.5 s |
| `plastic` a sheet of kill-room plastic wrap drags across, refracting and glinting, and the next shot is behind it | kill rooms, dark shots, near the end before the breather | 0.7 s |
| ~~`water`~~ water rise. **The user disliked it; use `plastic` or another new graphic transition.** | | |
| `iris` blood drop falls and swells into a lens showing clip 1 | **always the last transition: the loop** | 0.7 s |
| `cut` plain hard cut | 3–4 in the second half | 0 |

Other overlays: `fx=['leak_bubbles']` (red light leak and drifting particles) on a long hold before the ending.

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

1. **Song section:**
   ```bash
   python edit/tools/sections.py media/song.mp3 -o media/<name>_section.json
   ```
   This prints energy per bar, the chorus sections, the chosen section and the loudest half-beats. If there's a reference edit using the same song, use `python edit/tools/beatgrid.py media/song.mp3 --ref media/reference.mp4` instead.
2. **Shortlist footage:** shot lists are in `scenepacks/catalog/`.
   - Score the packs, then make contact sheets of only the strong ones: face > 0.2 of frame height, sharpness > ~150.
     ```bash
     python edit/tools/score.py <pack> [<pack> ...]
     ```
     This writes `media/scores.json`.
   - Then look at 8-frame strips to set in-points:
     ```bash
     python edit/tools/catalog.py media/scenepacks/<pack>.mp4 -o media/catalog/<pack> --strip <ids>
     ```
   - Prefer the sharpest packs. For Dexter, S2 (pack a) and the mixed-season pack score highest.
   - Watch for the same scene appearing in two packs; don't use it twice.
3. **Plan:** write a shot table: clip, in-point, length in half-beats, transition out, with reasons. Show the user this plan before rendering, unless they asked for a finished video straight away.
4. **Configure: write only `edits/<name>/config.py`.** The engine in `edit/` is shared: run it from the edit folder and that folder's `config.py` is used (see `edits/hold_on/config.py` for a full example). Set:
   - media and timing: `SCENEPACK`, `SONG`, `SONG_OFFSET`, `GRID0` and `HALF` (from beats.json: `grid[0]` and `period / 2`), `FIRST_CUT`.
   - `SHOTS`: one dict per clip, with `beats` (2–4 half-beats, one 5–6 hold), `out` (transition name), optional `out_opts`, `fx`, `punch`, `pack=` (to take that clip from a different scene pack), plus `src` in-point, `x` face hint, `z`, grade and push.

   Cut times, transition timing (wipes placed slightly late, per `timing.py`) and every sound effect are derived automatically. Importing the config or timeline rejects bad plans with a clear message: wrong lengths, unknown names, overlapping transitions, or no `iris` at the end.
5. **Extract:** from the edit folder, `python ../../edit/extract.py` cuts all clips in parallel (about 100 s for 22 clips). It tracks faces, converts any source to 60 fps, applies each shot's grade once, and stores 4:4:4 JPEG frames. Check `out/sheets/clips.jpg`, and read the WARNING lines, which flag a clip that runs out of usable frames (it would freeze). After changing one shot's `src`, framing or grade, re-extract only that shot: `python ../../edit/extract.py 7`. A WARNING usually means the catalog merged two source shots, so move the in-point past the cut or pick another shot.
6. **Draft:** `python ../../edit/render.py video --draft` makes a 30 fps review copy in about 45 s. Check transitions in detail with `python ../../edit/render.py sheet NAME t0 t1 0.05 10`: look for matte edges, flat-looking graphics, and shots that are too dark or too bright.
7. **Final:** `python ../../edit/render.py video [--out NAME.mp4]` renders straight into the encoder (x264 medium, film tuning, crf 18) with the soundtrack in one pass, giving `out/edit.mp4` in about 2.5 min. `bash edit/build.sh edits/<name>` runs extract plus final in one step from the repo root, and adding `--draft` gives the quick version.
8. **QA:**
   - Re-measure the cuts against the grid; every hard cut should be within ±20 ms.
   - Look at a 4 fps overview sheet of the encoded file.
   - Check the last frame flows into the first.
   - Check the loop seam: the difference between the last and first frame should be close to a normal frame step.
   - Check the file size (about 75–90 MB for 19–22 s).
9. Deliver the MP4 and list anything unverified. You cannot listen to the mix, so say that the SFX level should be checked by ear.

To add a new transition: write `t_<name>` in `timeline.py`, register it in `TRANSITIONS` with its window relative to the cut, add its sound in `audio.py` `sfx_for`, and add a `SHIFT` in `timing.py` if it should land late.

## Known limits to tell the user about

- 720p scene packs get upscaled about 2.7× in the 9:16 crop and look soft. Ask for a 1080p or 4K pack when quality matters.
- The graphics (fronds, smoke, blood, water) are procedural approximations of After Effects-style assets. The leaf wipe and blood spatter are the most stylized.

- Packs at 30 or 24 fps are converted to 60 fps by repeating frames, so playback speed is right but slow motion looks steppy. Prefer 60 fps packs for slow, flowing shots.

## When the user gives feedback

Update this file so the preference sticks. Change the rule and add a dated line to the log below. If a change only applies to one edit, put it in that edit's config instead.

### Preference log
- 2026-11-09: Style adopted as the default for all future edits, based on the Dexter × Doomsday voidfx reference (first build: `edit/`, `dexter_doomsday_edit_v1.mp4`).
- 2026-11-10: Feedback on the Hold On edit: the user disliked the glass light-streak glow, the floating slide graphics with the slide wipe, and the water rise, and asked for new, unique animated transitions instead. They were replaced with `shatter`, `scope` and `plastic`. The user also rejected the dark extreme close-up (teeth) after the smoke, which was replaced with the blood-room look-up shot. Prefer fresh, physical, themed animations over overlays that just glow or tint.
- 2026-11-10: Second edit, Dexter × "Hold On, We're Going Home" (`edits/hold_on/`). It established that a section is a final chorus of exactly 8 bars that loops on its downbeat, that big transitions go on the measured loudest hits, that shots are chosen by sharpness and face size (sharper sources than the first edit), and added the `confetti` transition.
- 2026-11-09: Faster pipeline. The plan lives in config.py alone, with cuts, transitions and SFX derived from it. The grade is applied once at extract time, the draft mode takes 45 s, and render plus encode is a single pass with no intermediate image files.
