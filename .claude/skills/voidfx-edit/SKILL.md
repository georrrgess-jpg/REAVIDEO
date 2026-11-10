---
name: voidfx-edit
description: The user's preferred editing style for short vertical fan edits ("voidfx" style). The approved STANDARD is the Dexter × "Hold On, We're Going Home" v3 edit (edits/hold_on); new edits start from it with edit/tools/new_edit.py. Use for ANY request to make, remake, recut or restyle an edit, TikTok/Reels/Shorts edit, fan edit, scene-pack edit or music edit, even if the style isn't named. Covers the style rules (beat-grid timing, themed transitions, grade, framing, loop ending, SFX level), the build workflow with the tools in edit/, and the QA checklist.
---

# voidfx edit style (the user's default)

Every edit made in this repo uses this style unless the user asks otherwise. It was first
reverse-engineered from the reference `doomsday_voidfx_9x16` (analysis: `docs/style-breakdown.md`;
first build: `edit/config.py`).

## The standard (user-approved, 2026-11-10)

**`edits/hold_on/config.py`, Dexter × "Hold On, We're Going Home" v3, is the standard.** The user
asked for it to be the template for future edits. Start every new edit from it, and match or
beat it on the quality bar below. Where an older rule in this file disagrees with the standard,
the standard wins.

**Starting a new edit**
```bash
python edit/tools/sections.py media/<song>.mp3 -o media/<name>_section.json
```
```bash
python edit/tools/new_edit.py <name> --song media/<song>.mp3 --section media/<name>_section.json
```
This writes `edits/<name>/config.py` with the standard's 22 slots: lengths, transitions,
overlays, punch-ins, grade and push. Each slot names the kind of shot it wants. Fill `src`,
`pack` and `x` for every slot (the config refuses to build while any slot is empty), then
retune grade and gain per shot.

**What the standard is**
- **Song section:** the last chorus with full drums, exactly 8 bars (64 half-beats; 19.2 s at 100 BPM). The video is exactly that long, so the closing drum fill loops onto the chorus downbeat.
- **Rhythm:** first cut at half-beat 4, then 2-4-2-4-2-3-3-2-4-2-2-4-2-2-4-2-2-2-4-2-5 half-beats. Every cut is on a beat (hard cuts measured at 0 ms).
- **Big hits:** the heaviest moments sit on the loudest measured hits: knife on the loudest early snare (#10), punch-in (#26), blood spatter (#38), punch-in on the poster (#46), red flash on the loudest kick (#52). With a new song, move cuts so these land on its loudest hits (`sections.py` lists them).
- **Transition order:** leaf → leaf → **knife** → blur → smoke → dip → **scope** (microscope dive into the held object) → blood pour → cut + punch → spin → whip → **polaroid** → drops → whip → **confetti** → cut → punch + vertical whip → cut → **red** → cut → **plastic** → **iris** loop.
- **Arc:** sunlit public persona → darker, effects build → lab/blood climax at about 7 s → celebration colliding with the dark (parade vs red-lit stares) → red light → cold blue → warm still hold → loop.
- **Shots:** 22 clips, mostly face close-ups. Sources are chosen by measured sharpness and face size (`tools/score.py`): the standard's median face sharpness is 261 and faces average 61% of frame height. No scene is repeated across packs, and each clip has a moment of change inside its slot.
- **Grade:** muted (sat 0.35–0.55, con 1.12, tint (1.04, 1.0, 0.96)), reds protected (rk 0.8–1.0), dark shots lifted with gain 1.3–1.45, plus bloom, vignette and grain. A push-in of 5–8% runs on every clip.
- **Sound:** the song section is untouched; synthesized SFX per transition sit about 23 dB under the music.

**Quality bar (check before delivering)**
- Hard cuts within one frame of the beat (`render.py` QA in the workflow); the loop seam is close to a normal frame step; the length equals the section exactly.
- No WARNING lines from `extract.py` (no frozen frames).
- Median face sharpness ≥ ~200 and mean face size ≥ ~55%.
- Every transition previewed with `render.py sheet` before the final.
- Avoid what the user rejected (glass light-streak glow, floating slides, water, glass shatter); see the preference log.

**Second standard: speedster / superpower edits (user-approved 2026-11-10, with a caveat).**
`edits/flash/config.py`, The Flash × "SLAUGHTER HOUSE" v3, is the standard for edits about powered
characters (speed, lightning, energy). Start one with:
```bash
python edit/tools/new_edit.py <name> --song media/<song>.mp3 --section media/<name>_section.json --from flash
```
The template includes the beat lists (`KICKS`, `SNARES`) taken from the new section.

What it is:
- **Section:** the first drop plus its riser bar, 8 bars (64 half-beats); the riser launches the loop back into the drop. Use this instead of the last chorus when the song is a drop-based track (phonk, EDM).
- **Rhythm:** first cut at 4, then 2-2-2-4-2-4-2-4-2-2-3-3-2-2-2-4-2-4-2-4-5 half-beats.
- **Transition order:** speedlines → whip → blur → spin → **emblem** (drawn, animated) → whip → cut → **phase** → cut → cut → punch + eye gleam + vertical whip → blur → cut → blur → **strike** → cut → cut → punch + whip → **red** → cut → punch + dip → **boltsplit** loop.
- **Fill:** bumps on kicks, colour hits on snares, gold afterimages on running and flying shots, blue arcs on the tech shot, shockwave ripples on the riser hold.
- **Grade:** muted with red and gold kept (sat .45, rk 1.0, yk 1.0); dark shots lifted (gain 1.3–1.6).

**Caveat (the user's words: adequate, "still not quite up to the quality I'm pursuing").** Treat it as the minimum, not the target. Every new powered-character edit should try to go beyond it, and say how. Known gaps to work on:
- Sources are 720p, upscaled 2.7×. Ask for 1080p or 4K packs.
- Effects are procedural: lightning, speed lines and ripples are cleaner than before but not at After Effects asset quality. Push toward richer graphics (layered glows, motion blur on graphics, depth and parallax, particles that interact with the footage) and crafted, drawn animations like the emblem, which the user preferred.
- Sound effects are synthesized and their level is unverified by ear. Ask the user to check the mix.
- Faces are smaller in frame than in the Dexter standard (41% against 55%), because speed shots are wider.
Ask the user, early, for a reference of the quality they're after when starting a new edit of this kind.

**Flagship candidate (built 2026-11-10, awaiting the user's verdict): `edits/less_i_know`, Dexter × "The Less I Know The Better".**
The user asked for the best Dexter edit yet, one that sets a new standard. If they approve it, make it the Dexter standard in place of Hold On (update `new_edit.py` TEMPLATES and the slot roles). What it added beyond Hold On:
- **Sharper footage:** ML upscaling at extract time (`extract.py --sr`, Real-ESRGAN general-x4v3 on CPU, about 3–4 min per shot, about 70 min for 19 shots). Weights go in `media/models/` (see `edit/sr.py`). On identical frames it scores about 3× bicubic on a fine-detail measure (partly the model's own sharpening; visually, edges are cleaner and there's no compression blocking). Run the fast extract first to plan and preview, and `--sr` only once the shot list is settled.
- **Kinetic lyric typography** (`TEXT` in the config, drawn by `edit/typo.py`). Word timing comes from vocal-stem onsets (`rea-video stems` on the section, then onset peaks). Entrances: type, pop, slide, forever (slow letters with widening tracking on a held note), glitch, impact (a huge word slams in with a flash and colour split). Exits: fade, cut, blur, scatter. White on a black stroke with soft shadow; keywords in red. This overrides "no captions" only when the user asks for references with lyrics, as voidfx's own Passenger edit has.
- **Black-and-white grade with selective red:** `sat=0`, `rk=1`, `rth=0.22` (strict red key: only blood, red light and the red string keep colour, not skin; orange is rejected). Raise `rth` on a shot whose skin turns pink.
- **Section:** bridge into chorus, 10 bars (`sections.py --start S --bars 10`), the same stretch the voidfx reference uses (`beatgrid.py --ref` found it at correlation 0.95).
- **New drawn transitions:** `ink` (black ink blooms in from the edges with droplets and a wet rim, then parts onto the next shot), `orange` (a drawn blood-orange slice spins in, lands with a squash and a juice spray, then its segments open onto the next shot as the camera dives in), `slices` (knife cuts slash down the frame, then the strips fall away), `syringe` (a drawn M99 syringe is pushed at the viewer, the plunger goes down, the world swims, the eyelids close with red glowing through, and the next shot blinks awake).
- **New overlays:** `dust` (light-catching motes in 3 depth layers), `glint` (4-point star glints on specular highlights, pulsing on hits), `ghost` (glowing outline doubles drifting off the subject, the voidfx signature), `trails` (psychedelic frame echoes).
- **Result:** `dexter_less_i_know_edit_v1.mp4`, 20.51 s at 60 fps, 80.6 MB; hard cuts 1–12 ms from the beat; no frozen frames.
- **Sound for every new piece:** ink splats, juice, knife shings, syringe tick, rubber squeak, heartbeat and woozy downer; ticks, shimmer, zaps and an impact boom on the lyrics.

**For another character:** keep the slot structure, rhythm, arc and big-hit placement. Swap the Dexter-signature transitions (knife, scope, plastic, polaroid) for shape transitions built from that character's own signature objects.

**Reproducing the standard:** `bash scenepacks/fetch.sh dexter`, then put the song at `media/hold_on.mp3` (it isn't in the Drive manifest, so the user must supply it again), then `bash edit/build.sh edits/hold_on`.

## The style, as rules

**Format**
- 1080×1920, 60 fps, exactly the chosen section (8 bars; 19.2 s at 100 BPM in the standard), H.264 (x264 medium, film, crf 18), AAC 256k.
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
| `knife` Dexter's carving knife thrusts in (drop shadow, edge glint); the blade is a window onto the next shot; the camera dives into the blade until it fills the frame | a Dexter signature moment on a loud snare | 0.64 s |
| ~~`shatter`~~ glass shatter. **The user disliked it; prefer `knife` or another character-themed shape transition.** | | |
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

**The Flash kit** (first used in `edits/flash`). Lightning is drawn procedurally (`bolt_path`, `bolt_tree`, `draw_bolts`: gold core and glow; `BLUE_GLOW` for Speed Force blue).

| name | what | use for | length |
|---|---|---|---|
| `speedlines` (`out_opts` d=±1) | the frame smears sideways into golden speed streaks, and the next shot resolves out of them | a speedster taking off; the opening | 0.36 s |
| `phase` | Speed Force vibration: RGB-split ghost copies shake, a pop, then it phases into the next shot, with small arcs | a loud hit mid-edit | 0.40 s |
| `emblem` | **drawn Flash emblem (`draw_emblem`), rendered procedurally every frame so it stays crisp at any zoom**: the gold ring draws itself on with a spark at its head, the pearl disc pops in, the bevelled gold bolt strikes down into place (flash and arcs), a glint sweeps the metal, then the camera dives into the bolt's wide centre, rotating its long axis upright, into the next shot. Bevels come from analytic and distance-transform normals with gold Blinn shading. | the signature moment on the loudest early hit | 0.60 s + 0.36 s |
| `emblem_real` | (older) built on the real chest emblem in the outgoing shot (found by brightness, followed by template matching near it): a light sweep crosses it, lightning leaps off its rim, the camera pushes in on it, it flashes on the hit, and the next shot opens out of the emblem's own shape as the camera dives in | the signature moment on the loudest early hit; the outgoing shot must show the emblem | 0.80 s charge + 0.32 s |
| `strike` | a lightning leader draws down the frame, then strikes; the frame tears open along the bolt with a glowing edge | a shot of throwing lightning; a loud hit | 0.52 s |
| `boltsplit` (loop) | a lightning-bolt seam draws across the last shot and burns, then the two halves fly apart along it with motion blur, and shot 0 is behind them | **loop ending for speedsters** | 0.52 s |
| ~~`tunnel`~~ Speed Force tunnel loop. **The user found it loose; use `boltsplit`.** | | |
| `fx=['eyes']` | a brief, subtle gold gleam in the eyes on the hit (small core, soft glow, short glint; no arcs), on YuNet eye landmarks | a cowl close-up with a punch-in. **Keep it subtle: the user found glowing eyes with lightning too much.** | about 0.3 s |
| `fx=['crackle']`, `['crackle_blue']` (arcs over the top of the frame, Speed Force blue) / ~~`['crackle_build']`~~ (the user disliked the loose wandering arcs on the last hold) | lightning arcs crawl over the shot; `_build` grows with the shot (put it on the riser hold before the loop) | charged-up shots; the last hold | the whole shot |

Fill-the-dead-space overlays (any edit; config `KICKS` / `SNARES` are half-beat indices from `sections.py`):
- `fx=['ripple']`: Speed Force shockwave rings (refraction plus a thin blue edge) on each kick or snare inside the shot, growing toward the shot's end. Use it on a riser hold before the loop.
- `fx=['bump']`: a +4.5% zoom bump on each strong kick inside the shot (shot 0 holds it through the loop tail, so the seam matches).
- `fx=['rgbhit']`: a colour-split kick on each snare.
- `fx=['echo']`: gold speedster afterimages of earlier frames (running or flying shots).
- ~~`fx=['sweep']`~~: a warm light sweep across a hold. **The user disliked the gold shine over a face; don't use it.** Use `ripple` on holds instead.
Use them on holds of 3 or more half-beats that have no transition.

Lightning is drawn crisp: a white-hot core, a tight inner glow and a soft outer glow, with low displacement. Avoid thin wandering arcs.

Grade option `yk` (yellow keep) keeps gold, orange and lightning saturated in a muted grade, the way `rk` does for red. The Flash look is `sat .45, rk 1.0, yk 1.0`.

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
4. **Configure: start with `tools/new_edit.py` (see The standard), then edit only `edits/<name>/config.py`.** The engine in `edit/` is shared: run it from the edit folder and that folder's `config.py` is used (see `edits/hold_on/config.py` for a full example). Set:
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

- Most packs repeat about 60% of frames (24 fps material in a 60 fps file). `--sr` reuses one upscale per unique source frame.
- Packs at 30 or 24 fps are converted to 60 fps by repeating frames, so playback speed is right but slow motion looks steppy. Prefer 60 fps packs for slow, flowing shots.

## When the user gives feedback

Update this file so the preference sticks. Change the rule and add a dated line to the log below. If a change only applies to one edit, put it in that edit's config instead.

### Preference log
- 2026-11-10: The user asked for the best Dexter edit yet, "above and beyond", with multiple custom effects, animations, graphics and transitions, using voidfx references and The Less I Know The Better. Built `edits/less_i_know` (see "Flagship candidate"). It is the first edit with lyric typography, a black-and-white selective-red grade, and ML upscaling.
- 2026-11-10: **The user made Flash × Slaughter House v3 the standard for speedster / powered-character edits**, while noting it is still not at the quality they're pursuing (see "Second standard"). `new_edit.py --from flash` scaffolds from it.
- 2026-11-10: Flash edit v3. The user disliked the gold light sweep over Barry's face on the final hold (removed everywhere; `ripple` replaces it) and the bolt left in the middle after `strike` (a bug: it now burns out within about 0.1 s and the torn edges fade). They then asked for the emblem to be **drawn as a high-quality animated graphic** instead of tracking the real one, which led to `draw_emblem` and the new `emblem`. Takeaway: for character symbols, the user prefers a crafted, well-animated graphic (draw-on, pop-in, strike, glint, dive) over effects pasted onto footage. QA now scans every frame for invalid pixels (NaN shows as a black frame).
- 2026-11-10: Flash edit revision (v2). The user liked the transitions' structure but not how the animations looked. Changes: the emblem transition must use the **real emblem in the footage**, with the effect drawn on it and the transition coming out of it (not a drawn graphic); the Speed Force tunnel and the wandering arcs on the last hold were too loose (replaced with `boltsplit`); the eye lightning was too much (now a subtle gleam). The user asked for animation wherever there is dead space, which led to bump, rgbhit, echo, sweep and crisper lightning. General preference: **anchor effects to real objects in the shot, keep them tight and crisp, and fill holds with motion tied to the beat.**
- 2026-11-10: Third edit, The Flash × "SLAUGHTER HOUSE" (`edits/flash`). The user asked for the standard's style with the edit's own Flash flair (lightning, emblem, eyes), using elements of a reference edit (`media/references/flash_wonderwall_ref.mp4`, user-supplied, not in the Drive manifest) without copying it. The kit above was added. Choices: section = first drop + its riser bar (the riser launches the loop back into the drop), rather than the last chorus; no lyric captions, although the reference had them (the style excludes text). Footage is from all seven Flash packs; median sharpness 663, faces 41% of frame height (lower than the standard on purpose: speed and lightning shots).
- 2026-11-10: **The user approved Dexter × Hold On v3 and made it the standard for future edits** (see The standard above; `tools/new_edit.py` scaffolds from it).
- 2026-11-09: Style adopted as the default for all future edits, based on the Dexter × Doomsday voidfx reference (first build: `edit/`, `dexter_doomsday_edit_v1.mp4`).
- 2026-11-10: The user rejected the glass `shatter` too and asked for a creative transition themed to the character (not blood, not forensic), such as a shape that selects an object and pans into it. It was replaced with `knife`, and the user generally likes the edit's quality. For new edits, prefer character-signature objects used as shape windows or masks.
- 2026-11-10: Feedback on the Hold On edit: the user disliked the glass light-streak glow, the floating slide graphics with the slide wipe, and the water rise, and asked for new, unique animated transitions instead. They were replaced with `shatter`, `scope` and `plastic`. The user also rejected the dark extreme close-up (teeth) after the smoke, which was replaced with the blood-room look-up shot. Prefer fresh, physical, themed animations over overlays that just glow or tint.
- 2026-11-10: Second edit, Dexter × "Hold On, We're Going Home" (`edits/hold_on/`). It established that a section is a final chorus of exactly 8 bars that loops on its downbeat, that big transitions go on the measured loudest hits, that shots are chosen by sharpness and face size (sharper sources than the first edit), and added the `confetti` transition.
- 2026-11-09: Faster pipeline. The plan lives in config.py alone, with cuts, transitions and SFX derived from it. The grade is applied once at extract time, the draft mode takes 45 s, and render plus encode is a single pass with no intermediate image files.
