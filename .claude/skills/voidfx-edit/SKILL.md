---
name: voidfx-edit
description: The user's preferred editing style for short vertical fan edits ("voidfx" style). The current Dexter STANDARD is "X Gon' Give It To Ya" v3 (edits/x_gon); the speedster standard is edits/flash. New edits start from a standard with edit/tools/new_edit.py. Use for ANY request to make, remake, recut or restyle an edit, TikTok/Reels/Shorts edit, fan edit, scene-pack edit or music edit, even if the style isn't named. Covers the style rules (beat-grid timing, themed transitions, grade, framing, loop ending, SFX level), the build workflow with the tools in edit/, and the QA checklist.
---

# voidfx edit style (the user's default)

Every edit made in this repo uses this style unless the user asks otherwise. It was first
reverse-engineered from the reference `doomsday_voidfx_9x16` (analysis: `docs/style-breakdown.md`;
first build: `edit/config.py`).

## The Dexter standard (user-approved, 2026-11-12): `edits/x_gon`

**Dexter × "X Gon' Give It To Ya" v3 is the standard for Dexter edits** ("I like this edit, make it a standard").
Start new Dexter edits from it (`python edit/tools/new_edit.py <name> --song ... --section ...`, `--from x_gon` is the default).
It keeps every New Person rule below (moving shots, auto framing, drawn physical transitions, previews, QA) and adds:
- **A new kit for every song.** Never reuse the previous edit's transitions (the user rejected X Gon' v1 for that). Design drawn, physical, labelled objects from the song's own words, sound and mood (Kit IV is the model; see its table and the song-tailoring method). Whips, cuts and punch-ins are glue only.
- **Act out the song:** pick the section the listener knows (the hook), transcribe the vocal stem for word times, and land each key word's transition on the nearest half-beat; when the hook repeats, mirror the plan with a different object for the same word.
- **Sparse lyric hits** (`TEXT`): a few key words, impact/type/pop, white with red keywords, fading out; only on punch-in shots.
- **Polish defaults:** `MBLUR = True` (motion blur inside every transition), soft zoom bumps on the strongest kicks only (`BUMP = 0.028`, `BUMP_KICKS`, `BUMP_DECAY = 0.12`), no snare colour-split hits, a gritty grade (con 1.2, sat ×0.85, reds kept).
- **The loop is a thematic object that reads instantly** (the vinyl record: fast pull-back plus a tilted turntable view). If a loop object isn't recognisable in its first few frames, fix it.
- **Rhythm:** 18 slots of 2 and 4 half-beats on even half-beats (first cut at 4, last slot 3 plus the tail), punch-ins on the lyric's key words.

## The previous Dexter standard (user-approved, 2026-11-12): `edits/new_person`

**Dexter × "New Person, Same Old Mistakes" v6 is the standard for Dexter (and grounded-character) edits.**
The user liked the edit and all its transitions. Start new Dexter edits from it:
```bash
python edit/tools/new_edit.py <name> --song media/<song>.mp3 --section media/<name>_section.json
```
(`--from new_person` is the default.) What makes it the standard; each point is a rule:
- **Every shot moves.** Pick shots with `python edit/tools/motion.py <packs> --scores media/dexter_scores.json`, using motion ≥ 0.6. The QA "still steps" count (frame-to-frame change < 0.3) must be 0. No shot may read as a still picture.
- **Faces are framed on point:** `frame='auto'` on every shot (crop sized so the face is about 30% of the height, eyes in the upper third, tracking both axes).
- **Every transition is a custom drawn graphic, physical and labelled:** `badge` (opener), `tape`, `newspaper`, `bag`, `blinds`, `plastic`, `crown`, `red`, `knife`, `casefile`, plus the `slidebox` loop. Plain whips and cuts are glue only (2 whips and 4 cuts in 16 shots).
- **Overlays are sparse:** `idbox` face-lock on 3 shots, all white text. No stamps or evidence markers (the user rejected them here).
- **The loop is a thematic graphic** (`slidebox`: the edit becomes his slide collection), not just an effect.
- **Rhythm:** the section is a chorus of 6 bars (48 half-beats; 18.95 s at 76 BPM). Cuts fall on the kicks, which land on off-beats here, so slot lengths run 3-3-2 (first cut at half-beat 5).
- **Footage:** plain 720p extract (no AI upscaling, per the user). Grade sat 0.45–0.6, rk 1.0, con 1.12, tint (1.04, 1.0, 0.96), gain lifted on night shots.
- **Quality bar:** hard cuts within one frame of the beat; no black or frozen frames; still steps 0; seamless loop seam; every transition previewed with `render.py sheet`, and fixed if any edge, colour or text clips.
- **Every new edit gets its own transitions, made for its song** (user, 2026-11-12: X Gon' v1 "reuses too many of the same transitions"). Take the standard's structure, rhythm, framing and quality bar, but not its kit: build a fresh set of drawn, physical graphics from the song's own words and sound (transcribe the vocal stem, see Kit IV), and reuse at most one or two earlier transitions.

## Earlier Dexter standard (user-approved, 2026-11-10): `edits/hold_on`

**`edits/hold_on/config.py`, Dexter × "Hold On, We're Going Home" v3, was the first standard; `new_person` supersedes it.** The user
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

Blue-sky kit (first used in `edits/blue_sky`, Flash × Mr. Blue Sky):
- `sunburst` (`out_opts` sx, sy = sun position): a drawn sun rises from a corner with 16 rotating rays, a corona, an anamorphic streak and hexagonal lens ghosts; the light floods white and the next shot develops out of the glare. Use it on sky shots.
- `clouds`: a lit cumulus bank (height-field shading, warm crowns, blue-grey bellies, silver lining, two parallax layers) rolls across, with the next shot behind it.
- `runby`: the hero crosses the frame as a lightning comet (white-hot head, red-gold tail, lightning in the wake, wind smear), revealing the next shot in his wake.
- `clock` (loop): a drawn clock with lightning-bolt hands pops in, spins backwards faster and faster while the world swirls, then its face opens and the camera falls through it into shot 0. "Time rewinds" to loop; good for speedsters.
- Section note: when the chorus finder is unreliable (long instrumental outros), map vocals with `rea-video stems`, then pick the 8 bars of the strongest full-band vocal stretch that end at a section change.

Duality kit (first used in `edits/new_person`, Dexter × New Person, Same Old Mistakes):
- `kaleido`: the frame folds into a turning 8-way kaleidoscope with colour fringes and unfolds into the next shot. Psychedelic; good for Tame Impala-style songs.
- `tape`: layered yellow "CRIME SCENE DO NOT CROSS" tape whips in lengthwise until it covers the frame, then rips fully clear.
- `rorschach`: a symmetric ink blot opens from the centre fold with an ink rim; the next shot is inside it.
- ~~`strings`~~ (the user disliked it as an opener): blood-spatter string analysis. Stains splat around the frame, pinned red strings shoot to the point of origin and twang taut, then the frame bursts open from that point. A strong Dexter opener.
- `bag`: the frame shrinks into a drawn EVIDENCE bag (header label, plastic sheen, zip), a red seal sweeps the zip, and the bag falls away onto the next shot.
- `crown`: a glossy blood drop falls, hits, throws a crown of spikes and droplets, and the spreading pool becomes the next shot.
- ~~`bloodwater`~~ (the user disliked it): the frame goes underwater (tint, caustics, wobble), crimson clouds bloom with bright wispy edges, then thin out onto the next shot.
- `newspaper`: a drawn front page (MIAMI TRIBUNE, BAY HARBOR BUTCHER STRIKES AGAIN, subhead, columns) spins in and slaps down, then the camera pushes into its halftone photo of the next shot, whose dots dissolve into colour as it fills the screen. The headline auto-fits the page.
- `badge` (opener): Dexter's drawn lanyard ID badge (Miami Metro Police, Forensics, name, title, barcode) drops in and swings to rest, a glint crosses the laminate, then the camera pushes into the 9:16 photo window, which plays the next shot 1:1 at full zoom.
- `slidebox` (loop): the last shot shrinks into the blood drop of a glass slide, pull back to the rosewood slide box (one labelled slide per shot of the edit), race up past them all, then dive into slide No. 01's drop, which is shot 0. The edit becomes his trophy collection.
- ~~`photo`~~ (the user disliked it as an opener): a forensic camera viewfinder (brackets, case number, REC, ISO line, scale) frames the shot, focus pulls, six aperture blades snap shut, flash, and they open on the next shot.
- `casefile`: a manila case-file cover (typed MORGAN, DEXTER tab, CONFIDENTIAL stamp) swings shut on its hinge, then opens onto the next shot.
- Fill overlays (graphics on holds, timed to the shot's first hits). **The user did not want the stamps or markers in New Person (they don't dislike them, just not there). Use them sparingly and only where they fit.** `stamp:TEXT` (a worn red rubber stamp slams in, e.g. SUSPECT, CLEARED, GUILTY, CASE CLOSED, telling a story across the edit); `markers` (yellow numbered evidence tents pop in on hits; numbering continues across shots); `idbox` (forensic face-lock brackets snap onto the face and track it, with typed SUBJECT and MATCH readout).
- `blinds`: drawn Venetian blinds tilt shut top to bottom (lit slats, sun bars through the gaps), then tilt open onto the next shot. Noir.
- `rewind` (loop): the whole edit plays backwards at speed with a VHS look (row jitter, tracking band, chroma bleed, scanlines) and lands exactly on frame 0. Use it when the song is about repetition.
- ~~`mirror`~~ (the user disliked it): the face folds into a perfect mirror double (two faces), a light seam splits it, and the halves swing open like perspective doors.
- Rhythm note: when the kick falls on odd half-beats, cut in 3-3-2 groups so every kick gets a cut.
- Footage note: all stored packs are really 720p. The user asked for 1080p clips without AI upscaling, chose "720p, no AI" when told no 1080p exists, and asked for efficiency. Default to plain extract (no `--sr`) unless asked; ask for 1080p+ packs when quality matters.

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

**Dexter kit IV: "X Gon' Give It To Ya"** (`edits/x_gon` v2). Street-crime objects that act out the hook's words. Every graphic is drawn per frame (no stock assets).

| name | what | lyric / use | length |
|---|---|---|---|
| `sub` | a subwoofer (gunmetal basket, bolts, rubber surround, cone, glossy dust cap with a red X, "XG-15 2000W" plate) rises over the shot and pumps on the beat with pressure rings; the cap blows out and the camera dives through the cone | opener, the bass | 0.55 + 0.30 s |
| `ducttape` | three silver duct-tape rolls ("DUCT TAPE 48mm x 55m" on the core) roll across in alternating directions, taping the shot shut; strips are ripped off on the hit | "ROLL" | 0.40 + 0.36 s |
| `mugshot` | booking camera pre-flash, then POP; the next shot appears as a mugshot (bleached flash, height chart, "MORGAN, DEXTER" board pushed up into frame) | "POP" | 0.26 + 0.48 s |
| `xslash` | a blade carves a giant X in two slashes (white-hot, cooling to bleeding gashes); the four wedges fall away | "X gon' give it to ya" | 0.32 + 0.40 s |
| `cuffs` | a handcuff (bow, toothed strand, lock box, chain) swings round the face, ratchets shut, the world outside darkens, the camera dives through the ring | after the X | 0.42 + 0.36 s |
| `luminol` | lights out, a UV torch sweeps the frame and luminol makes hidden spatter glow blue; the scene flares into the next shot | phrase downbeat (loudest hit) | 0.45 + 0.35 s |
| `rollup` | the shot is a print that rolls itself up from the bottom (evidence-photo printing on its back) | second "ROLL" | 0.40 + 0.14 s |
| ~~`bulb`~~ bare bulb that pops into shards. **The user disliked the light-breaking animation; use `target`.** | | | |
| `target` | a shooting-range target (MIAMI METRO POLICE, B-27 silhouette, LANE 07) is run up to camera; POP POP POP: three shots punch ragged, powder-burned holes with torn white edges (the next shot shows through), then it is yanked back down the range and fades | second "POP" | 0.52 + 0.42 s |
| `spray` | a red X spray-painted in two strokes (overspray, sheen, runs dripping); the X swells open onto the next shot | second "X" | 0.42 + 0.35 s |
| `vinyl` (loop) | (v3: fast ease-out pull-back plus a 34-degree turntable tilt, so it reads as a record within a few frames; the user couldn't tell it was a record in v2) the last shot shrinks into the photo label of a spinning record on a turntable (strobe-dot platter, "SIDE A · MIAMI METRO RECORDS" label), the tonearm drops the needle, the record is scratched back (the label becomes shot 1), the camera dives into the label | loop ending | 0.85 + 0.10 s |

**Song-tailoring method (from X Gon'):** separate the vocals (`rea-video stems` on a short clip of the section), transcribe the vocal stem with word timestamps (`faster-whisper` `small.en`, `word_timestamps=True`, pass the known lyric as `initial_prompt`; feed audio as an array), then put each key word's transition cut on the half-beat nearest the word. If the hook is two identical phrases, mirror the plan: the same word gets a different object each time. Add sparse lyric hits with `TEXT` (impact/type/pop, keywords red) on punch-in shots only.

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
- 2026-11-12: **The user approved X Gon' v3 and made it the Dexter standard** ("I like this edit make it a standard"). `new_edit.py` now defaults to `x_gon`.
- 2026-11-12: Dexter × "X Gon' Give It To Ya" (DMX, `edits/x_gon`). v1 (verse section 1:28.1–1:48.3, New Person's kit reused) was rejected: **"reuses too many of the same transitions from the previous edit"**; the user asked for my own twist and new style, more tailored to the song, with new animations, graphics and transitions. v2: moved to the hook (1:07.878–1:28.088, 8 bars at 95 BPM, two identical 4-bar phrases), all-new Kit IV acting out the words (ROCK punch, ROLL tape/roll-up, POP mugshot/bulb, X knife/spray, cuffs, luminol, vinyl loop), lyric hits, grittier grade (con 1.2, sat ×0.85), zoom bumps on kicks and colour hits on the punch shots. `sections.py` picks the outro here (3:11.7); choose the hook by hand.
- 2026-11-12: X Gon' v2 feedback: the loop's start didn't read as a record (fixed: fast pull-back, tilted turntable view, stronger vinyl sheen, spindle); the user disliked the light-breaking (bulb) animation (replaced with `target`); "smooth and polish the edit". Polish for v3: motion blur in every transition (`MBLUR = True`, two sub-frames, 180-degree shutter), soft zoom bumps only on the strongest kicks (`BUMP`, `BUMP_KICKS`, `BUMP_DECAY`), no snare colour-split hits, lyric text fades out instead of cutting. Treat these as the default polish for future edits.
- 2026-11-12: **The user made New Person, Same Old Mistakes v6 the Dexter standard, and likes all its transitions** (badge, tape, newspaper, evidence bag, blinds, plastic, crown, red flash, knife, case file, slide-box loop). `new_edit.py` defaults to it.
- 2026-11-12: New Person v5. Opener `photo` replaced with `badge`; red stamps and evidence markers removed ("clunky" counters); the VHS rewind loop wasn't creative enough, so it was replaced with `slidebox`. Loops should be visually rich, thematic graphics, not just an effect.
- 2026-11-12: New Person v4. The user disliked the strings opener and the blood-in-water transition, and **loved the evidence-bag graphic**: physical, labelled forensic objects are the direction. They asked to fill the dead space with animations and graphics. Added `photo`, `casefile`, `stamp:`, `markers` and `idbox`. Every hold now carries a graphic.
- 2026-11-12: New Person v3. The user disliked the kaleidoscope opener and found most of the other transitions lackluster. They want **every transition to be unique and tied to a custom drawn graphic**. Replaced kaleido, rorschach, drops and a blur with `strings`, `bag`, `crown` and `bloodwater`. Treat plain blur, whip and dip as glue only, and keep them rare.
- 2026-11-12: Feedback on New Person v1. **No shots that read as still pictures; every shot must visibly move** (a head turn, an action, a camera move), so pick shots with `tools/motion.py` (optical flow; aim for motion >= 0.6, the top half of a pack). **Faces must be framed on point**, so use `frame='auto'` (extract.py sizes the crop so the face is about 30% of the height with the eyes in the upper third, and follows the face vertically too). The mirror-doors transition is rejected; `blinds` replaces it. The blood-drop `iris` loop felt lame here; `rewind` replaced it. Prefer loops that mean something for the song.
- 2026-11-12: Dexter × New Person, Same Old Mistakes (`edits/new_person`): the user asked for the best Dexter edit yet, with stunning unique transitions, no AI upscaling (1080p clips preferred; none exist, so 720p with plain resizing) and an efficient build. Final `dexter_new_person_edit_v1.mp4`: 18.95 s at 60 fps, 74.5 MB, built in about 1 hour with no upscaling pass.
- 2026-11-11: The user asked for the best Flash edit yet, with custom graphics and animations, set to Mr. Blue Sky. Built `edits/blue_sky` (final `flash_blue_sky_edit_v1.mp4`: 21.65 s at 60 fps, 84.8 MB, hard cuts 4–14 ms from the beat, no frozen or black frames): a bright sunny grade (sat .78) instead of the muted speedster look, ML-upscaled footage, and the blue-sky kit above.
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
