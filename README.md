# REAVIDEO

A beat-synced vertical fan edit rebuilt in code from a reference edit's style.

- `docs/style-breakdown.md`: what the reference edit does (timing, transitions, look).
- `.claude/skills/voidfx-edit/SKILL.md`: the house editing style (the default for every new edit) and the workflow for building one.
- `edit/tools/`: `beatgrid.py` (tempo, half-beat grid and section for any song) and `catalog.py` (shot detection and contact sheets for any scene pack).
- `edits/x_gon/config.py`: **the Dexter standard** (Dexter × X Gon' Give It To Ya v3): the hook as two mirrored halves whose drawn transitions act out the lyrics (subwoofer, duct tape, mugshot, knife X, handcuffs, luminol, roll-up, range target, spray X), lyric hits, motion blur, vinyl-scratch loop. `new_edit.py` defaults to it.
- `edits/new_person/config.py`: the previous Dexter standard (New Person v6): ID badge, newspaper, evidence bag, case file, slide-box loop.
- `edits/hold_on/config.py`: the earlier Dexter standard (Dexter × Hold On v3).
- `edits/less_i_know/config.py`: Dexter × The Less I Know The Better, the flagship candidate (ML-upscaled footage, kinetic lyrics, black-and-white with selective red, ink, orange, slices and syringe transitions).
- `edits/flash/config.py`: **the speedster standard** (Flash × Slaughter House v3), the template for powered-character edits (`new_edit.py --from flash`). Approved as adequate, not yet at the target quality. Start one with `python edit/tools/new_edit.py <name> --song ... --section ...` and build with `bash edit/build.sh edits/<name>`.
- `edit/`: the shared renderer, plus the first edit's config. `config.py` is the whole edit plan (clips, half-beat lengths, transition names), `timing.py` turns it into cut times, `timeline.py` holds the transitions, `fx.py` the effect primitives and procedural graphics, `audio.py` the soundtrack.

Footage and music are not included. Put them at `media/scenepack.mp4` and `media/song.mp3`, then:

```bash
bash edit/build.sh           # final
```

```bash
bash edit/build.sh --draft   # 30 fps review copy
```

The result is written to `edit/out/edit.mp4` (1080×1920, 60 fps, 22.65 s), or `edit/out/edit_draft.mp4` for the draft.
