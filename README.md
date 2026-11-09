# REAVIDEO

A beat-synced vertical fan edit rebuilt in code from a reference edit's style.

- `docs/style-breakdown.md`: what the reference edit does (timing, transitions, look).
- `edit/`: the renderer. `config.py` holds the cut times and clip list, `timeline.py` the transitions, `fx.py` the effect primitives and procedural graphics, `audio.py` the soundtrack.

Footage and music are not included. Put them at `media/scenepack.mp4` and `media/song.mp3`, then:

```bash
bash edit/build.sh
```

The result is written to `edit/out/dexter_doomsday_edit_v1.mp4` (1080×1920, 60 fps, 22.65 s).
