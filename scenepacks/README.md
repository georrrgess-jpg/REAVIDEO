# Scene pack library

The footage itself is not in git (`media/` is ignored). This folder holds what's needed to get it back and plan with it:

- `manifest.tsv`: every stored file with its Google Drive id, size, full sha256, resolution, fps, length and original title.
- `fetch.sh`: re-downloads into `media/scenepacks/` and `media/references/` and verifies each checksum.
  - `bash scenepacks/fetch.sh` fetches everything (~800 MB).
  - `bash scenepacks/fetch.sh flash` fetches only matching names.
- `catalog/<name>.json`: detected shots (`[start, end]` seconds) for each pack, from `edit/tools/catalog.py`. To rebuild the contact sheets for picking clips, run:
  ```bash
  python edit/tools/catalog.py media/scenepacks/<name>.mp4 -o media/catalog/<name>
  ```

| File | Show | Length | fps | Shots |
|---|---|---|---|---|
| dexter_s1.mp4 | Dexter S1 (used for the first edit) | 6:47 | 60 | 117 |
| dexter_s2_a.mp4 | Dexter S2 | 5:18 | 60 | 77 |
| dexter_s2_b.mp4 | Dexter S2 | 6:51 | 60 | 150 |
| dexter_s4.mp4 | Dexter S4 | 4:03 | 60 | 97 |
| dexter_mix_1440p.mp4 | Dexter, mixed seasons | 7:07 | 60 | 189 |
| dexter_mix_a.mp4 | Dexter, short mix | 0:35 | 60 | 14 |
| flash_s1.mp4 | The Flash S1 | 4:54 | 30 | 109 |
| flash_s2_a.mp4 | The Flash S2 | 2:27 | 30 | 64 |
| flash_s2_b.mp4 | The Flash S2 | 5:09 | 60 | 139 |
| flash_s3.mp4 | The Flash S3 | 2:16 | 30 | 62 |
| flash_s5.mp4 | The Flash S5 | 6:22 | 23.976 | 221 |
| flash_all_seasons.mp4 | The Flash, all seasons | 2:31 | 60 | 72 |
| flash_blitzkrieg.mp4 | The Flash (Blitzkrieg) | 3:02 | 60 | 92 |

References (finished edits to study, not footage):

| File | What | Length |
|---|---|---|
| voidfx_passengers_love_song.mp4 | voidfx edit, 1080×1920 60 fps | 0:23 |
| flash_wonderwall_ref.mp4 | Flash edit (Wonderwall), 720×1280 30 fps; user-uploaded, **not in the manifest** (not re-downloadable) | 0:22 |

Notes:
- **Resolution:** every pack is really 1280×720, whatever its title says. Expect the 9:16 crop to look soft (about 2.7× upscale).
- **Frame rate:** the 30 fps and 23.976 fps packs are converted to 60 fps by the renderer, which repeats frames rather than interpolating. They're fine for normal-speed shots but look steppy in slow motion.
- **Shot counts:** these are from automatic detection. A few may split on flashes or merge on soft dissolves.
