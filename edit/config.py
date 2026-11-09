"""Edit plan. Change this file (and only this file) to make a new edit.

Timing is in half-beats of the song: each shot lasts `beats` half-beats (2, 3 or 4; one 5-6 hold
allowed). Shot 0 runs from 0 to grid point FIRST_CUT. `out` names the transition into the next
shot (see TRANSITIONS in timeline.py); its exact timing, and its sound effect, are worked out
from the cut. `fx` adds overlays on a shot; `punch` adds a punch-in hit on the shot's first frame.

Per-shot look: src = in-point in the scene pack (s) at slot start; x = crop-centre hint
(fraction of width); z = crop tightness; cy = crop centre y when z > 1; track=False keeps the
crop fixed; valid=(lo, hi) overrides the usable frame range. Grade: sat, rk (red keep), con,
tint (b, g, r), gain; push = zoom at slot start/end. Grade is baked in by extract.py, so after
changing a shot's grade or framing re-run: python extract.py <shot index>.
"""
W, H, FPS = 1080, 1920, 60
SCENEPACK = '../media/scenepack.mp4'   # source footage, any 16:9 resolution
SONG = '../media/song.mp3'
SONG_OFFSET = 65.643                    # section start in the song, from tools/beatgrid.py
GRID0, HALF = 0.311, 0.31446            # first half-beat and half-beat length (beatgrid.py: grid[0], period / 2)
FIRST_CUT = 2                           # shot 0 ends on this grid point
TAIL = 0.25                             # video runs this long past the last cut (loop transition)
PRE = 0.60   # source handle before slot start (s)
POST = 0.50  # source handle after slot end (s)

SHOTS = [
 dict(name='white_bg', out='leaf', out_opts={'ang': -38, 'seed': 3},   src=377.12, x=.25, z=1.0, sat=.85, rk=.9, con=1.08, tint=(.96,1.0,1.05), gain=1.0, push=(1.00,1.07)),
 dict(name='palm_walk', beats=3, out='leaf', out_opts={'ang': 162, 'seed': 9},  src=213.05, x=.66, z=1.0, sat=.75, rk=.8, con=1.08, tint=(1.0,1.0,1.0), gain=.98, push=(1.00,1.06)),
 dict(name='car_walk', beats=3, out='blur',   src=58.20, x=.57, z=1.0, sat=.60, rk=.7, con=1.10, tint=(1.0,1.0,.98), gain=.95, push=(1.06,1.00)),
 dict(name='hood', beats=3, out='glass', fx=['streaks'],       src=303.75, x=.50, z=1.0, sat=.50, rk=.6, con=1.10, tint=(1.03,1.0,.97), gain=1.25, push=(1.00,1.07)),
 dict(name='flame', beats=4, out='smoke',      src=223.35, x=.56, z=1.0, sat=.55, rk=1.0, con=1.15, tint=(1.02,1.0,.98), gain=.95, push=(1.00,1.06)),
 dict(name='eye', beats=2, out='dip', fx=['fringe'],        src=122.25, x=.60, z=1.0, sat=.30, rk=.5, con=1.2,  tint=(1.02,1.0,1.0), gain=.80, push=(1.10,1.16)),
 dict(name='slide', beats=4, out='slidewipe', fx=['slides'],      src=47.00,  x=.46, z=1.0, sat=.35, rk=1.0, con=1.10, tint=(1.05,1.0,.95), gain=1.45, push=(1.00,1.06)),
 dict(name='red_bg', beats=3, out='blood',     src=272.45, x=.75, track=False, z=1.0, sat=.95, rk=1.0, con=1.10, tint=(1.0,1.0,1.0), gain=1.40, push=(1.00,1.07)),
 dict(name='blood_face', beats=3, out='blur', src=346.30, x=.47, z=1.0, sat=.45, rk=1.0, con=1.15, tint=(1.0,1.0,1.0), gain=.95, push=(1.08,1.00)),
 dict(name='kill_room', beats=4, out='spin',  src=131.55, x=.59, z=1.45, cy=.42, sat=.20, rk=.3, con=1.15, tint=(1.05,1.0,.95), gain=.95, push=(1.00,1.07)),
 dict(name='camera', beats=4, out='polaroid',     src=127.00, x=.47, z=1.0, sat=.45, rk=.6, con=1.10, tint=(1.0,1.0,1.0), gain=.95, push=(1.00,1.06)),
 dict(name='coffee', beats=4, out='drops',     src=333.50, x=.49, z=1.0, sat=.30, rk=.4, con=1.12, tint=(1.05,1.0,.95), gain=.92, push=(1.00,1.06)),
 dict(name='spatter', beats=2, out='whip', out_opts={'d': 1},    src=84.00,  x=.37, z=1.0, sat=.40, rk=1.0, con=1.12, tint=(1.0,1.0,1.0), gain=.95, push=(1.00,1.05)),
 dict(name='shock', beats=2, out='blur',      src=290.80, x=.21, z=1.0, sat=.55, rk=.9, con=1.12, tint=(1.0,1.0,1.0), gain=.92, push=(1.00,1.05)),
 dict(name='corridor', beats=4, out='cut',   src=350.95, x=.54, z=1.25, cy=.40, sat=.30, rk=.4, con=1.15, tint=(1.05,1.0,.95), gain=.92, push=(1.00,1.08)),
 dict(name='smirk', beats=4, out='red', punch={},      src=308.60, x=.30, z=1.0, sat=.50, rk=.6, con=1.10, tint=(1.0,1.0,1.0), gain=.95, push=(1.00,1.06)),
 dict(name='red_drive', beats=3, out='whip', out_opts={'dur': 0.14, 'vertical': True, 'amount': 0.5},  src=317.00, x=.43, z=1.0, sat=.95, rk=1.0, con=1.15, tint=(1.0,1.0,1.0), gain=1.15, push=(1.00,1.06)),
 dict(name='back_turn', beats=2, out='cut',  src=358.72, x=.52, z=1.0, sat=.45, rk=1.0, con=1.15, tint=(1.0,1.0,1.0), gain=1.0, push=(1.00,1.05)),
 dict(name='blue_room', beats=2, out='whip', out_opts={'d': -1}, punch={'dur': 0.18, 'z': 1.14, 'flash': 0.1},  src=256.60, x=.46, z=1.40, cy=.45, sat=.80, rk=.8, con=1.2,  tint=(1.05,1.0,.95), gain=1.25, push=(1.00,1.06)),
 dict(name='look_up', beats=5, out='glass', out_opts={'ang': 148, 'rev': True, 'bright': 0.8}, fx=['leak_bubbles'],    src=380.30, x=.23, z=1.0, sat=.30, rk=.5, con=1.15, tint=(1.04,1.0,.96), gain=.80, push=(1.00,1.09)),
 dict(name='boat', beats=4, out='water',       src=260.50, x=.60, z=1.30, cy=.50, valid=(0,999), sat=.15, rk=.2, con=1.12, tint=(1.06,1.0,.95), gain=.95, push=(1.00,1.06)),
 dict(name='fence', beats=3, out='iris',      src=367.20, x=.33, z=1.0, sat=.30, rk=.5, con=1.12, tint=(1.02,1.0,.97), gain=.80, push=(1.00,1.06)),
]

# ---- derived timing (do not edit) ----
from timing import build_cuts
CUTS, DUR = build_cuts(SHOTS, GRID0, HALF, FIRST_CUT, TAIL)
for i, s in enumerate(SHOTS):
    s['start'], s['end'] = CUTS[i], CUTS[i + 1]
