"""Dexter x "New Person, Same Old Mistakes" (Tame Impala): second chorus, 2:41.523-3:00.470.

6 bars at 76 BPM = 48 half-beats = 18.95 s, chorus into chorus, so it loops. The song's kick falls on
the odd half-beats (5, 13, 21, 29, 37, 45), so the cuts run in a 3-3-2 pattern that lands on every kick.
Theme: the "new person" (friendly Dexter) against the same old killer. Four new transitions about
duality and psychedelia: kaleidoscope, crime-scene tape, Rorschach blot, two faces (mirror doors);
plus the approved knife, kill-room plastic, blood spatter, red flash and the blood-drop loop.
Footage: 720p packs with plain bicubic resizing (no AI upscaling, at the user's request).
Build: bash ../../edit/build.sh .
"""
W, H, FPS = 1080, 1920, 60
PK = '../../media/scenepacks/'
S1, A2, B2, C4, D = (PK + n + '.mp4' for n in ('dexter_s1', 'dexter_s2_a', 'dexter_s2_b', 'dexter_s4', 'dexter_mix_1440p'))
SCENEPACK = D
SONG = '../../media/new_person.mp3'
SONG_OFFSET = 161.523
GRID0, HALF = 0.0, 0.39474
FIRST_CUT = 5
TAIL = 18.947 - (47 * 0.39474 + 0.077)          # exactly 48 half-beats
PRE, POST = 0.60, 0.50
KICKS = [5, 13, 21, 23, 24, 29, 37, 45]
SNARES = [0, 3, 4, 7, 8, 10, 11, 15, 16, 19, 20, 25, 27, 28, 31, 32, 35, 36, 39, 43, 44, 47]

MUTED = dict(con=1.12, tint=(1.04, 1.0, .96))
def g(sat, gain, **kw): return dict(MUTED, sat=sat, rk=1.0, gain=gain, **kw)

SHOTS = [
 dict(name='smile',      out='kaleido', fx=['bump'],            pack=C4, src=66.60,  x=.48, push=(1.00, 1.07), **g(.55, 1.0)),
 dict(name='night',      beats=3, out='whip', out_opts={'d': 1}, pack=D, src=103.60, x=.53, push=(1.00, 1.06), **g(.45, 1.35)),
 dict(name='red_ecu',    beats=3, out='tape',                   pack=B2, src=111.40, x=.54, push=(1.00, 1.06), **g(.50, 1.25)),
 dict(name='smirk',      beats=2, out='cut',                    pack=D,  src=36.35,  x=.51, push=(1.00, 1.05), **g(.45, 1.35)),
 dict(name='green',      beats=3, out='blur', punch={},         pack=S1, src=152.40, x=.45, push=(1.00, 1.06), **g(.60, 1.2)),
 dict(name='fire_face',  beats=3, out='rorschach',              pack=C4, src=196.90, x=.46, push=(1.00, 1.06), **g(.50, 1.0)),
 dict(name='slides',     beats=2, out='cut',                    pack=S1, src=264.70, x=.50, push=(1.00, 1.05), **g(.50, 1.1)),
 dict(name='kill_room',  beats=3, out='mirror', punch={},       pack=A2, src=224.40, x=.61, push=(1.00, 1.06), **g(.50, 1.25)),
 dict(name='spatter_room', beats=3, out='whip', out_opts={'d': -1}, pack=D, src=106.40, x=.58, push=(1.00, 1.06), **g(.55, 1.0)),
 dict(name='lab_photos', beats=2, out='cut',                    pack=B2, src=178.50, x=.58, push=(1.00, 1.05), **g(.50, 1.0)),
 dict(name='red_shirt',  beats=3, out='plastic', punch={},      pack=D,  src=160.00, x=.34, push=(1.00, 1.06), **g(.50, 1.3)),
 dict(name='go_dexter',  beats=3, out='drops', fx=['bump'],     pack=D,  src=419.40, x=.46, push=(1.00, 1.06), **g(.55, 1.0)),
 dict(name='kill_stare', beats=2, out='red',                    pack=B2, src=322.53, x=.37, push=(1.00, 1.05), **g(.50, 1.4)),
 dict(name='bloody',     beats=3, out='knife',                  pack=D,  src=380.00, x=.60, push=(1.00, 1.06), **g(.50, 1.3)),
 dict(name='smirk_s1',   beats=3, out='blur', fx=['bump'],      pack=S1, src=307.60, x=.39, push=(1.00, 1.06), **g(.50, 1.05)),
 dict(name='profile',    beats=4, out='iris', fx=['bump'],      pack=S1, src=227.40, x=.40, push=(1.00, 1.07), **g(.50, 1.0)),
]
for s in SHOTS: s.setdefault('z', 1.0)

from timing import build_cuts
CUTS, DUR = build_cuts(SHOTS, GRID0, HALF, FIRST_CUT, TAIL)
for i, s in enumerate(SHOTS):
    s['start'], s['end'] = CUTS[i], CUTS[i + 1]
