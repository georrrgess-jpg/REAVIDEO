"""Dexter x "New Person, Same Old Mistakes" (Tame Impala): second chorus, 2:41.523-3:00.470.

6 bars at 76 BPM = 48 half-beats = 18.95 s, chorus into chorus, so it loops. The song's kick falls on
the odd half-beats (5, 13, 21, 29, 37, 45), so the cuts run in a 3-3-2 pattern that lands on every kick.
v2: every shot moves (ranked by optical flow), automatic face framing, Venetian blinds instead of
the mirror doors, and a VHS rewind loop (the edit rewinds to its first frame: same old mistakes).
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
 # every shot has real movement inside it (ranked by optical flow, tools/motion.py); framing is automatic
 dict(name='sun_glasses', out='kaleido',                       pack=S1, src=156.90, x=.46, push=(1.00, 1.05), **g(.55, 1.0)),
 dict(name='poncho_walk', beats=3, out='whip', out_opts={'d': 1}, pack=D, src=97.60, x=.50, valid=(0, 136), push=(1.00, 1.04), **g(.55, 1.0)),
 dict(name='lab_walk',   beats=3, out='tape',                   pack=B2, src=172.30, x=.45, push=(1.00, 1.04), **g(.50, 1.0)),
 dict(name='slide_lamp', beats=2, out='cut',                    pack=B2, src=268.60, x=.55, push=(1.00, 1.04), **g(.55, 1.3)),
 dict(name='hood_up',    beats=3, out='blur', punch={},         pack=S1, src=303.40, x=.50, push=(1.00, 1.04), **g(.55, 1.0)),
 dict(name='smirk_walk', beats=3, out='rorschach',              pack=C4, src=43.60,  x=.42, push=(1.00, 1.04), **g(.50, 1.05)),
 dict(name='containers', beats=2, out='cut',                    pack=S1, src=372.60, x=.42, push=(1.00, 1.04), **g(.50, 1.0)),
 dict(name='car_glance', beats=3, out='blinds', punch={},       pack=D,  src=26.60,  x=.40, push=(1.00, 1.04), **g(.50, 1.05)),
 dict(name='parade',     beats=3, out='whip', out_opts={'d': -1}, pack=D, src=361.20, x=.36, push=(1.00, 1.04), **g(.55, 1.0)),
 dict(name='hood_white', beats=2, out='cut',                    pack=B2, src=188.40, x=.46, push=(1.00, 1.04), **g(.50, 0.95)),
 dict(name='laugh',      beats=3, out='plastic', punch={},      pack=D,  src=38.15,  x=.55, push=(1.00, 1.04), **g(.50, 1.05)),
 dict(name='night_walk', beats=3, out='drops',                  pack=C4, src=81.80,  x=.45, push=(1.00, 1.04), **g(.50, 1.3)),
 dict(name='blood_hand', beats=2, out='red',                    pack=S1, src=272.40, x=.62, push=(1.00, 1.04), **g(.60, 1.1)),
 dict(name='suburb_turn', beats=3, out='knife',                 pack=B2, src=281.40, x=.55, valid=(13, 112), push=(1.00, 1.04), **g(.50, 1.0)),
 dict(name='look_up',    beats=3, out='blur',                   pack=S1, src=355.80, x=.45, push=(1.00, 1.04), **g(.50, 1.25)),
 dict(name='profile_turn', beats=4, out='rewind',               pack=S1, src=226.60, x=.40, push=(1.00, 1.05), **g(.50, 1.0)),
]
for s in SHOTS: s.setdefault('z', 1.0); s.setdefault('frame', 'auto')

from timing import build_cuts
CUTS, DUR = build_cuts(SHOTS, GRID0, HALF, FIRST_CUT, TAIL)
for i, s in enumerate(SHOTS):
    s['start'], s['end'] = CUTS[i], CUTS[i + 1]
