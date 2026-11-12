"""Dexter x "X Gon' Give It To Ya" (DMX): hook break to hook break, 1:28.093-1:48.303.

8 bars at 95 BPM = 64 half-beats = 20.21 s; the next bar is the hook break again, so the loop lands on a hook.
Built from the Dexter standard (edits/new_person): every shot moves (tools/motion.py), auto face framing,
custom-graphic transitions (badge, tape, scope, newspaper, evidence bag, blinds, plastic, crown, red, knife,
case file) and the slide-box loop. Cuts land on this song's kicks (even half-beats) and snares.
Build: bash ../../edit/build.sh .
"""
W, H, FPS = 1080, 1920, 60
PK = '../../media/scenepacks/'
S1, A2, B2, C4, D, E = (PK + n + '.mp4' for n in ('dexter_s1', 'dexter_s2_a', 'dexter_s2_b', 'dexter_s4', 'dexter_mix_1440p', 'dexter_mix_a'))
SCENEPACK = D
SONG = '../../media/x_gon.mp3'
SONG_OFFSET = 88.093
GRID0, HALF = 0.0, 0.31579
FIRST_CUT = 4
TAIL = 20.211 - (63 * 0.31579 + 0.077)          # exactly 64 half-beats
PRE, POST = 0.60, 0.50
KICKS = [0, 8, 12, 16, 17, 19, 24, 25, 27, 28, 32, 40, 43, 44, 45, 48, 49, 50, 51, 56, 57, 60]
SNARES = [6, 10, 14, 18, 26, 30, 34, 38, 42, 46, 54, 58, 62]

MUTED = dict(con=1.12, tint=(1.04, 1.0, .96))
def g(sat, gain, **kw): return dict(MUTED, sat=sat, rk=1.0, gain=gain, **kw)

SHOTS = [
 dict(name='street_smile', out='badge', pack=C4, src=48.27, x=0.48, push=(1.00, 1.04), **g(0.55, 1.00)),  # C17 len 4.1s motion 1.7
 dict(name='crime_crowd', beats=4, out='whip', out_opts={'d': 1}, pack=C4, src=85.54, x=0.50, push=(1.00, 1.04), **g(0.55, 1.00)),  # B20 len 4.8s motion 2.9
 dict(name='office_walk', beats=2, out='tape', pack=B2, src=124.75, x=0.42, push=(1.00, 1.04), **g(0.50, 1.00)),  # E5 len 3.2s motion 1.9
 dict(name='slides', beats=4, out='scope', out_opts={'px': 0.5, 'py': 0.55}, pack=S1, src=264.42, x=0.50, push=(1.00, 1.04), **g(0.55, 1.10)),  # S65 len 5.8s motion 1.2
 dict(name='bushes', beats=4, out='newspaper', pack=S1, src=340.62, x=0.58, push=(1.00, 1.04), **g(0.55, 1.05)),  # S88 len 6.2s motion 2.5
 dict(name='shadows', beats=4, out='bag', pack=C4, src=78.97, x=0.42, push=(1.00, 1.04), **g(0.50, 1.20)),  # C26 len 2.5s motion 2.2
 dict(name='mangroves', beats=2, out='cut', pack=S1, src=176.03, x=0.48, push=(1.00, 1.04), **g(0.50, 1.05)),  # S42 len 3.6s motion 1.9
 dict(name='car', beats=4, out='blinds', fx=['idbox'], pack=B2, src=276.98, x=0.69, push=(1.00, 1.04), **g(0.50, 1.05)),  # B103 len 5.6s motion 2.0
 dict(name='smile', beats=4, out='whip', out_opts={'d': -1}, pack=C4, src=67.22, x=0.48, push=(1.00, 1.04), **g(0.55, 1.00)),  # C23 len 3.2s motion 1.9
 dict(name='dark_hall', beats=2, out='cut', pack=D, src=123.55, x=0.61, push=(1.00, 1.04), **g(0.50, 1.35)),  # D59 len 2.0s motion 1.5
 dict(name='hands_face', beats=4, out='cut', pack=A2, src=303.45, x=0.51, push=(1.00, 1.04), **g(0.50, 1.15)),  # A73 len 2.8s motion 1.3
 dict(name='files_walk', beats=4, out='plastic', punch={}, pack=A2, src=176.94, x=0.46, push=(1.00, 1.04), **g(0.50, 1.00)),  # A36 len 6.8s motion 3.3
 dict(name='night', beats=4, out='crown', fx=['idbox'], pack=C4, src=24.68, x=0.52, push=(1.00, 1.04), **g(0.50, 1.35)),  # C7 len 2.4s motion 1.2
 dict(name='spatter_room', beats=2, out='red', pack=D, src=106.70, x=0.58, push=(1.00, 1.04), **g(0.60, 1.00)),  # D52 len 2.9s motion 1.1
 dict(name='lean', beats=4, out='knife', pack=C4, src=161.72, x=0.38, push=(1.00, 1.04), **g(0.50, 1.10)),  # C62 len 2.7s motion 1.0
 dict(name='look_up', beats=4, out='casefile', pack=D, src=46.40, x=0.50, push=(1.00, 1.04), **g(0.50, 1.00)),  # D18 len 2.9s motion 1.0
 dict(name='upside_saw', beats=2, out='cut', pack=S1, src=329.48, x=0.47, push=(1.00, 1.04), **g(0.50, 1.30)),  # S86 len 4.2s motion 1.0
 dict(name='night_look', beats=5, out='slidebox', pack=S1, src=380.33, x=0.37, push=(1.00, 1.04), **g(0.50, 1.00)),  # S116 len 1.5s motion 0.7
]
for s in SHOTS: s.setdefault('z', 1.0); s.setdefault('frame', 'auto')

from timing import build_cuts
CUTS, DUR = build_cuts(SHOTS, GRID0, HALF, FIRST_CUT, TAIL)
for i, s in enumerate(SHOTS):
    s['start'], s['end'] = CUTS[i], CUTS[i + 1]
