"""Dexter x "X Gon' Give It To Ya" (DMX), v2: the hook, 1:07.878-1:28.088.

8 bars at 95 BPM = 64 half-beats = 20.21 s. The hook is two identical 4-bar phrases ("first we gonna ROCK,
then we gonna ROLL, then we let it POP, don't let it GO / X gon' give it to ya, he gon' give it to ya"), so the
edit is built as two mirrored halves whose transitions act out the words (word times from the vocal stem):
  ROCK  -> punch-in + lyric hit          ROCK  -> punch-in + lyric hit
  ROLL  -> duct-tape rolls tape it shut  ROLL  -> the print rolls itself up
  POP   -> mugshot camera flash          POP   -> three shots punch a range target
  GO    -> punch-in + lyric hit          GO    -> punch-in + lyric hit
  X     -> a knife carves an X           X     -> a spray-painted X
  ...   -> handcuffs lock, luminol       ...   -> whip, cut, vinyl scratch back to the start (loop)
The opener is a subwoofer pumping on the bass. Look: grittier than the standard (more contrast, less colour),
soft zoom bumps on the strongest kicks, motion blur in every transition (v3 polish). Started from the Dexter standard (edits/new_person) but with an all-new kit (kit IV).
Build: bash ../../edit/build.sh .
"""
W, H, FPS = 1080, 1920, 60
PK = '../../media/scenepacks/'
S1, A2, B2, C4, D, E = (PK + n + '.mp4' for n in ('dexter_s1', 'dexter_s2_a', 'dexter_s2_b', 'dexter_s4', 'dexter_mix_1440p', 'dexter_mix_a'))
SCENEPACK = D
SONG = '../../media/x_gon.mp3'
SONG_OFFSET = 67.878
GRID0, HALF = 0.0, 0.31579
FIRST_CUT = 4
TAIL = 20.211 - (63 * 0.31579 + 0.077)          # exactly 64 half-beats
PRE, POST = 0.60, 0.50
KICKS = [0, 4, 8, 12, 16, 17, 18, 19, 20, 24, 27, 28, 32, 36, 40, 41, 43, 44, 48, 49, 50, 51, 52, 56, 57, 60]
SNARES = [6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62]

MBLUR = True                                                  # motion blur (two sub-frames) inside every transition
BUMP, BUMP_KICKS, BUMP_DECAY = 0.028, [0, 8, 16, 24, 32, 40, 48, 52, 56, 60], 0.12   # soft zoom bumps on the strongest kicks only
GRIT = dict(con=1.20, tint=(1.03, 1.0, .97))                  # harder and greyer than the standard; reds kept
def g(sat, gain, **kw): return dict(GRIT, sat=sat * 0.85, rk=1.0, gain=gain, **kw)

SHOTS = [
 dict(name='street_smile', out='sub', fx=['bump'], pack=C4, src=48.27, x=0.48, push=(1.00, 1.05), **g(0.55, 1.00)),
 dict(name='dark_hall', beats=2, out='cut', fx=['bump'], pack=D, src=123.55, x=0.61, push=(1.00, 1.05), **g(0.50, 1.35)),
 dict(name='crime_crowd', beats=4, out='ducttape', punch={}, fx=['bump'], pack=C4, src=85.54, x=0.50, push=(1.00, 1.05), **g(0.55, 1.00)),
 dict(name='hands_face', beats=4, out='mugshot', fx=['bump'], pack=A2, src=303.45, x=0.51, push=(1.00, 1.05), **g(0.50, 1.15)),
 dict(name='office_walk', beats=4, out='cut', fx=['bump'], pack=B2, src=124.44, x=0.42, push=(1.00, 1.05), **g(0.50, 1.00)),
 dict(name='mangroves', beats=2, out='xslash', punch={}, fx=['bump'], pack=S1, src=176.03, x=0.48, push=(1.00, 1.05), **g(0.50, 1.05)),
 dict(name='smile', beats=4, out='cuffs', fx=['bump'], pack=C4, src=67.22, x=0.48, push=(1.00, 1.05), **g(0.55, 1.00)),
 dict(name='car', beats=4, out='whip', out_opts={'d': 1}, fx=['bump'], pack=B2, src=276.98, x=0.69, push=(1.00, 1.05), **g(0.50, 1.05)),
 dict(name='files_walk', beats=4, out='luminol', fx=['bump'], pack=A2, src=176.94, x=0.46, push=(1.00, 1.05), **g(0.50, 1.00)),
 dict(name='night', beats=4, out='whip', out_opts={'d': -1}, fx=['bump'], pack=C4, src=24.68, x=0.52, push=(1.00, 1.05), **g(0.50, 1.35)),
 dict(name='upside_saw', beats=2, out='cut', fx=['bump'], pack=S1, src=329.48, x=0.47, push=(1.00, 1.05), **g(0.50, 1.30)),
 dict(name='lean', beats=4, out='rollup', punch={}, fx=['bump'], pack=C4, src=161.72, x=0.38, push=(1.00, 1.05), **g(0.50, 1.10)),
 dict(name='look_up', beats=4, out='target', fx=['bump'], pack=D, src=46.40, x=0.50, push=(1.00, 1.05), **g(0.50, 1.00)),
 dict(name='shadows', beats=4, out='cut', fx=['bump'], pack=C4, src=78.97, x=0.42, push=(1.00, 1.05), **g(0.50, 1.20)),
 dict(name='spatter_room', beats=2, out='spray', punch={}, fx=['bump'], pack=D, src=106.70, x=0.58, push=(1.00, 1.05), **g(0.60, 1.00)),
 dict(name='slides', beats=4, out='whip', out_opts={'d': 1}, fx=['bump'], pack=S1, src=264.42, x=0.50, push=(1.00, 1.05), **g(0.55, 1.10)),
 dict(name='bushes', beats=4, out='cut', fx=['bump'], pack=S1, src=340.62, x=0.58, push=(1.00, 1.05), **g(0.55, 1.05)),
 dict(name='night_look', beats=3, out='vinyl', fx=['bump'], pack=S1, src=380.33, x=0.37, push=(1.00, 1.05), **g(0.50, 1.00)),
]
for s in SHOTS: s.setdefault('z', 1.0); s.setdefault('frame', 'auto')

# Lyric hits on the hook's words (section time, from the vocal stem): white, the X in red.
RED = (0.18, 0.13, 0.90)
TEXT = [
 dict(y=0.74, size=230, out=2.66, exit='blur', words=[(1.90, 'ROCK', 'impact')]),
 dict(y=0.78, size=104, out=5.99, exit='fade', words=[(5.18, "DON'T", 'type'), (5.36, 'LET', 'type'), (5.50, 'IT', 'type'), (5.69, 'GO', 'impact', RED)]),
 dict(y=0.80, size=92, out=7.56, exit='fade', words=[(6.32, 'X', 'impact', RED), (6.55, "GON'", 'pop'), (6.98, 'GIVE', 'pop'), (7.12, 'IT', 'pop'), (7.26, 'TO', 'pop'), (7.40, 'YA', 'pop')]),
 dict(y=0.74, size=230, out=12.80, exit='blur', words=[(12.00, 'ROCK', 'impact')]),
 dict(y=0.78, size=104, out=16.00, exit='fade', words=[(15.28, "DON'T", 'type'), (15.46, 'LET', 'type'), (15.60, 'IT', 'type'), (15.79, 'GO', 'impact', RED)]),
 dict(y=0.80, size=92, out=19.15, exit='fade', words=[(17.86, 'HE', 'slide'), (17.98, "GON'", 'slide'), (18.40, 'GIVE', 'slide'), (18.50, 'IT', 'slide'), (18.62, 'TO', 'slide'), (18.76, 'YA', 'pop', RED)]),
]

from timing import build_cuts
CUTS, DUR = build_cuts(SHOTS, GRID0, HALF, FIRST_CUT, TAIL)
for i, s in enumerate(SHOTS):
    s['start'], s['end'] = CUTS[i], CUTS[i + 1]
