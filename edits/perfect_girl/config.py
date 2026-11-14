"""Dexter x "The Perfect Girl" (Mareux): MIAMI NIGHT DRIVE. The build into the second groove, 1:49.931-2:08.536.

The song is 129 BPM synthwave with a half-time feel; the grid here steps on its beats (0.465 s), so the cuts glide
rather than chatter. 5 of sections.py's bars = 40 steps = 18.6 s: one bar of the no-hats build, the crash and the
full groove at step 8 (3.72 s), another crash at step 24 (11.16 s), and the end of the phrase loops back into the build.
The song is effectively instrumental (the vocal stem is synth bleed), so there is no lyric text.
Built from the Dexter standard (edits/x_gon) with a new, deliberately restrained kit (kit VI): a night drive
through Miami -- a palm trunk passing the lens, the outrun horizon on the drop (striped sun, neon grid), a tinted
car window rolling down, sodium streetlights sweeping over, rain and the windshield wiper on the crash, and the
rear-view mirror for the loop (what's behind you is the start; OBJECTS IN MIRROR ARE CLOSER THAN THEY APPEAR).
Look: neon-noir grade (a little magenta in the lights, reds kept), soft bumps on the strongest kicks, motion blur.
Plain 720p extract, no AI upscaling (user's request).
Build: bash ../../edit/build.sh .
"""
W, H, FPS = 1080, 1920, 60
PK = '../../media/scenepacks/'
S1, A2, B2, C4, D = (PK + n + '.mp4' for n in ('dexter_s1', 'dexter_s2_a', 'dexter_s2_b', 'dexter_s4', 'dexter_mix_1440p'))
SCENEPACK = D
SONG = '../../media/perfect_girl.mp3'
SONG_OFFSET = 109.931
GRID0, HALF = 0.0, 0.46512
FIRST_CUT = 4
TAIL = 18.605 - (39 * 0.46512 + 0.077)          # exactly 40 steps
PRE, POST = 0.60, 0.50
KICKS = [8, 11, 12, 13, 15, 16, 17, 19, 20, 21, 22, 24, 26, 28, 30, 34, 35, 40]
SNARES = [9, 11, 14, 20, 21, 22, 23, 25, 27, 35, 36, 37]
MBLUR = True
BUMP, BUMP_KICKS, BUMP_DECAY = 0.025, [8, 15, 17, 20, 22, 24, 28, 35], 0.14

NEON = dict(con=1.16, tint=(1.05, 0.96, 1.03))                 # neon noir: a touch of magenta in the light
def g(sat, gain, **kw): return dict(NEON, sat=sat * 0.9, rk=1.0, gain=gain, **kw)

SHOTS = [
 dict(name='night_turn',  out='pass', fx=['bump'],                        pack=A2, src=189.60, x=0.44, push=(1.00, 1.05), **g(0.55, 1.65)),   # A39 night walk, turns
 dict(name='neon_drive',  beats=4, out='outrun', fx=['bump'],             pack=B2, src=276.69, x=0.69, push=(1.00, 1.05), **g(0.60, 1.10)),   # B103 driving past neon
 dict(name='sunglasses',  beats=4, out='cut', fx=['bump'],                pack=S1, src=157.60, x=0.46, push=(1.00, 1.05), **g(0.55, 1.00)),   # S37 sunglasses on, Miami sun
 dict(name='smile_down',  beats=2, out='cut', punch={'z': 1.12, 'flash': 0.08}, fx=['bump'], pack=D, src=29.65, x=0.47, push=(1.00, 1.04), **g(0.55, 1.05)),   # D8
 dict(name='turn_stare',  beats=2, out='window', fx=['bump'],             pack=D,  src=115.00, x=0.61, push=(1.00, 1.04), **g(0.55, 1.00)),   # D56
 dict(name='night_car',   beats=4, out='streetlights', fx=['bump'],       pack=D,  src=276.15, x=0.47, push=(1.00, 1.05), **g(0.55, 1.15)),   # D139 in the car at night
 dict(name='corridor',    beats=4, out='wiper', fx=['bump'],              pack=S1, src=353.30, x=0.48, push=(1.00, 1.05), **g(0.55, 1.00)),   # S90
 dict(name='top_lit',     beats=4, out='cut', fx=['bump'],                pack=C4, src=24.35,  x=0.52, push=(1.00, 1.05), **g(0.55, 1.30)),   # C7 lit from above (the crash)
 dict(name='dark_smile',  beats=2, out='cut', fx=['bump'],                pack=C4, src=43.60,  x=0.40, push=(1.00, 1.04), **g(0.55, 1.25)),   # C14
 dict(name='sunset',      beats=2, out='whip', out_opts={'d': 1}, fx=['bump'], pack=S1, src=402.15, x=0.56, push=(1.00, 1.04), **g(0.60, 1.00)),   # S1 sunset
 dict(name='sun_profile', beats=4, out='cut', fx=['bump'],                pack=S1, src=227.30, x=0.40, push=(1.00, 1.05), **g(0.55, 0.95)),   # S56
 dict(name='car_glance',  beats=3, out='mirror', fx=['bump'],             pack=D,  src=26.60,  x=0.38, push=(1.00, 1.05), **g(0.55, 1.05)),   # D7 in the car, glancing over
]
for s in SHOTS: s.setdefault('z', 1.0); s.setdefault('frame', 'auto')

from timing import build_cuts
CUTS, DUR = build_cuts(SHOTS, GRID0, HALF, FIRST_CUT, TAIL)
for i, s in enumerate(SHOTS):
    s['start'], s['end'] = CUTS[i], CUTS[i + 1]
