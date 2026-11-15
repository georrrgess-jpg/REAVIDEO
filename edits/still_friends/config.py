"""The Flash x "ARE WE STILL FRIENDS?" (Tyler, the Creator): the bridge's last breath into the final chorus,
2:33.633-2:52.910.

5 of sections.py's bars (40 steps of 0.482 s, 124 BPM) = 19.28 s. The music almost stops (2.2-2.9 s), then the
full chorus explodes on step 6 -- the drop -- and the loop falls back into the hush. Vocals: "are we still friends?"
at ~8.5-11.5 s and ~12.5-15.5 s (no on-screen text).
Built from the speedster standard (edits/flash) with an all-new kit (kit VII), aimed above it: the particle
accelerator's dark-matter wave on the drop, Barry's speed tornado, Flash Time (the world freezes, his gold trail
streaks through), Barry's ring popping open (the drawn emblem as its lid), and a multiverse breach as the loop.
Look: warm cinematic grade with red and gold kept, letterbox over the hush that opens on the drop, soft bumps on the
big kicks, motion blur in every transition. Plain 720p extract (no AI upscaling).
Build: bash ../../edit/build.sh .
"""
W, H, FPS = 1080, 1920, 60
PK = '../../media/scenepacks/'
S1, S2B, S3, S5, ALL, BLZ = (PK + n + '.mp4' for n in ('flash_s1', 'flash_s2_b', 'flash_s3', 'flash_s5', 'flash_all_seasons', 'flash_blitzkrieg'))
SCENEPACK = ALL
SONG = '../../media/still_friends.mp3'
SONG_OFFSET = 153.633
GRID0, HALF = 0.0, 0.48193
FIRST_CUT = 6
TAIL = 19.277 - (39 * 0.48193 + 0.077)          # exactly 40 steps
PRE, POST = 0.60, 0.50
KICKS = [6, 8, 10, 14, 18, 22, 26, 30, 34, 38]
SNARES = [10, 16, 20, 24, 28, 32, 36]
MBLUR = True
BUMP, BUMP_KICKS, BUMP_DECAY = 0.03, [6, 10, 22, 26, 34, 38], 0.12
LETTERBOX = dict(h=0.09, open=[(2.80, 3.05)], close=[(18.80, 19.20)])

WARM = dict(rk=1.0, yk=1.0, con=1.12, tint=(.97, 1.0, 1.05))   # warm cinematic; Flash red and gold kept
def g(sat, gain, **kw): return dict(WARM, sat=sat, gain=gain, **kw)

SHOTS = [
 dict(name='bloodied', out='accel', fx=['bump'], pack=S1, src=271.95, x=0.40, push=(1.00, 1.05), **g(0.55, 1.15)),   # F197 4.6s m1.9
 dict(name='blue_core', beats=4, out='cut', fx=['bump'], pack=ALL, src=44.92, x=0.60, push=(1.00, 1.05), **g(0.60, 1.05)),   # FA13 5.0s m1.3
 dict(name='yell', beats=2, out='whip', out_opts={'d': 1}, fx=['bump'], pack=ALL, src=127.85, x=0.72, push=(1.00, 1.05), **g(0.60, 1.00)),   # FA64 1.7s m3.3
 dict(name='mask_close', beats=4, out='tornado', fx=['bump'], pack=ALL, src=110.20, x=0.50, push=(1.00, 1.05), **g(0.60, 1.00)),   # FA7 5.2s m3.4
 dict(name='mask_day', beats=2, out='cut', fx=['bump'], pack=S2B, src=3.64, x=0.53, push=(1.00, 1.05), **g(0.55, 1.00)),   # F21 1.9s m1.4
 dict(name='run', beats=4, out='flashtime', fx=['bump', 'echo'], pack=S5, src=16.49, x=0.43, push=(1.00, 1.05), **g(0.55, 1.00)),   # F510 2.9s m4.4
 dict(name='suit_bolts', beats=2, out='cut', fx=['bump'], pack=ALL, src=90.37, x=0.57, push=(1.00, 1.05), **g(0.60, 1.00)),   # FA48 2.3s m1.2
 dict(name='mask_dark', beats=4, out='ringpop', fx=['bump'], pack=S5, src=269.07, x=0.44, push=(1.00, 1.05), **g(0.55, 1.25)),   # F5147 6.8s m2.1
 dict(name='eyes', beats=2, out='cut', fx=['bump'], pack=ALL, src=117.47, x=0.55, push=(1.00, 1.05), **g(0.55, 1.05)),   # FA59 2.9s m2.0
 dict(name='fire_mask', beats=2, out='cut', punch={'z': 1.18, 'flash': 0.12}, fx=['bump'], pack=S5, src=181.88, x=0.45, push=(1.00, 1.05), **g(0.60, 1.05)),   # F5116 5.1s m2.3
 dict(name='blue_mask', beats=2, out='whip', out_opts={'d': -1}, fx=['bump'], pack=BLZ, src=22.45, x=0.50, push=(1.00, 1.05), **g(0.55, 1.30)),   # FB2 2.0s m4.0
 dict(name='fence_look', beats=3, out='cut', fx=['bump'], pack=S1, src=85.45, x=0.50, push=(1.00, 1.05), **g(0.55, 1.05)),   # F124 2.6s m1.8
 dict(name='blz_unmasked', beats=2, out='breach', fx=['bump'], pack=BLZ, src=131.55, x=0.31, push=(1.00, 1.05), **g(0.55, 1.15)),   # FB64 1.9s m1.7
]
for s in SHOTS: s.setdefault('z', 1.0); s.setdefault('frame', 'auto')
TEXT = []

from timing import build_cuts
CUTS, DUR = build_cuts(SHOTS, GRID0, HALF, FIRST_CUT, TAIL)
for i, s in enumerate(SHOTS):
    s['start'], s['end'] = CUTS[i], CUTS[i + 1]
