"""Dexter x "fragment" (slowed): THE KILL RITUAL. The riser into the second drop, 0:39.159-0:58.741.

8 bars at 98 BPM = 64 half-beats = 19.58 s. Bar 1 is the riser with the song's only vocal, "I'm gonna kill you!"
(0.9-1.9 s), typed on screen; the 808 drop lands a half-beat early (#7), so the cuts sit on the odd half-beats
where the bass hits, with quick pairs on the loud #15/#16 and the projector set-piece on the dense #47-#51.
Built from the Dexter standard (edits/x_gon) with an all-new kit (kit V): every transition is a step of
Dexter's ritual -- the bone saw rips the frame open on the drop, zip ties cinch it, the leather tool roll
unrolls (dive into the cleaver), the fingerprint card, the black contractor bag swallows it, the kill-room slide
projector shows the victims (projector, click, lights up), the heart monitor flatlines like a dying CRT, and frost
creeps over the lens (the Ice Truck Killer) until a wipe reveals the start: the loop.
Look: cold night-ritual grade (blue-black shadows, reds kept), red heartbeat pulses on the big kicks, soft
bumps, motion blur in every transition.
Build: bash ../../edit/build.sh .
"""
W, H, FPS = 1080, 1920, 60
PK = '../../media/scenepacks/'
S1, A2, B2, C4, D = (PK + n + '.mp4' for n in ('dexter_s1', 'dexter_s2_a', 'dexter_s2_b', 'dexter_s4', 'dexter_mix_1440p'))
SCENEPACK = D
SONG = '../../media/fragment.mp3'
SONG_OFFSET = 39.159
GRID0, HALF = 0.0, 0.30597
FIRST_CUT = 7
TAIL = 19.582 - (63 * 0.30597 + 0.077)          # exactly 64 half-beats
PRE, POST = 0.60, 0.50
KICKS = [0, 4, 7, 8, 12, 15, 16, 20, 23, 24, 26, 31, 32, 34, 38, 39, 42, 44, 46, 47, 48, 52, 55, 56, 60, 63]
SNARES = [10, 14, 36, 58, 62]
MBLUR = True
BUMP, BUMP_KICKS, BUMP_DECAY = 0.03, [0, 4, 7, 15, 16, 23, 31, 32, 39, 44, 47, 48, 55, 56, 63], 0.12

NIGHT = dict(con=1.22, tint=(1.08, 1.0, .92))                  # cold, blue-black night ritual; reds kept
def g(sat, gain, **kw): return dict(NIGHT, sat=sat * 0.8, rk=1.0, gain=gain, **kw)

SHOTS = [
 dict(name='dark_turn', out='saw', fx=['pulse'], pack=D, src=26.51, x=0.38, push=(1.00, 1.05), **g(0.50, 1.25)),   # D7, 3.8s, motion 2.6
 dict(name='kill_suit', beats=4, out='ziptie', fx=['bump', 'pulse'], pack=D, valid=(0, 138), src=97.61, x=0.43, push=(1.00, 1.05), **g(0.55, 1.00)),   # D48, 3.1s, motion 4.0
 dict(name='night_walk', beats=4, out='cut', fx=['bump'], pack=C4, src=81.96, x=0.50, push=(1.00, 1.05), **g(0.50, 1.35)),   # C27, 3.4s, motion 2.8
 dict(name='bw_stare', beats=2, out='cut', punch={'z': 1.25, 'flash': 0.18}, fx=['bump'], pack=B2, src=188.44, x=0.46, push=(1.00, 1.05), **g(0.30, 1.05)),   # B73, 1.8s, motion 2.3
 dict(name='grin', beats=2, out='whip', out_opts={'d': 1}, fx=['bump'], pack=D, src=38.39, x=0.55, push=(1.00, 1.05), **g(0.50, 1.00)),   # D13, 1.9s, motion 2.1
 dict(name='sunglasses', beats=4, out='toolroll', fx=['bump'], pack=S1, src=158.60, x=0.46, push=(1.00, 1.05), **g(0.50, 1.00)),   # S37, 6.5s, motion 2.5
 dict(name='office_turn', beats=4, out='cut', fx=['bump'], pack=B2, src=172.20, x=0.48, push=(1.00, 1.05), **g(0.50, 0.95)),   # B65, 2.5s, motion 4.6
 dict(name='match_light', beats=4, out='fingerprint', fx=['bump', 'pulse'], pack=B2, src=268.58, x=0.57, push=(1.00, 1.05), **g(0.60, 1.15)),   # B101, 2.3s, motion 3.2
 dict(name='orange_stare', beats=4, out='whip', out_opts={'d': -1}, fx=['bump'], pack=D, src=135.59, x=0.64, push=(1.00, 1.05), **g(0.50, 1.00)),   # D71, 1.8s, motion 1.2
 dict(name='gloves_bag', beats=4, out='bag2', fx=['bump'], pack=S1, src=334.42, x=0.48, push=(1.00, 1.05), **g(0.50, 1.00)),   # S87, 6.3s, motion 0.8
 dict(name='dark_smile', beats=4, out='cut', fx=['bump', 'pulse'], pack=C4, src=43.49, x=0.40, push=(1.00, 1.05), **g(0.50, 1.25)),   # C14, 1.7s, motion 2.7
 dict(name='look_up_night', beats=4, out='projector', punch={'z': 1.22}, fx=['bump'], pack=S1, src=355.57, x=0.62, push=(1.00, 1.05), **g(0.50, 1.30)),   # S91, 1.7s, motion 1.5
 dict(name='slide_1', beats=2, out='click', projected=True, pack=D, src=53.41, x=0.44, push=(1.00, 1.05), **g(0.50, 1.05)),   # D21, 2.6s, motion 1.9
 dict(name='slide_2', beats=2, out='lightsup', projected=True, pack=D, src=142.74, x=0.55, push=(1.00, 1.05), **g(0.50, 1.10)),   # D77, 2.2s, motion 1.2
 dict(name='monitor_turn', beats=4, out='flatline', fx=['ekg'], pack=D, src=145.64, x=0.53, push=(1.00, 1.05), **g(0.50, 1.00)),   # D79, 3.7s, motion 1.3
 dict(name='car_glance', beats=4, out='cut', fx=['bump'], pack=D, src=276.47, x=0.47, push=(1.00, 1.05), **g(0.50, 1.05)),   # D139, 2.8s, motion 0.9
 dict(name='last_lean', beats=4, out='frost', fx=['bump'], pack=C4, src=158.42, x=0.64, push=(1.00, 1.05), **g(0.50, 1.15)),   # C61, 3.9s, motion 0.8
]
for s in SHOTS: s.setdefault('z', 1.0); s.setdefault('frame', 'auto')

# The song's one line, typed over the riser.
RED = (0.18, 0.13, 0.90)
TEXT = [
 dict(y=0.74, size=96, out=2.10, exit='fade', words=[(0.94, "I'M", 'type'), (1.20, 'GONNA', 'type')]),
 dict(y=0.82, size=150, out=2.10, exit='fade', words=[(1.44, 'KILL', 'impact', RED), (1.80, 'YOU', 'pop', RED)]),
]

from timing import build_cuts
CUTS, DUR = build_cuts(SHOTS, GRID0, HALF, FIRST_CUT, TAIL)
for i, s in enumerate(SHOTS):
    s['start'], s['end'] = CUTS[i], CUTS[i + 1]
