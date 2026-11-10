"""Dexter x "The Less I Know The Better" (Tame Impala): bridge -> chorus, 2:07.537-2:28.050.

10 bars at 117 BPM = 80 half-beats = 20.51 s, starting on the "just don't make me" pickup and
ending as "no surprise..." finishes, so the loop keeps the meter. The same stretch the voidfx
Passenger reference uses (found by waveform correlation 0.95).

Look: black and white with selective red (only saturated reds keep colour: blood, red light,
the red string) plus coloured drawn graphics. Kinetic lyric typography (white, red keywords).
Story: the bridge is a dream (bed, stripes of light, flame) -> the knife wakes the killer ->
the M99 syringe is pushed at you, you black out and wake up to Dexter looming over you ->
the pre-drop machine-gun cuts in red -> "LOVE" on the drop -> the chorus parade -> blood-drop loop.
Loud hits (half-beats): 0, 2, 6, 14, 26, 32, 50 (drop), 66, 72.
Build: bash ../../edit/build.sh .  (or --draft). Field meanings: edit/config.py; TEXT: edit/typo.py.
"""
W, H, FPS = 1080, 1920, 60
PK = '../../media/scenepacks/'
S1, A2, B2, C4, D, E = (PK + n + '.mp4' for n in ('dexter_s1', 'dexter_s2_a', 'dexter_s2_b', 'dexter_s4', 'dexter_mix_1440p', 'dexter_mix_a'))
SCENEPACK = D
SONG = '../../media/less_i_know.mp3'
SONG_OFFSET = 127.537
GRID0, HALF = 0.0, 0.25641
FIRST_CUT = 6
TAIL = 20.513 - (79 * 0.25641 + 0.077)       # exactly 80 half-beats
PRE, POST = 0.60, 0.50
SR_MODEL = '../../media/models/realesr-general-x4v3.pth'   # ML upscaling at extract time (optional; see the skill)
KICKS = [0, 2, 6, 10, 14, 18, 21, 22, 26, 28, 32, 46, 50, 59, 62, 65, 66, 69, 70, 72, 74, 77, 79]
SNARES = [4, 8, 12, 16, 20, 24, 30, 34, 36, 38, 40, 42, 44, 48, 52, 56, 60, 64, 68, 76, 78]

BW = dict(sat=0.0, rk=1.0, rth=0.22, con=1.24, tint=(1.03, 1.0, .97))   # monochrome, saturated reds kept
def g(gain, **kw): return dict(BW, gain=gain, **kw)

SHOTS = [
 # -- bridge: "just don't make me wait forever" -- a dream
 dict(name='slide_eye',  out='ink', fx=['dust', 'glint'],          pack=S1, src=46.30,  x=.60, track=False, push=(1.00, 1.06), **g(1.05)),
 dict(name='wake_bed',   beats=4, out='blur', fx=['dust'],        pack=A2, src=18.70,  x=.46, push=(1.00, 1.07), **g(1.35)),
 dict(name='tilt_dark',  beats=4, out='orange', fx=['dust'],      pack=B2, src=328.30, x=.32, push=(1.00, 1.06), **g(1.6)),
 dict(name='flame',      beats=4, out='slices', fx=['dust'],      pack=S1, src=222.60, x=.57, push=(1.00, 1.06), **g(1.2)),
 dict(name='stripes',    beats=4, out='dip', fx=['dust'],         pack=A2, src=21.30,  x=.42, z=1.2, cy=.40, push=(1.00, 1.06), **g(1.1)),
 dict(name='upside_knife', beats=4, out='knife',                  pack=S1, src=329.00, x=.47, push=(1.00, 1.06), **g(1.35)),
 # -- build: the killer, the syringe, wake up
 dict(name='kill_room',  beats=4, out='whip', out_opts={'d': 1},  pack=E,  src=3.60,   x=.58, z=1.3, cy=.45, push=(1.00, 1.06), **g(1.25)),
 dict(name='red_string', beats=2, out='cut',                      pack=D,  src=199.90, x=.49, push=(1.00, 1.04), **g(1.0)),
 dict(name='bloody_shirt', beats=4, out='syringe', punch={},      pack=A2, src=235.50, x=.58, push=(1.00, 1.08), **g(1.3)),
 dict(name='looming',    beats=2, out='cut', fx=['ghost'],        pack=S1, src=388.10, x=.53, push=(1.06, 1.00), **g(1.15)),
 dict(name='hands_face', beats=4, out='whip', out_opts={'d': -1}, fx=['ghost', 'trails'], pack=B2, src=162.60, x=.48, push=(1.00, 1.07), **g(1.35)),
 dict(name='phone_red',  beats=2, out='cut', fx=['bump', 'rgbhit'], pack=B2, src=207.30, x=.63, push=(1.00, 1.05), **g(1.55)),
 dict(name='blood_hand', beats=2, out='cut', fx=['bump', 'rgbhit'], pack=S1, src=268.90, x=.25, track=False, push=(1.00, 1.05), **g(1.1)),
 dict(name='dark_ecu',   beats=2, out='cut', fx=['bump', 'rgbhit'], pack=A2, src=288.30, x=.29, push=(1.00, 1.05), **g(1.55)),
 dict(name='red_light',  beats=2, out='red', fx=['bump'],         pack=S1, src=81.00,  x=.52, z=1.5, cy=.42, push=(1.00, 1.06), **g(1.3)),
 # -- drop: "oh my LOVE" -- chorus: "can't you see yourself by my side?"
 dict(name='smirk',      beats=4, out='blur', fx=['bump'],        pack=S1, src=307.70, x=.39, push=(1.00, 1.06), **g(1.0)),
 dict(name='sunglasses', beats=4, out='whip', out_opts={'d': 1}, fx=['bump', 'glint'], pack=S1, src=158.20, x=.46, push=(1.00, 1.06), **g(0.95, rth=0.34)),
 dict(name='close_look', beats=4, out='drops', fx=['bump'],       pack=A2, src=3.50,   x=.44, push=(1.00, 1.06), **g(1.0)),
 dict(name='parade',     beats=4, out='confetti', fx=['bump'],    pack=D,  src=360.80, x=.36, push=(1.00, 1.06), **g(1.0)),
 dict(name='grin',       beats=3, out='cut', fx=['bump'],         pack=C4, src=45.95,  x=.34, push=(1.00, 1.05), **g(0.95)),
 dict(name='hood_white', beats=3, out='cut', fx=['bump', 'glint'], pack=B2, src=188.70, x=.46, push=(1.00, 1.05), **g(0.95)),
 dict(name='night_up',   beats=2, out='cut', punch={'dur': 0.18, 'z': 1.14, 'flash': 0.10}, pack=S1, src=380.60, x=.37, push=(1.00, 1.05), **g(1.2)),
 dict(name='sunset_end', beats=5, out='iris', fx=['dust', 'bump', 'glint'],        pack=S1, src=402.00, x=.56, push=(1.00, 1.08), **g(1.0, con=1.4)),
]
for s in SHOTS: s.setdefault('z', 1.0)

# Kinetic lyric typography (section time). Each line: words with their own entrance time.
RED = (0.18, 0.13, 0.90)
TEXT = [
 dict(y=0.735, size=70, out=2.62, exit='fade', words=[(1.06, "DON'T", 'type'), (1.40, 'MAKE', 'type'), (1.68, 'ME', 'type')]),
 dict(y=0.785, size=104, out=6.45, exit='scatter', words=[(2.10, 'WAIT', 'pop'), (2.81, 'FOREVER', 'forever', RED)]),
 dict(y=0.770, size=96, out=10.85, exit='cut', words=[(9.28, 'WAIT', 'glitch'), (9.86, 'FOREVER', 'glitch', RED)]),
 dict(y=0.735, size=78, out=12.80, exit='cut', words=[(11.03, 'OH,', 'pop'), (12.09, 'MY', 'pop')]),
 dict(y=0.610, size=300, out=13.50, exit='blur', words=[(12.80, 'LOVE', 'impact', RED)]),
 dict(y=0.735, size=64, out=18.00, exit='fade', words=[(13.55, "CAN'T", 'slide'), (13.95, 'YOU', 'slide'), (14.30, 'SEE', 'slide'), (14.55, 'YOURSELF', 'slide')]),
 dict(y=0.785, size=104, out=18.00, exit='blur', words=[(15.67, 'BY', 'pop'), (16.21, 'MY', 'pop'), (16.83, 'SIDE?', 'pop', RED)]),
]

from timing import build_cuts
CUTS, DUR = build_cuts(SHOTS, GRID0, HALF, FIRST_CUT, TAIL)
for i, s in enumerate(SHOTS):
    s['start'], s['end'] = CUTS[i], CUTS[i + 1]
