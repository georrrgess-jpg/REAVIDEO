"""TEST EDIT (style not approved; not a standard): The Flash x "See You Again" (Tyler, the Creator ft. Kali Uchis),
1:02.699-1:24.101 (7 bars, 56 half-beats of 0.382 s at 78.5 BPM, 21.40 s).

Story from the lyrics: "Can I get a kiss? / And can you make it last forever?" (Barry and Iris, a ring passing
between hands, Nora) -> "I said I'm bout to go to war" (the suit flies onto him) -> "And I don't know if I'ma see
you again" (Iris in tears) -> "I said okay..." (his eye glows) -> the drop on half-beat 36 (13.76 s): speed and
lightning; the one silent half-beat (#51, 19.49 s) freezes time; the loop races back in time to the first shot.
Pieces built from the footage: eye-to-eye match (Iris's landmarked eye into Barry's real eye), the drop through his
glowing eye (found by colour), a whip that carries the fight's measured motion (optical flow), the real lightning on
his arm followed across the frame until it splits the frame open, a time-stop on the silence, a Speed Force rewind
loop. Simple cuts between them for breathing room. Letterbox over the quiet half. Plain 720p extract (no AI upscaling).
Build: bash ../../edit/build.sh .
"""
W, H, FPS = 1080, 1920, 60
PK = '../../media/scenepacks/'
S1, S5 = (PK + n + '.mp4' for n in ('flash_s1', 'flash_s5'))
SCENEPACK = S5
SONG = '../../media/see_you_again.mp3'
SONG_OFFSET = 62.699
GRID0, HALF = 0.0, 0.38217
FIRST_CUT = 6
TAIL = 21.401 - (55 * 0.38217 + 0.077)
PRE, POST = 0.60, 0.50
KICKS = [36, 39, 41, 43, 44, 47, 49, 52, 55]
SNARES = [38, 43, 46, 50, 54]
MBLUR = True
BUMP, BUMP_KICKS, BUMP_DECAY = 0.03, [39, 44, 47, 52], 0.12
LETTERBOX = dict(h=0.08, open=[(13.62, 13.92)], close=[(20.9, 21.40)])

SOFT = dict(rk=1.0, yk=1.0, con=1.08, tint=(.95, 1.0, 1.06))   # the quiet half: warm, soft
HARD = dict(rk=1.0, yk=1.0, con=1.15, tint=(.98, 1.0, 1.03))   # the drop: punchier
def g(sat, gain, look=SOFT, **kw): return dict(look, sat=sat, gain=gain, **kw)

SHOTS = [
 dict(name='laugh', out='blur', pack=S5, src=30.00, x=0.55, push=(1.00, 1.05), **g(0.60, 1.10)),                          # with Iris, laughing
 dict(name='ring', beats=3, out='cut', pack=S5, src=35.25, x=0.50, frame='wide', push=(1.00, 1.08), **g(0.55, 1.25)),      # hands, a ring passes
 dict(name='nora', beats=5, out='cut', pack=S5, src=37.60, x=0.50, push=(1.00, 1.05), **g(0.55, 1.20)),                   # Nora, smiling
 dict(name='resolve', beats=5, out='cut', pack=S1, src=287.90, x=0.50, push=(1.00, 1.05), **g(0.55, 1.05)),               # masked, bloodied
 dict(name='suitup', beats=5, out='cut', punch={'z': 1.12, 'flash': 0.08}, pack=S5, src=46.55, x=0.50, frame='wide', push=(1.00, 1.05), **g(0.60, 1.20)),   # "go to war": the suit flies on
 dict(name='suited', beats=6, out='cut', pack=S5, src=53.60, x=0.50, push=(1.00, 1.06), **g(0.60, 1.10)),                 # in the suit, walking up
 dict(name='iris', beats=3, out='eyematch', pack=S5, src=353.30, x=0.40, push=(1.00, 1.05), **g(0.55, 1.20)),             # "see you again": Iris in tears
 dict(name='eye', beats=3, out='eyeportal', pack=S1, src=291.72, x=0.22, frame='wide', push=(1.00, 1.04), **g(0.60, 1.10)),   # his eye glows
 dict(name='burst', beats=2, out='cut', fx=['bump'], pack=S5, src=2.68, x=0.50, push=(1.00, 1.05), **g(0.65, 1.05, HARD)),       # the drop: lightning bursts off him
 dict(name='fight', beats=3, out='flowwhip', punch={'z': 1.14, 'flash': 0.08}, fx=['bump'], pack=S5, src=335.75, valid=(0, 134), x=0.50, push=(1.00, 1.05), **g(0.65, 1.05, HARD)),
 dict(name='crouch', beats=3, out='cut', fx=['bump'], pack=S5, src=67.30, x=0.50, push=(1.00, 1.05), **g(0.65, 1.05, HARD)),
 dict(name='arm', beats=2, out='boltfollow', fx=['bump'], pack=S5, src=69.30, x=0.50, frame='wide', push=(1.00, 1.04), **g(0.65, 1.00, HARD)),   # lightning on his arm
 dict(name='punch', beats=3, out='cut', fx=['bump'], pack=S5, src=339.95, valid=(0, 134), x=0.50, push=(1.00, 1.05), **g(0.65, 1.05, HARD)),
 dict(name='clash', beats=3, out='timestop', punch={'z': 1.12, 'flash': 0.08}, fx=['bump'], pack=S5, src=332.55, x=0.50, frame='wide', push=(1.00, 1.05), **g(0.65, 1.05, HARD)),
 dict(name='final', beats=3, out='timerewind', fx=['bump'], pack=S5, src=16.20, x=0.50, push=(1.00, 1.05), **g(0.60, 1.10, HARD)),   # masked, after it all
]
for s in SHOTS: s.setdefault('z', 1.0); s.setdefault('frame', 'auto')
TEXT = []

from timing import build_cuts
CUTS, DUR = build_cuts(SHOTS, GRID0, HALF, FIRST_CUT, TAIL)
for i, s in enumerate(SHOTS):
    s['start'], s['end'] = CUTS[i], CUTS[i + 1]
