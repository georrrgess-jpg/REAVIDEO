"""The Flash x "Mr. Blue Sky" (Electric Light Orchestra): the full-band singalong, 2:31.720-2:53.372.

8 bars at 88.67 BPM = 64 half-beats = 21.65 s; the phrase ends where the song drops, so it loops.
Built from the speedster standard (edits/flash) and aimed above it: every shot ML-upscaled, a bright
sunny grade instead of the muted one, and four new drawn animations themed to the song and the
character: a sunburst, a rolling cumulus cloud bank, the Flash running by as a lightning comet, and a
clock with lightning-bolt hands that rewinds time to loop the edit.
Arc: Barry under a blue sky -> the speed kicks in -> the emblem -> lightning -> pure joy (grins,
confetti) -> warm smile -> time rewinds to the start.
Build: bash ../../edit/build.sh .  then  python ../../edit/extract.py --sr  for the final.
"""
W, H, FPS = 1080, 1920, 60
PK = '../../media/scenepacks/'
A1, B2, C2, D3, E5, F, G = (PK + n + '.mp4' for n in ('flash_s1', 'flash_s2_a', 'flash_s2_b', 'flash_s3', 'flash_s5', 'flash_all_seasons', 'flash_blitzkrieg'))
SCENEPACK = F
SONG = '../../media/blue_sky.mp3'
SONG_OFFSET = 151.720
GRID0, HALF = 0.0, 0.33831
FIRST_CUT = 4
TAIL = 21.652 - (63 * 0.33831 + 0.077)          # exactly 64 half-beats
PRE, POST = 0.60, 0.50
SR_MODEL = '../../media/models/realesr-general-x4v3.pth'
KICKS = [1, 3, 4, 6, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23, 25, 26, 27, 28, 29, 30, 33, 34, 35, 36, 37, 39, 41, 42, 44, 45, 46, 49, 53]
SNARES = [2, 7, 11, 17, 24, 31, 32, 38, 40, 43, 47, 48, 50, 51, 52, 54, 56, 57, 58, 60, 61]

SUN = dict(sat=.78, rk=1.0, yk=1.0, con=1.08, tint=(.97, 1.0, 1.04))     # bright and warm, red suit and gold kept
def g(gain, **kw): return dict(SUN, gain=gain, **kw)

SHOTS = [
 # -- "blue sky": Barry looks up, the sun bursts, clouds roll, the Flash runs by
 dict(name='sky_look',   out='sunburst',                        pack=G,  src=94.05,  x=.37, push=(1.00, 1.06), **g(1.0)),
 dict(name='towers',     beats=2, out='whip', out_opts={'d': 1}, pack=A1, src=277.60, x=.57, valid=(20, 106), push=(1.00, 1.05), **g(1.05)),
 dict(name='car_park',   beats=2, out='clouds',                 pack=B2, src=35.00,  x=.64, push=(1.00, 1.05), **g(1.0)),
 dict(name='suit_run',   beats=4, out='runby', fx=['bump'],     pack=F,  src=4.40,   x=.72, push=(1.00, 1.06), **g(1.0)),
 dict(name='wall_run',   beats=2, out='cut',                    pack=G,  src=158.40, x=.45, track=False, push=(1.00, 1.05), **g(1.0)),
 dict(name='sunglasses', beats=4, out='phase', punch={}, fx=['bump'], pack=D3, src=28.80, x=.63, push=(1.00, 1.06), **g(1.15)),
 # -- the speed and the emblem
 dict(name='lightning_face', beats=2, out='whip', out_opts={'dur': 0.14, 'vertical': True, 'amount': 0.5}, pack=F, src=117.30, x=.55, push=(1.00, 1.05), **g(0.95)),
 dict(name='road_pov',   beats=2, out='cut',                    pack=G,  src=63.75,  x=.50, track=False, push=(1.00, 1.10), **g(0.95)),
 dict(name='charge',     beats=4, out='emblem', punch={}, fx=['bump'], pack=F, src=90.50, x=.60, push=(1.00, 1.07), **g(1.05)),
 dict(name='red_face',   beats=2, out='cut', fx=['eyes'],       pack=E5, src=187.00, x=.41, push=(1.00, 1.05), **g(1.05)),
 dict(name='run_trails', beats=2, out='blur', fx=['echo'],      pack=E5, src=108.90, x=.30, push=(1.00, 1.05), **g(1.0)),
 dict(name='throw',      beats=4, out='strike', fx=['crackle'], pack=F,  src=127.60, x=.66, push=(1.00, 1.05), **g(1.0)),
 dict(name='suit_grin',  beats=2, out='cut', fx=['bump'],       pack=D3, src=44.30,  x=.52, push=(1.00, 1.05), **g(1.0)),
 dict(name='purple',     beats=2, out='whip', out_opts={'d': -1}, pack=E5, src=304.00, x=.69, push=(1.00, 1.05), **g(1.0)),
 # -- pure joy
 dict(name='speedforce', beats=4, out='speedlines', out_opts={'d': 1}, fx=['bump'], pack=D3, src=132.85, x=.35, push=(1.00, 1.06), **g(1.1)),
 dict(name='sun_blast',  beats=2, out='cut',                    pack=C2, src=6.50,   x=.50, track=False, push=(1.00, 1.05), **g(0.95)),
 dict(name='blue_core',  beats=2, out='cut', punch={},          pack=F,  src=51.60,  x=.58, push=(1.00, 1.05), **g(1.05)),
 dict(name='wow',        beats=4, out='confetti', fx=['bump'],  pack=E5, src=127.45, x=.61, push=(1.00, 1.06), **g(1.05)),
 dict(name='arms_crossed', beats=2, out='whip', out_opts={'d': 1}, pack=F, src=92.60, x=.33, push=(1.00, 1.05), **g(1.05)),
 dict(name='lab_run',    beats=4, out='blur', fx=['echo'],      pack=E5, src=264.20, x=.51, push=(1.00, 1.06), **g(0.95)),
 dict(name='suit_smile', beats=2, out='cut',                    pack=E5, src=98.20,  x=.61, push=(1.00, 1.05), **g(1.1)),
 dict(name='warm_smile', beats=5, out='clock',                  pack=C2, src=93.50,  x=.42, push=(1.00, 1.08), **g(1.1)),
]
for s in SHOTS: s.setdefault('z', 1.0)

from timing import build_cuts
CUTS, DUR = build_cuts(SHOTS, GRID0, HALF, FIRST_CUT, TAIL)
for i, s in enumerate(SHOTS):
    s['start'], s['end'] = CUTS[i], CUTS[i + 1]
