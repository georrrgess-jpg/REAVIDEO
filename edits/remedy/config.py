"""The Flash x "Remedy" (RJ Pasin): the build into the drop, 1:01.470-1:20.670.

5 of sections.py's bars (40 steps of 0.48 s, 125 BPM) = 19.20 s. The 808 is out for the first 7 steps (a hushed
build), stutters on step 7 and drops on step 8 (3.84 s); after that it plays 3-on/1-off blocks (silent steps 11, 15,
19, 27, 31, 35), with a lighter bar at steps 19-23 and a pickup at 23.5, and drops out again on step 36, so the loop
falls back into the build. The reference TikTok (media/references/flash_remedy_ref.mp4) uses the stretch from 0:46;
this edit keeps its look (angled slabs of light, collage panels, a title) without its dialogue, with new shots.
Kit XIII (all new): prism light-slab wipe, a triptych collage that slams in on the 808 stutter and bursts open on the
drop, chroma snap on the silent steps, light slabs as glue, THE FLASH title as a window into the next shot, and the
STAR Labs cortex monitor wall as the loop.
Look: warm cinematic grade with red and gold kept, letterbox over the build that opens on the drop, bumps on the big
kicks, motion blur in every transition. Plain 720p extract (no AI upscaling).
Build: bash ../../edit/build.sh .
"""
W, H, FPS = 1080, 1920, 60
PK = '../../media/scenepacks/'
S1, S2B, S3, S5, ALL, BLZ = (PK + n + '.mp4' for n in ('flash_s1', 'flash_s2_b', 'flash_s3', 'flash_s5', 'flash_all_seasons', 'flash_blitzkrieg'))
SCENEPACK = ALL
SONG = '../../media/remedy.mp3'
SONG_OFFSET = 61.47
GRID0, HALF = 0.0, 0.48
FIRST_CUT = 4
TAIL = 19.2 - (39 * 0.48 + 0.077)          # exactly 40 steps
PRE, POST = 0.60, 0.50
KICKS = [7, 8, 12, 16, 24, 28, 32]
SNARES = [10, 14, 18, 26, 30, 34]
MBLUR = True
BUMP, BUMP_KICKS, BUMP_DECAY = 0.03, [8, 12, 16, 24, 28, 32], 0.12
LETTERBOX = dict(h=0.09, open=[(3.70, 3.92)], close=[(18.70, 19.20)])

WARM = dict(rk=1.0, yk=1.0, con=1.12, tint=(.97, 1.0, 1.05))   # warm cinematic; Flash red and gold kept
def g(sat, gain, **kw): return dict(WARM, sat=sat, gain=gain, **kw)

SHOTS = [
 dict(name='emblem_walk', out='prism', fx=['bump'], pack=ALL, src=48.95, x=0.50, push=(1.00, 1.06), **g(0.55, 1.00)),   # FA14 walks out, emblem lit blue
 dict(name='mask_blue', beats=4, out='triptych', fx=['bump'], pack=ALL, src=131.75, x=0.50, push=(1.00, 1.06), **g(0.60, 1.10)),   # FA66 cowl, blue bokeh
 dict(name='streaks', beats=2, out='cut', fx=['bump'], pack=S5, src=303.93, x=0.55, push=(1.00, 1.05), **g(0.60, 1.00)),   # F5161 lightning streaks past him
 dict(name='run_gold', beats=2, out='chromasnap', fx=['bump'], pack=ALL, src=4.80, x=0.50, push=(1.00, 1.05), **g(0.60, 1.00)),   # FA1 crouch, gold lightning
 dict(name='grin', beats=2, out='slab', out_opts={'col': 'gold'}, fx=['bump'], pack=ALL, src=56.60, x=0.50, push=(1.00, 1.05), **g(0.55, 1.20)),   # FA19 masked grin
 dict(name='vibrate', beats=2, out='cut', fx=['bump'], pack=S5, src=280.78, x=0.50, push=(1.00, 1.05), **g(0.60, 1.05)),   # F5151 vibrating, lightning
 dict(name='landing', beats=2, out='whip', out_opts={'d': 1}, punch={'z': 1.18, 'flash': 0.12}, fx=['bump'], pack=S2B, src=198.60, x=0.50, push=(1.00, 1.05), **g(0.55, 1.05)),   # F93 crouch up into his eyes
 dict(name='dark_bolts', beats=2, out='cut', fx=['bump'], pack=BLZ, src=20.20, x=0.50, push=(1.00, 1.05), **g(0.60, 1.25)),   # FB64 lightning in a dark alley
 dict(name='golden', beats=4, out='title', fx=['bump'], pack=S1, src=122.80, x=0.50, push=(1.00, 1.06), **g(0.55, 1.00)),   # F135 golden hour turn
 dict(name='lunge', beats=2, out='slab', out_opts={'col': 'cyan'}, fx=['bump'], pack=BLZ, src=35.60, x=0.50, push=(1.00, 1.05), **g(0.60, 1.20)),   # FB5 lightning lunge
 dict(name='mask_xcu', beats=2, out='cut', fx=['bump'], pack=ALL, src=54.90, x=0.50, push=(1.00, 1.05), **g(0.55, 1.05)),   # FA18 cowl extreme close-up
 dict(name='debris', beats=2, out='prism', fx=['bump'], pack=S1, src=279.00, x=0.50, push=(1.00, 1.05), **g(0.55, 1.00)),   # F100 debris storm
 dict(name='suit_smile', beats=2, out='chromasnap', fx=['bump'], pack=ALL, src=93.00, x=0.50, push=(1.00, 1.05), **g(0.55, 1.00)),   # FA49 suit, smile
 dict(name='mask_bright', beats=2, out='whip', out_opts={'d': -1, 'vertical': True}, fx=['bump'], pack=ALL, src=85.60, x=0.50, push=(1.00, 1.05), **g(0.55, 1.00)),   # FA46 cowl, daylight
 dict(name='night_suit', beats=2, out='cut', fx=['bump'], pack=S5, src=73.50, x=0.50, push=(1.00, 1.05), **g(0.60, 1.10)),   # F533 night street
 dict(name='lab_suit', beats=3, out='cortex', fx=['bump'], pack=BLZ, src=167.60, x=0.50, push=(1.00, 1.05), **g(0.55, 1.15)),   # FB88 suited up in STAR Labs
]
for s in SHOTS: s.setdefault('z', 1.0); s.setdefault('frame', 'auto')
TEXT = []

from timing import build_cuts
CUTS, DUR = build_cuts(SHOTS, GRID0, HALF, FIRST_CUT, TAIL)
for i, s in enumerate(SHOTS):
    s['start'], s['end'] = CUTS[i], CUTS[i + 1]
