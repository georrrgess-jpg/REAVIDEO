"""The Flash x "SLAUGHTER HOUSE" (Phonkha x zecki): first drop + riser, 0:34.308-0:51.025.

8 bars at 114.85 BPM = 64 half-beats = 16.72 s. The last bar is the riser (no kick), so the loop
launches straight back into the drop. Built on the standard's structure (22 slots, big moves on the
loudest measured hits, loop ending) with Flash transitions instead of Dexter's: golden speed lines,
the emblem dive, Speed-Force phasing, a lightning strike that tears the frame, glowing eyes, crackle,
and a bolt-split loop (v2: real-emblem charge, crisp lightning, subtle eyes, beat bumps, afterimages, colour hits). Arc: civilian Barry -> lightning finds him -> the suit -> speed ->
dark fights -> red hit -> blue -> warm Barry as the lightning builds -> tunnel back to the start.
Grade: muted world, red suit (rk) and gold lightning (yk) kept vivid.
Loud hits (half-beats): 14 emblem, 22 phase, 28 punch, 40 strike, 46 punch, 52 red, 54 punch.
Build: bash ../../edit/build.sh .  (or --draft). Field meanings: edit/config.py.
"""
W, H, FPS = 1080, 1920, 60
PK = '../../media/scenepacks/'
A1, B2, C2, D3, E5, F, G = (PK + n + '.mp4' for n in ('flash_s1', 'flash_s2_a', 'flash_s2_b', 'flash_s3', 'flash_s5', 'flash_all_seasons', 'flash_blitzkrieg'))
SCENEPACK = F
SONG = '../../media/slaughter_house.mp3'
SONG_OFFSET = 34.308
GRID0, HALF = 0.0, 0.26121
FIRST_CUT = 4
TAIL = 16.717 - (63 * 0.26121 + 0.077)          # exactly 64 half-beats: loops on the drop downbeat
PRE, POST = 0.60, 0.50
KICKS = [0, 2, 4, 8, 10, 14, 20, 22, 28, 30, 36, 40, 42, 46, 52, 54]      # strong kicks (half-beats): fx 'bump' zooms on these
SNARES = [3, 6, 7, 11, 13, 15, 18, 19, 21, 23, 26, 27, 29, 31, 34, 35, 38, 39, 43, 45, 48, 50, 51, 53, 55, 56, 58, 59, 60, 61, 62, 63]     # strong snares without a kick: fx 'rgbhit' splits colour on these

L = dict(sat=.45, rk=1.0, yk=1.0, con=1.12, tint=(1.03, 1.0, .97))     # the Flash look
def g(gain, **kw): return dict(L, gain=gain, **kw)

SHOTS = [
 # -- bars 1-2: civilian Barry, the lightning finds him, the suit, the emblem dive on the loudest early hit
 dict(name='barry_sky',   out='speedlines', out_opts={'d': 1}, fx=['bump'], pack=G,  src=94.05,  x=.37, push=(1.00, 1.06), **g(1.0)),
 dict(name='sparks',      beats=2, out='whip', out_opts={'d': -1}, pack=G, src=48.40, x=.45, push=(1.00, 1.06), **g(1.35)),
 dict(name='cowl_down',   beats=2, out='blur',  pack=F,  src=84.20,  x=.55, push=(1.00, 1.06), **g(1.25)),
 dict(name='grin_sky',    beats=2, out='spin',  pack=D3, src=44.55,  x=.52, push=(1.00, 1.06), **g(1.0)),
 dict(name='hero_charge', beats=4, out='emblem', fx=['bump'], pack=F, src=90.75,  x=.63, track=False, push=(1.00, 1.07), **g(1.05)),
 # -- bars 3-4: the eye, running, phasing on the 5.7 s hit, Speed Force face, speed tunnel, eyes ignite on the punch
 dict(name='eye_ecu',     beats=2, out='whip', out_opts={'d': 1}, pack=C2, src=2.30, x=.30, track=False, push=(1.04, 1.10), **g(1.4)),
 dict(name='run_sunset',  beats=4, out='cut', fx=['echo', 'bump'], pack=F,  src=4.45,   x=.72, push=(1.00, 1.06), **g(1.0)),
 dict(name='bolt_face',   beats=2, out='phase', pack=E5, src=264.85, x=.48, push=(1.00, 1.06), **g(0.95)),
 dict(name='speedforce',  beats=4, out='cut', fx=['bump', 'rgbhit'], pack=F,  src=117.20, x=.55, push=(1.00, 1.07), **g(0.95)),
 dict(name='road_streak', beats=2, out='cut', fx=['bump'], pack=G,  src=63.85,  x=.50, track=False, push=(1.00, 1.10), **g(0.9)),
 dict(name='cowl_eyes',   beats=2, out='whip', out_opts={'dur': 0.14, 'vertical': True, 'amount': 0.5}, punch={}, fx=['eyes'], pack=F, src=54.65, x=.51, push=(1.00, 1.06), **g(1.25)),
 # -- bars 5-6: speed and fights, the lightning throw tears the frame on the 10.4 s hit
 dict(name='purple_trail', beats=3, out='blur', pack=E5, src=304.00, x=.69, push=(1.00, 1.06), **g(1.0)),
 dict(name='fly_trail',   beats=3, out='cut', fx=['echo', 'rgbhit'], pack=G,  src=75.70,  x=.45, track=False, push=(1.00, 1.08), **g(1.3)),
 dict(name='time_ring',   beats=2, out='blur',  pack=E5, src=67.00,  x=.42, push=(1.00, 1.06), **g(1.4)),
 dict(name='throw',       beats=2, out='strike', pack=F, src=124.20, x=.64, push=(1.00, 1.05), **g(1.1)),
 dict(name='night_shout', beats=2, out='cut', fx=['rgbhit'], pack=E5, src=13.20,  x=.50, push=(1.00, 1.06), **g(1.35)),
 # -- bars 6-7: punch on the 12.0 s hit, red flash on the loudest kick (13.6 s), punch on 14.1 s
 dict(name='suit_sparks', beats=4, out='cut',   pack=F,  src=133.85, x=.50, push=(1.00, 1.07), fx=['crackle', 'bump'], **g(1.35)),
 dict(name='grin_dark',   beats=2, out='whip', out_opts={'d': -1}, punch={}, pack=F, src=56.55, x=.44, push=(1.00, 1.05), **g(1.4)),
 dict(name='laugh',       beats=4, out='red', fx=['bump'], pack=E5, src=241.60, x=.60, push=(1.00, 1.06), **g(1.35)),
 dict(name='earpiece',    beats=2, out='cut', fx=['rgbhit'], pack=F,  src=69.60,  x=.50, push=(1.00, 1.05), **g(1.0)),
 # -- bar 8 (the riser): cold blue, then warm Barry while the lightning builds, Speed Force tunnel loop
 dict(name='speed_cap',   beats=4, out='dip', fx=['crackle_blue', 'bump'], punch={'dur': 0.18, 'z': 1.14, 'flash': 0.10}, pack=G, src=55.00, x=.54, push=(1.00, 1.06), **g(1.5, tint=(1.06, 1.0, .95))),
 dict(name='warm_hold',   beats=5, out='boltsplit', fx=['ripple'], pack=E5, src=221.50, x=.31, push=(1.00, 1.08), **g(1.6)),
]
for s in SHOTS: s.setdefault('z', 1.0)

from timing import build_cuts
CUTS, DUR = build_cuts(SHOTS, GRID0, HALF, FIRST_CUT, TAIL)
for i, s in enumerate(SHOTS):
    s['start'], s['end'] = CUTS[i], CUTS[i + 1]
