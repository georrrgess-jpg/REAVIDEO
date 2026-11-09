"""Dexter x "Hold On, We're Going Home" (Drake ft. Majid Jordan): final chorus, 3:12.49-3:31.69.

8 bars at 100 BPM = 64 half-beats = 19.2 s, so the video loops on the chorus downbeat: the closing
snare roll carries the blood-drop iris back into shot 0. Every cut is on a beat; the heavy
transitions sit on the loudest measured hits (snare at 3.0 s, 7.8 s, 11.4 s, 13.8 s, kick at 15.6 s).
Arc: sunlit public Dexter -> lab and blood -> the "GO DEXTER" parade colliding with the kill rooms
-> red light -> water -> loop. Footage: Dexter S2 (a, b), S4 and the mixed-season pack only.
Build: bash ../../edit/build.sh .  (or --draft). See edit/config.py for what each field means.
"""
W, H, FPS = 1080, 1920, 60
PK = '../../media/scenepacks/'
SCENEPACK = PK + 'dexter_mix_1440p.mp4'      # default pack; other shots set pack=
S2A, S2B, S4 = PK + 'dexter_s2_a.mp4', PK + 'dexter_s2_b.mp4', PK + 'dexter_s4.mp4'
SONG = '../../media/hold_on.mp3'
SONG_OFFSET = 192.487                        # final-chorus downbeat (beatgrid + bar analysis)
GRID0, HALF = 0.0, 0.3                       # 100.00 BPM, the section starts on a grid point
FIRST_CUT = 4
TAIL = 19.2 - (63 * 0.3 + 0.077)             # makes DUR exactly 64 half-beats so the audio loops on the downbeat
PRE, POST = 0.60, 0.50

MUTED = dict(con=1.12, tint=(1.04, 1.0, .96))
SHOTS = [
 # -- bar 1-2: sunlit public Dexter, two organic wipes, Dexter's knife thrusts in and the camera dives into the blade on the loudest snare (3.0 s)
 dict(name='sky_face',   out='leaf', out_opts={'ang': -38, 'seed': 3},  src=46.30, x=.50, sat=.55, rk=.9, gain=.95, push=(1.00, 1.07), **MUTED),
 dict(name='shh_jungle', beats=2, out='leaf', out_opts={'ang': 162, 'seed': 9}, pack=S2B, src=346.78, x=.47, z=1.35, cy=.40, sat=.50, rk=.8, gain=1.05, push=(1.00, 1.06), **MUTED),
 dict(name='look_up',    beats=4, out='knife', src=53.30, x=.44, sat=.50, rk=.9, gain=1.15, push=(1.00, 1.07), **MUTED),
 dict(name='car_window', beats=2, out='blur', src=276.45, x=.47, sat=.40, rk=.8, gain=1.15, push=(1.06, 1.00), **MUTED),
 # -- bar 3-4: darker, effects build: smoke, the blood room, dive into the slide's blood under a microscope, lab, blood pour
 dict(name='intense',    beats=4, out='smoke', src=57.70, x=.59, sat=.45, rk=1.0, gain=1.40, push=(1.00, 1.07), con=1.15, tint=(1.03, 1.0, .97)),
 dict(name='look_up_room', beats=2, out='dip', src=123.45, x=.61, sat=.40, rk=1.0, gain=1.30, push=(1.00, 1.07), **MUTED),
 dict(name='slide_lamp', beats=3, out='scope', out_opts={'px': 0.86, 'py': 0.64}, pack=S2B, src=268.75, x=.57, sat=.55, rk=1.0, gain=1.25, push=(1.00, 1.06), con=1.15, tint=(1.0, 1.0, 1.0)),
 dict(name='lab_photos', beats=3, out='blood', pack=S2B, src=171.70, x=.48, sat=.45, rk=1.0, gain=.95, push=(1.00, 1.06), **MUTED),
 dict(name='goggles',    beats=2, out='cut', src=110.25, x=.65, sat=.40, rk=1.0, gain=.95, push=(1.00, 1.05), **MUTED),
 # -- bar 5-6: punch on the 7.8 s snare, spin into the upside-down shot, camera flash -> Polaroid
 dict(name='gun_aim',     beats=4, out='spin', punch={}, pack=S2B, src=375.62, x=.31, sat=.45, rk=1.0, gain=1.0, push=(1.00, 1.07), **MUTED),
 dict(name='upside',     beats=2, out='whip', out_opts={'d': 1}, pack=S4, src=125.30, x=.54, z=1.15, cy=.50, sat=.40, rk=.8, gain=1.0, push=(1.00, 1.06), **MUTED),
 dict(name='camera',     beats=2, out='polaroid', pack=S4, src=121.40, x=.49, z=1.05, sat=.40, rk=.8, gain=.95, push=(1.00, 1.05), **MUTED),
 dict(name='parade_up',  beats=4, out='drops', src=343.15, x=.38, sat=.50, rk=1.0, gain=.95, push=(1.00, 1.06), **MUTED),
 dict(name='blood_wall', beats=2, out='whip', out_opts={'d': -1}, src=112.10, x=.63, sat=.40, rk=1.0, gain=.95, push=(1.00, 1.06), **MUTED),
 # -- bar 6-7: confetti curtain, the parade vs the dark, punch on the 13.8 s hit, red flash on the 15.6 s kick
 dict(name='confetti_smile', beats=2, out='confetti', src=360.05, x=.36, sat=.55, rk=1.0, gain=.95, push=(1.00, 1.06), **MUTED),
 dict(name='red_stare',  beats=4, out='cut', pack=S2B, src=324.00, x=.35, sat=.50, rk=1.0, gain=1.45, push=(1.00, 1.08), con=1.15, tint=(1.0, 1.0, 1.0)),
 dict(name='go_dexter',  beats=2, out='whip', out_opts={'dur': 0.14, 'vertical': True, 'amount': 0.5}, punch={}, src=341.95, x=.47, sat=.55, rk=1.0, gain=.95, push=(1.00, 1.05), **MUTED),
 dict(name='green_profile', beats=2, out='cut', src=231.60, x=.40, sat=.55, rk=.8, gain=1.35, push=(1.00, 1.05), **MUTED),
 dict(name='smile',      beats=2, out='red', src=38.15, x=.55, sat=.45, rk=1.0, gain=.95, push=(1.00, 1.06), **MUTED),
 # -- bar 8: the real red light comes on as the red flash fades; kill-room plastic drags across; sunset hold; loop
 dict(name='red_light',  beats=4, out='cut', src=120.62, x=.61, sat=.75, rk=1.0, gain=1.05, push=(1.00, 1.07), con=1.12, tint=(1.0, 1.0, 1.0)),
 dict(name='blue_room',  beats=2, out='plastic', pack=S2A, src=30.25, x=.60, sat=.60, rk=.8, gain=1.20, push=(1.00, 1.06), con=1.12, tint=(1.02, 1.0, .98)),
 dict(name='sunset_yard', beats=5, out='iris', fx=['leak_bubbles'], src=250.45, x=.31, sat=.40, rk=.9, gain=1.05, push=(1.00, 1.08), **MUTED),
]
for s in SHOTS: s.setdefault('z', 1.0)

from timing import build_cuts
CUTS, DUR = build_cuts(SHOTS, GRID0, HALF, FIRST_CUT, TAIL)
for i, s in enumerate(SHOTS):
    s['start'], s['end'] = CUTS[i], CUTS[i + 1]
