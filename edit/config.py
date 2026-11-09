# Timeline: cut centres sit on the 95.4 BPM half-beat grid measured from the reference
# (song offset 65.643 s, grid g(k) = 0.311 + 0.31446 k).
W, H, FPS = 1080, 1920, 60
DUR = 22.65
SONG_OFFSET = 65.643
PRE = 0.60   # source handle before slot start (s)
POST = 0.50  # source handle after slot end (s)
CUTS = [0.0, 0.940, 1.883, 2.827, 3.900, 5.028, 5.657, 6.950, 7.820, 8.801, 10.059, 11.317,
        12.575, 13.204, 13.833, 15.090, 16.348, 17.291, 17.920, 18.549, 20.250, 21.379, 22.40]
# src: in-point in the scene pack (s) at slot start; x: crop-centre hint (fraction of width);
# z: crop tightness; cy: crop centre y (fraction) when z>1; grade: sat, redkeep, contrast, tint(b,g,r), gain
SHOTS = [
 dict(name='white_bg',   src=377.12, x=.25, z=1.0, sat=.85, rk=.9, con=1.08, tint=(.96,1.0,1.05), gain=1.0, push=(1.00,1.07)),
 dict(name='palm_walk',  src=213.05, x=.66, z=1.0, sat=.75, rk=.8, con=1.08, tint=(1.0,1.0,1.0), gain=.98, push=(1.00,1.06)),
 dict(name='car_walk',   src=58.20, x=.57, z=1.0, sat=.60, rk=.7, con=1.10, tint=(1.0,1.0,.98), gain=.95, push=(1.06,1.00)),
 dict(name='hood',       src=303.75, x=.50, z=1.0, sat=.50, rk=.6, con=1.10, tint=(1.03,1.0,.97), gain=1.25, push=(1.00,1.07)),
 dict(name='flame',      src=223.35, x=.56, z=1.0, sat=.55, rk=1.0, con=1.15, tint=(1.02,1.0,.98), gain=.95, push=(1.00,1.06)),
 dict(name='eye',        src=122.25, x=.60, z=1.0, sat=.30, rk=.5, con=1.2,  tint=(1.02,1.0,1.0), gain=.80, push=(1.10,1.16)),
 dict(name='slide',      src=47.00,  x=.46, z=1.0, sat=.35, rk=1.0, con=1.10, tint=(1.05,1.0,.95), gain=1.45, push=(1.00,1.06)),
 dict(name='red_bg',     src=272.45, x=.75, track=False, z=1.0, sat=.95, rk=1.0, con=1.10, tint=(1.0,1.0,1.0), gain=1.40, push=(1.00,1.07)),
 dict(name='blood_face', src=346.30, x=.47, z=1.0, sat=.45, rk=1.0, con=1.15, tint=(1.0,1.0,1.0), gain=.95, push=(1.08,1.00)),
 dict(name='kill_room',  src=131.55, x=.59, z=1.45, cy=.42, sat=.20, rk=.3, con=1.15, tint=(1.05,1.0,.95), gain=.95, push=(1.00,1.07)),
 dict(name='camera',     src=127.00, x=.47, z=1.0, sat=.45, rk=.6, con=1.10, tint=(1.0,1.0,1.0), gain=.95, push=(1.00,1.06)),
 dict(name='coffee',     src=333.50, x=.49, z=1.0, sat=.30, rk=.4, con=1.12, tint=(1.05,1.0,.95), gain=.92, push=(1.00,1.06)),
 dict(name='spatter',    src=84.00,  x=.37, z=1.0, sat=.40, rk=1.0, con=1.12, tint=(1.0,1.0,1.0), gain=.95, push=(1.00,1.05)),
 dict(name='shock',      src=290.80, x=.21, z=1.0, sat=.55, rk=.9, con=1.12, tint=(1.0,1.0,1.0), gain=.92, push=(1.00,1.05)),
 dict(name='corridor',   src=350.95, x=.54, z=1.25, cy=.40, sat=.30, rk=.4, con=1.15, tint=(1.05,1.0,.95), gain=.92, push=(1.00,1.08)),
 dict(name='smirk',      src=308.60, x=.30, z=1.0, sat=.50, rk=.6, con=1.10, tint=(1.0,1.0,1.0), gain=.95, push=(1.00,1.06)),
 dict(name='red_drive',  src=317.00, x=.43, z=1.0, sat=.95, rk=1.0, con=1.15, tint=(1.0,1.0,1.0), gain=1.15, push=(1.00,1.06)),
 dict(name='back_turn',  src=358.72, x=.52, z=1.0, sat=.45, rk=1.0, con=1.15, tint=(1.0,1.0,1.0), gain=1.0, push=(1.00,1.05)),
 dict(name='blue_room',  src=256.60, x=.46, z=1.40, cy=.45, sat=.80, rk=.8, con=1.2,  tint=(1.05,1.0,.95), gain=1.25, push=(1.00,1.06)),
 dict(name='look_up',    src=380.30, x=.23, z=1.0, sat=.30, rk=.5, con=1.15, tint=(1.04,1.0,.96), gain=.80, push=(1.00,1.09)),
 dict(name='boat',       src=260.50, x=.60, z=1.30, cy=.50, valid=(0,999), sat=.15, rk=.2, con=1.12, tint=(1.06,1.0,.95), gain=.95, push=(1.00,1.06)),
 dict(name='fence',      src=367.20, x=.33, z=1.0, sat=.30, rk=.5, con=1.12, tint=(1.02,1.0,.97), gain=.80, push=(1.00,1.06)),
]
assert len(SHOTS) == len(CUTS) - 1
for i, s in enumerate(SHOTS):
    s['start'], s['end'] = CUTS[i], CUTS[i + 1]
