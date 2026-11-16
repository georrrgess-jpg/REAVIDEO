"""Timeline: which shot / transition is on screen at time t. frame(t) returns a float32 BGR image."""
import math
import numpy as np, cv2
from config import *
from fx import *

C = CUTS
_lazy = {}


def lazy(key, fn):
    if key not in _lazy: _lazy[key] = fn()
    return _lazy[key]


# ------------------------------------------------------------ per-shot overlays
def streaks(f, t, t0=3.25, dur=0.65):
    u = (t - t0) / dur
    if u <= 0: return f
    amp = ease(u * 1.6)
    p = proj(32)
    acc = np.zeros((H, W), np.float32)
    for p0, v, sig, am in [(2500, -2500, 26, .50), (2950, -2300, 80, .26), (2250, -2900, 10, .55), (3300, -2600, 46, .30)]:
        acc += am * np.exp(-((p - (p0 + v * u)) / sig) ** 2)
    return f + (acc * amp)[..., None] * np.float32([1.0, 0.97, 0.92])


SLIDE_T0 = 0.0   # set per call from the shot start


def slide_pose(which, t, t0=None):
    u = t - (SLIDE_T0 if t0 is None else t0)
    if which == 0: return (0.40 * W + 20 * u, 0.33 * H - 16 * u, math.radians(-5 + 3.5 * u), 0.50)
    return (0.60 * W - 14 * u, 0.71 * H + 22 * u, math.radians(6 - 3 * u), 0.64)


def draw_slide(f, which, t, alpha=1.0, pose=None, t0=0.0):
    spr, body = lazy('slide', make_slide)
    cx, cy, ang, sc = pose or slide_pose(which, t, t0)
    ent = eout((t - t0 - 0.05 * which) / 0.28)
    if pose is None: sc *= 1.3 - 0.3 * ent; alpha *= ent
    if alpha <= 0.003: return f
    M = affine((spr.shape[1] / 2, spr.shape[0] / 2), (cx, cy), ang, sc)
    layer_ = place(spr, M) * alpha
    if which == 1: layer_ = cv2.GaussianBlur(layer_, (0, 0), 2.0)
    return over(f, layer_)


def light_leak(f, t):
    g = np.exp(-(((XX + 0.02 * W) / 150) ** 2 + ((YY - 0.30 * H) / 260) ** 2))
    return f + g[..., None] * np.float32([0.10, 0.04, 0.70]) * (0.30 + 0.10 * math.sin(7 * t))


def bubbles(f, t, alpha, mask=None, seed=21, n=70, t0=19.8, small=False):
    if alpha <= 0.01: return f
    r = np.random.default_rng(seed)
    can = np.zeros((H, W), np.uint8)
    for j in range(n):
        x = r.uniform(0, W); y0 = r.uniform(0, H + 200); rad = r.uniform(3, 8) if small else r.uniform(4, 17)
        vy = r.uniform(120, 300) if small else r.uniform(350, 900); ph = r.uniform(0, 6.28)
        y = (y0 - vy * (t - t0)) % (H + 200) - 100
        xx = x + 14 * math.sin(ph + 5 * t)
        cv2.circle(can, (int(xx), int(y)), int(rad), 255, 2 if rad > 6 else 1, cv2.LINE_AA)
        if rad > 7: cv2.circle(can, (int(xx - rad * .35), int(y - rad * .35)), max(1, int(rad * .18)), 255, -1, cv2.LINE_AA)
    c = can.astype(np.float32) / 255 * alpha * 0.6
    if mask is not None: c = c * mask
    return f + c[..., None] * np.float32([1.0, 1.0, 0.95])


def fringe(f):
    f = f.copy(); f[..., 2] = np.roll(f[..., 2], 9, axis=1); f[..., 0] = np.roll(f[..., 0], -5, axis=1)
    f[..., 2] += 0.10 * ss(0.55 * W, W, XX)
    return f


def layer(i, t, ez=1.0):
    s = SHOTS[i]
    if 'bump' in s.get('fx', ()): ez *= 1 + bump_amount(t, s)
    f = shot(i, t, ez)
    for name in s.get('fx', ()):
        if name == 'streaks': f = streaks(f, t, t0=s['end'] - 0.65)          # glass light streaks, end on the wipe
        elif name == 'fringe': f = fringe(f)                                 # macro: red/cyan fringing
        elif name == 'slides':                                               # floating glass blood slides
            w0 = s['end'] - 0.45
            f = draw_slide(f, 1, t, alpha=1 - ease((t - w0) / 0.3), t0=s['start'])
            if t < w0 or s.get('out') != 'slidewipe': f = draw_slide(f, 0, t, t0=s['start'])
        elif name == 'leak_bubbles':                                         # red light leak + drifting particles
            f = light_leak(f, t)
            f = bubbles(f, t, ease((t - (s['end'] - 0.50)) / 0.35), small=True, n=55, t0=s['end'] - 0.45)
        elif name == 'eyes': f = eyes_fx(f, i, t, s)
        elif name == 'crackle': f = crackle_fx(f, i, t, s, amp_scale=0.7)
        elif name == 'crackle_blue': f = crackle_fx(f, i, t, s, glow=BLUE_GLOW, region=(0.15, 0.02, 0.85, 0.30), amp_scale=0.9)
        elif name == 'bump': pass
        elif name == 'rgbhit': f = rgbhit_fx(f, t, s)
        elif name == 'echo': f = echo_fx(f, i, t, s)
        elif name == 'sweep': f = sweep_fx(f, t, s)
        elif name == 'ripple': f = ripple_fx(f, t, s)
        elif name == 'dust': f = dust_fx(f, i, t, s)
        elif name == 'markers': f = markers_fx(f, i, t, s)
        elif name == 'idbox': f = idbox_fx(f, i, t, s)
        elif name.startswith('stamp:'): f = stamp_fx(f, t, s, name[6:])
        elif name == 'glint': f = glint_fx(f, t, s)
        elif name == 'ghost': f = ghost_fx(f, i, t, s)
        elif name == 'drip': f = drip_fx(f, i, t, s)
        elif name == 'passenger': f = ghost_fx(f, i, t, s, col=(0.20, 0.16, 0.95))   # the Dark Passenger: a red double
        elif name == 'trails': f = trails_fx(f, i, t, s)
        elif name == 'crackle_build': f = crackle_fx(f, i, t, s, build=True)
        elif name == 'pulse': f = pulse_fx(f, t, s)
        elif name == 'slabhit': f = slabhit_fx(f, t, s)
        elif name == 'gapsplit': f = gapsplit_fx(f, t, s)
        elif name == 'emblempulse': f = emblempulse_fx(f, i, t, s)
        elif name == 'ekg': f = ekg_overlay(f, t, s, flat_from=(s['end'] - 0.30) if s.get('out') == 'flatline' else None)
        else: raise ValueError('unknown fx %r on shot %d' % (name, i))
    if s.get('projected') and not _K4.get('raw'): f = proj_look(f, t, i)
    return f


# ------------------------------------------------------------ transitions
def t_leaf(t, c, A, B, ang, seed, dur=0.30):
    a, b = c - dur / 2, c + dur / 2
    th = math.radians(ang); dx, dy = math.cos(th), math.sin(th)

    def build():
        N = 3600; can = np.zeros((N, N, 4), np.float32)
        for off, extra, sc, sd, flip in [(-300, -15, 1.10, seed + 5, True), (0, 7, 1.30, seed, False)]:
            fr = make_frond(seed=sd)
            if flip: fr = fr[::-1].copy()
            M = affine((fr.shape[1] / 2, fr.shape[0] / 2), (N / 2 + dx * off, N / 2 + dy * off), th + math.radians(extra), sc)
            lay = cv2.warpAffine(fr, M, (N, N), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            can = can * (1 - lay[..., 3:4]) + lay
        return can
    can = lazy(('leaf', seed, ang), build); N = can.shape[0]
    S = 0.5 * (W * abs(dx) + H * abs(dy)) + 950
    pos = lambda tt: lerp(-S, S, 0.5 * (clamp((tt - a) / (b - a)) + ease((tt - a) / (b - a))))
    s = pos(t)
    pr = (XX - W / 2) * dx + (YY - H / 2) * dy
    m = ss(-90, 90, (s - 150) - pr)[..., None]                    # B behind the frond
    f = layer(A, t) * (1 - m) + layer(B, t) * m if 0.001 < m.mean() < 0.999 else (layer(B, t) if m.mean() >= 0.999 else layer(A, t))
    acc = np.zeros((H, W, 4), np.float32); K = 12
    for k in range(K):                                           # motion blur over the shutter
        sk = pos(t - (k / K) * 0.6 / FPS)
        M = np.float32([[1, 0, W / 2 + dx * sk - N / 2], [0, 1, H / 2 + dy * sk - N / 2]])
        acc += cv2.warpAffine(can, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return over(f, acc / K)


def t_blur(t, c, A, B, dur=0.30, r=55):
    a, b = c - dur / 2, c + dur / 2; p = clamp((t - a) / (b - a))
    k = math.sin(math.pi * p) ** 0.8
    fa = blur(layer(A, t, 1 + 0.14 * p), r * k) if p < 0.85 else None
    fb = blur(layer(B, t, 1 + 0.14 * (1 - p)), r * k) if p > 0.15 else None
    if fa is None: return fb
    if fb is None: return fa
    m = ease((p - 0.15) / 0.7)
    return fa * (1 - m) + fb * m


def t_glass(t, a, b, A, B, ang=32, rev=False, bright=1.0):
    p = ease((t - a) / (b - a)); pr = proj(ang); lo, hi = float(pr.min()) - 80, float(pr.max()) + 80
    s = lerp(lo, hi, p) if rev else lerp(hi, lo, p)
    m = ss(-3, 3, (s - pr) if rev else (pr - s))[..., None]
    f = layer(A, t) * (1 - m) + layer(B, t) * m
    d = np.abs(pr - s)
    return f + ((np.exp(-(d / 5) ** 2) * 0.95 + np.exp(-(d / 50) ** 2) * 0.28) * bright)[..., None]


def t_smoke(t, c, A, B, a, b):
    q = 4; hq, wq = H // q, W // q
    NS = lazy('smoke_n', lambda: tile_fbm(hq * 3, wq, octaves=5, seed=11, base=3))
    NS2 = lazy('smoke_n2', lambda: tile_fbm(hq * 3, wq, octaves=4, seed=12, base=6))
    f = layer(A, t) if t < c else layer(B, t)
    n = np.roll(NS, -int((t - a) * 1100 / q), axis=0)[:hq]
    n2 = np.roll(NS2, -int((t - a) * 1500 / q), axis=0)[:hq]
    yb = 1 - (np.arange(hq, dtype=np.float32) / hq)[:, None]
    if t < c:
        rise = lerp(-0.30, 1.45, clamp((t - a) / (c - a)) ** 1.3)
        v = rise - yb + 0.60 * (n - 0.5)
        alpha = ss(0.0, 0.20, v); glow = ss(0.0, 0.08, v) * (1 - ss(0.08, 0.34, v))
    else:
        u = (t - c) / (b - c)
        alpha = ss(0.0, 0.30, n * 1.2 + 0.25 - u * 1.55); glow = alpha * (1 - alpha) * 2.2 * (1 - u)
    col = np.float32([0.045, 0.06, 0.10]) + n2[..., None] * np.float32([0.07, 0.10, 0.17])
    col = col + glow[..., None] * np.float32([0.10, 0.48, 1.05]) * (0.7 + 0.8 * n2[..., None])
    alpha = cv2.resize(alpha, (W, H), interpolation=cv2.INTER_LINEAR)[..., None]
    col = cv2.resize(col, (W, H), interpolation=cv2.INTER_LINEAR)
    f = f * (1 - alpha) + col * alpha
    # embers
    r = np.random.default_rng(31); can = np.zeros((H, W), np.uint8)
    for j in range(46):
        x = r.uniform(0, W); y0 = r.uniform(0.2, 1.5) * H; vy = r.uniform(1800, 4200); ph = r.uniform(0, 6.28)
        y = y0 + 0.6 * H - vy * (t - a)
        cv2.circle(can, (int(x + 30 * math.sin(ph + 9 * t)), int(y)), int(r.uniform(2, 5)), 255, -1, cv2.LINE_AA)
    e = cv2.GaussianBlur(can.astype(np.float32) / 255, (0, 0), 3) * 2.2 + cv2.GaussianBlur(can.astype(np.float32) / 255, (0, 0), 12) * 2.0
    env = math.sin(math.pi * clamp((t - a) / (b - a)))
    return f + (e * env)[..., None] * np.float32([0.15, 0.55, 1.0])


def t_dip(t, c, A, B, a, b):
    if t < c: return layer(A, t) * (1 - 0.94 * ein((t - a) / (c - a)))
    u = (t - c) / (b - c)
    return blur(layer(B, t), 16 * (1 - u)) * (0.06 + 0.94 * eout(u))


def t_slidewipe(t, A, B, a, b):
    p = clamp((t - a) / (b - a))
    spr, body = lazy('slide', make_slide)
    f = layer(A, t)
    cx0, cy0, a0, s0 = slide_pose(0, t, SHOTS[A]['start'])
    e = ease(p)
    cx, cy = lerp(cx0, W / 2, e), lerp(cy0, H / 2, e)
    ang = lerp(a0, math.radians(-90), ease(p * 1.05))
    sc = s0 * math.exp(math.log(3.75 / s0) * p ** 1.9)
    M = affine((spr.shape[1] / 2, spr.shape[0] / 2), (cx, cy), ang, sc)
    matte = place(body, M)[..., None] * ease((p - 0.08) / 0.38)
    f = f * (1 - matte) + layer(B, t) * matte
    return over(f, place(spr, M) * (1 - ease((p - 0.55) / 0.45)))


def t_blood(t, A, B, a, sw, b):
    f = layer(A, t) if t < sw else layer(B, t)
    BN = lazy('blood_n', lambda: cv2.resize(tile_fbm(700, 540, octaves=5, seed=41, base=6), (W, 2 * H), interpolation=cv2.INTER_CUBIC))
    BN2 = lazy('blood_n2', lambda: cv2.resize(tile_fbm(500, 540, octaves=5, seed=42, base=22), (W, 2 * H), interpolation=cv2.INTER_CUBIC))
    x = np.arange(W, dtype=np.float32)
    r = np.random.default_rng(8)
    drips = sum(r.uniform(60, 190) * np.exp(-((x - r.uniform(0, W)) / r.uniform(28, 60)) ** 2) for _ in range(9))
    drips2 = sum(r.uniform(50, 150) * np.exp(-((x - r.uniform(0, W)) / r.uniform(30, 70)) ** 2) for _ in range(8))
    pb = clamp((t - a) / (sw - a - 0.03)); pt = clamp((t - (sw + 0.03)) / (b - sw - 0.03))
    yb = H * (-0.16 + 1.40 * (0.35 * pb + 0.65 * ease(pb))) + 40 * np.sin(x / 62 + 9 * t) + 22 * np.sin(x / 27 - 14 * t + 1) + drips * math.sin(math.pi * clamp(pb * 0.95 + 0.05))
    yt = H * (-0.22 + 1.42 * (0.35 * pt + 0.65 * ease(pt))) + 32 * np.sin(x / 74 - 7 * t + 2) + 18 * np.sin(x / 31 + 11 * t) + drips2 * math.sin(math.pi * clamp(pt)) * 1.2
    db = yb[None, :] - YY; dt_ = YY - yt[None, :]
    inside = (ss(-2, 2, db) * ss(-2, 2, dt_))
    if inside.max() <= 0: return f
    n = np.roll(BN, int((t - a) * 2300), axis=0)[:H]
    n2 = np.roll(BN2, int((t - a) * 3100), axis=0)[:H]
    col = np.stack([0.012 + 0.04 * n, 0.0 + 0.015 * n, 0.20 + 0.42 * n], -1)
    spec = ss(0.70, 0.86, n2) * 0.38
    col += spec[..., None] * np.float32([0.70, 0.72, 1.0])
    rim = np.exp(-(np.clip(db, 0, None) / 16) ** 2) * 0.55 + np.exp(-(np.clip(dt_, 0, None) / 9) ** 2) * 0.25
    col += rim[..., None] * np.float32([0.25, 0.25, 0.75])
    col *= (1 - 0.45 * np.exp(-(np.clip(dt_, 0, None) / 60) ** 2))[..., None]        # darker just under the top edge
    sh = np.exp(-(np.clip(-db, 0, None) / 34) ** 2) * (db < 0)                      # shadow cast below the front
    f = f * (1 - 0.5 * sh[..., None])
    m = inside[..., None]
    return f * (1 - m) + col * m


def t_spin(t, c, A, B, dur=0.34, maxang=105):
    a, b = c - dur / 2, c + dur / 2

    def pose(tt):
        if tt < c: u = clamp((tt - a) / (c - a)); return A, maxang * ein(u), 1 + 0.55 * u ** 2
        u = clamp((tt - c) / (b - c)); return B, -maxang * (1 - eout(u)), 1 + 0.55 * (1 - u) ** 2
    i, ang, z = pose(t); _, ang_p, _ = pose(t - 1.0 / FPS)
    if pose(t - 1.0 / FPS)[0] != i: ang_p = ang - 14
    src = layer(i, t); K = 6; acc = np.zeros_like(src)
    for k in range(K): acc += zoom(src, z, ang=-lerp(ang, ang_p, k / K * 0.9), cy=H / 2)
    return acc / K


def t_polaroid(t, c, A, B):
    if t < c:
        u = (t - (c - 0.10)) / 0.10
        return layer(A, t) + 0.95 * ein(u)

    def build():
        F = layer(A, c - 1.0 / FPS)
        ph = int(W * 1420 / 1280); y0 = int(0.42 * H - ph / 2)
        pic = cv2.resize(F[y0:y0 + ph], (1280, 1420), interpolation=cv2.INTER_CUBIC)
        card = np.ones((1800, 1440, 3), np.float32) * np.float32([0.90, 0.93, 0.95])
        card *= (0.97 + 0.05 * fbm(1800, 1440, 4, 61, 10))[..., None]
        card[80:1500, 80:1360] = pic
        cv2.rectangle(card, (80, 80), (1360, 1500), (0.55, 0.57, 0.58), 3)
        return card
    card = lazy('polaroid', build)
    u = t - c
    g = eout(u / 0.20); fall = ein((u - 0.30) / 0.26)
    sc = lerp(W / 1280, 0.50, g) * (1 - 0.04 * clamp(u / 0.5))
    ang = math.radians(lerp(0, -8, g) - 3 * clamp(u / 0.5) + 30 * fall)
    cx = lerp(W / 2, 0.50 * W, g) + fall * 0.55 * W; cy = lerp(0.42 * H + (110 * W / 1280), 0.47 * H, g) + fall * 1.15 * H
    M = affine((720, 900), (cx, cy), ang, sc)
    f = blur(layer(B, t), 26 * (1 - eout(u / 0.3))) * (0.55 + 0.45 * eout(u / 0.3))
    one = np.ones((1800, 1440), np.float32)
    mask = cv2.warpAffine(one, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    sh = blur(np.roll(np.roll(mask, 34, axis=0), 20, axis=1)[..., None].repeat(3, 2), 26)[..., :1]
    f = f * (1 - 0.55 * sh)
    img = cv2.warpAffine(card, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    f = f * (1 - mask[..., None]) + img
    return f + 0.95 * (1 - eout(u / 0.16))        # camera flash decaying


def t_drops(t, c, A, B, a, b):
    f = layer(A, t) if t < c else layer(B, t)
    r = np.random.default_rng(77)
    can = np.zeros((H, W), np.uint8); hi = np.zeros((H, W), np.uint8)
    out = (t - c) / (b - c) if t >= c else 0.0
    for j in range(34):
        x = r.uniform(-0.05, 1.05) * W; y = H * (1.05 - 1.15 * r.uniform(0, 1) ** 1.6); R = r.uniform(70, 300)
        t0 = a + r.uniform(0, 0.60) * (c - a); g = eout((t - t0) / 0.16) * (1 + 0.25 * out)
        sats = [(r.uniform(0, 6.28), r.uniform(0.9, 1.9), r.uniform(0.10, 0.34)) for _ in range(int(r.integers(4, 9)))]
        if g <= 0: continue
        cv2.circle(can, (int(x), int(y)), int(R * g), 255, -1, cv2.LINE_AA)
        if R > 150: cv2.ellipse(hi, (int(x - R * g * .36), int(y - R * g * .40)), (max(1, int(R * g * .20)), max(1, int(R * g * .11))), -35, 0, 360, 255, -1, cv2.LINE_AA)
        for th, dd, rs in sats:
            cv2.circle(can, (int(x + math.cos(th) * dd * R * g), int(y + math.sin(th) * dd * R * g)), max(1, int(R * rs * g)), 255, -1, cv2.LINE_AA)
    if can.max() == 0: return f
    NZ = lazy('drop_n', lambda: fbm(H, W, 5, 71, 14))
    soft = blur((can.astype(np.float32) / 255)[..., None].repeat(3, 2), 9)[..., 0]
    m = ss(0.42, 0.50, soft + 0.42 * (NZ - 0.5))                       # ragged, organic edge
    depth = blur(m[..., None].repeat(3, 2), 26)[..., 0]
    col = np.float32([0.008, 0.0, 0.13])[None, None] * (1 - depth[..., None] ** 1.6) + np.float32([0.045, 0.012, 0.50]) * depth[..., None] ** 1.6
    col *= (0.80 + 0.40 * NZ)[..., None]
    col += (cv2.GaussianBlur(hi.astype(np.float32) / 255, (0, 0), 9) * 0.30 * m)[..., None] * np.float32([0.8, 0.8, 1.0])
    edge = np.clip(blur(m[..., None].repeat(3, 2), 5)[..., 0] - m, 0, 1)   # thin shadow around each blot
    f = f * (1 - 0.6 * edge[..., None])
    m = (m * 0.96 * (1 - ease(out)))[..., None]
    return f * (1 - m) + col * m


def t_whip(t, c, A, B, dur=0.20, d=1, vertical=False, amount=0.9):
    a, b = c - dur / 2, c + dur / 2; p = clamp((t - a) / (b - a))
    span = H if vertical else W
    if p < 0.5: f = layer(A, t); off = -d * span * amount * ein(p * 2)
    else: f = layer(B, t); off = d * span * amount * (1 - eout((p - 0.5) * 2))
    k = int(300 * math.sin(math.pi * p)) | 1
    f = np.roll(f, int(off), axis=0 if vertical else 1)
    return cv2.blur(f, (1, k) if vertical else (k, 1)) if k > 1 else f


def t_punch(t, c, B, dur=0.20, z=1.22, flash=0.14):
    u = (t - c) / dur
    return layer(B, t, 1 + (z - 1) * (1 - eout(u))) + flash * (1 - eout(u * 2))


def t_red(t, c, A, B, a, b):
    if t < c: u = clamp((t - a) / (c - a)); f = layer(A, t, 1 + 0.10 * ein(u)); r = ein(u)
    else: f = layer(B, t, 1 + 0.16 * (1 - eout((t - c) / 0.22))); r = 1 - eout((t - c) / (b - c))
    l = luma(f)
    red = np.stack([l * 0.16 + 0.02, l * 0.10, l * 1.25 + 0.24], -1)
    f = f * (1 - r) + red * r
    D = lazy('ringD', lambda: np.sqrt((XX - W / 2) ** 2 + (YY - 0.42 * H) ** 2))
    ring = np.zeros((H, W), np.float32)
    for k in range(8): ring += np.exp(-((D - (k * 130 + (t - a) * 950)) / 3.0) ** 2)
    return f + (ring * 0.34 * r)[..., None] * np.float32([0.70, 0.70, 1.0])


def t_water(t, A, B, a, sw, b):
    f = layer(A, t) if t < sw else layer(B, t)
    x = np.arange(W, dtype=np.float32)
    wave = 22 * np.sin(x / 66 + 10 * t) + 10 * np.sin(x / 27 - 17 * t + 1)
    if t < sw:
        lvl = H * lerp(1.06, -0.10, ease((t - a) / (sw - a))) + wave
        m = ss(-2, 2, YY - lvl[None, :]); strength = 1.0
    else:
        u = (t - sw) / (b - sw)
        lvl = H * lerp(-0.10, 1.10, ease(u * 1.15)) + wave
        m = 0.40 + 0.60 * ss(-2, 2, YY - lvl[None, :]); strength = 1 - ease(u * 1.1)
    wob = cv2.remap(f, XX + 8 * np.sin(YY / 38 + 12 * t), YY + 5 * np.sin(XX / 52 + 9 * t), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    tinted = wob * np.float32([0.62, 0.74, 0.46]) + np.float32([0.085, 0.12, 0.03])
    mm = (m * strength)[..., None]
    f = f * (1 - mm) + tinted * mm
    d = YY - lvl[None, :]
    f = f + ((np.exp(-(d / 2.6) ** 2) * 0.9 + np.exp(-(d / 16) ** 2) * 0.22) * (1.0 if t < sw else strength ** 0.5))[..., None]
    return bubbles(f, t, strength, mask=m, seed=5, n=80, t0=a)


def t_iris(t, A, B, a, fall, end):
    if t >= end: return layer(B, t - DUR)
    fa = layer(A, t); fb = layer(B, t - DUR)
    if t < fall:
        u = (t - a) / (fall - a); cx = lerp(0.80, 0.72, u) * W; cy = lerp(-0.03, 0.34, u ** 1.7) * H; R = lerp(12, 78, u ** 1.3); m = 0.0
    else:
        u = (t - fall) / (end - fall); R = 78 * math.exp(math.log(2700 / 78) * u ** 1.5)
        cx = lerp(0.72, 0.5, ease(u)) * W; cy = lerp(0.34, 0.45, ease(u)) * H; m = ease((u - 0.2) / 0.8)
    dx = XX - cx; dy = YY - cy; rho = np.sqrt(dx * dx + dy * dy) / R
    inside = np.clip((1 - rho) * R / 1.5 + 0.5, 0, 1)[..., None]
    zf = lerp(R / (0.50 * H), 1.0, m); dist = lerp(0.6 + 0.4 * np.clip(rho, 0, 1) ** 2, 1.0, m)
    bs = cv2.remap(fb, lerp(W / 2, cx, m) + dx / zf * dist, lerp(0.42 * H, cy, m) + dy / zf * dist, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    k = 1 - m
    drop = bs * (1 - 0.85 * k * (1 - np.float32([0.18, 0.14, 0.95]))) + np.float32([0.0, 0.0, 0.16]) * k
    vis = float(ss(18, 95, R))
    drop = np.float32([0.02, 0.0, 0.42]) * (1 - vis) + drop * vis
    drop = drop * (1 - 0.6 * k * np.clip(rho, 0, 1) ** 3)[..., None]
    spec = np.exp(-(((dx + 0.36 * R) / (0.24 * R)) ** 2 + ((dy + 0.42 * R) / (0.15 * R)) ** 2)) * 0.8 * k
    drop = drop + spec[..., None]
    sh = np.exp(-(((dx - 0.10 * R) / (1.15 * R)) ** 2 + ((dy - 0.14 * R) / (1.15 * R)) ** 2) * 2.2) * 0.35 * k   # contact shadow
    fa = fa * (1 - sh[..., None])
    return fa * (1 - inside) + drop * inside


# ------------------------------------------------------------ confetti curtain
_CONF = {}


def _confetti_pieces():
    if 'p' not in _CONF:
        r = np.random.default_rng(57); n = 1500
        band = np.concatenate([r.exponential(110, 1000), r.uniform(0, 1.5 * H, 500)])   # dense at the front, sparse trail behind
        cols = np.array([[0.13, 0.12, 0.80], [0.70, 0.32, 0.12], [0.93, 0.93, 0.93], [0.20, 0.55, 0.90]], np.float32)  # red, blue, white, gold
        _CONF['p'] = dict(x=r.uniform(-60, W + 60, n), off=band, w=r.uniform(16, 34, n), h=r.uniform(26, 58, n),
                          rot=r.uniform(0, 6.28, n), vr=r.uniform(-9, 9, n), flip=r.uniform(8, 22, n), ph=r.uniform(0, 6.28, n),
                          sway=r.uniform(10, 40, n), col=cols[r.choice(4, n, p=[0.36, 0.30, 0.26, 0.08])], depth=r.uniform(0, 1, n))
    return _CONF['p']


def t_confetti(t, c, A, B, a, b):
    """A curtain of falling parade confetti sweeps down the frame; the next shot is revealed behind it."""
    u = clamp((t - a) / (b - a))
    front = lerp(-0.25 * H, 1.45 * H, 0.3 * u + 0.7 * ease(u))
    x = np.arange(W, dtype=np.float32)
    edge = front - 120 + 26 * np.sin(x / 47 + 6 * t) + 18 * np.sin(x / 19 - 9 * t)          # ragged reveal line behind the densest band
    m = ss(-25, 25, edge[None, :] - YY)[..., None]
    f = layer(A, t) * (1 - m) + layer(B, t) * m if 0.001 < float(m.mean()) < 0.999 else (layer(B, t) if float(m.mean()) >= 0.999 else layer(A, t))
    P = _confetti_pieces(); tt = t - a
    y = front - P['off'] + 18 * np.sin(P['ph'] + 3 * tt)
    vis = (y > -80) & (y < H + 80)
    can = np.zeros((H, W, 3), np.float32); al = np.zeros((H, W), np.float32)
    order = np.argsort(P['depth'])
    for i in order:
        if not vis[i]: continue
        cx = P['x'][i] + P['sway'][i] * math.sin(P['ph'][i] + 4.0 * tt); cy = y[i]
        sc = 0.55 + 0.75 * P['depth'][i]
        fl = math.cos(P['ph'][i] + P['flip'][i] * tt)                       # paper flipping: width and brightness flicker
        hw, hh = P['w'][i] * sc * (0.15 + 0.85 * abs(fl)) / 2, P['h'][i] * sc / 2
        r_ = P['rot'][i] + P['vr'][i] * tt; cr, sr = math.cos(r_), math.sin(r_)
        pts = np.array([[cx + dx * cr - dy * sr, cy + dx * sr + dy * cr] for dx, dy in ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh))], np.int32)
        shade = 0.55 + 0.45 * abs(fl)
        cv2.fillConvexPoly(can, pts, tuple(float(v) for v in P['col'][i] * shade), cv2.LINE_AA)
        cv2.fillConvexPoly(al, pts, 1.0, cv2.LINE_AA)
    al = cv2.GaussianBlur(al, (0, 0), 1.0)[..., None]; can = cv2.GaussianBlur(can, (0, 0), 1.0)
    trail = 1 - ease((u - 0.75) / 0.25)                                      # trailing pieces fade as they settle
    f = f * (1 - 0.35 * cv2.GaussianBlur(al, (0, 0), 9)[..., None] * trail)  # soft contact shadow
    return f * (1 - al * trail) + can * trail


# ------------------------------------------------------------ glass shatter
def _shards(cx, cy, n=46, seed=12):
    key = ('shards', round(cx), round(cy), n, seed)
    if key in _lazy: return _lazy[key]
    from scipy.spatial import cKDTree
    r = np.random.default_rng(seed)
    ang = r.uniform(0, 2 * np.pi, n); rad = (r.uniform(0, 1, n) ** 1.6) * 1.15 * max(W, H)   # denser near the impact
    pts = np.stack([cx + rad * np.cos(ang), cy + rad * np.sin(ang)], 1)
    pts = np.vstack([pts, [[cx, cy]]])
    q = 4; gy, gx = np.mgrid[0:H // q, 0:W // q]
    _, lab = cKDTree(pts).query(np.stack([gx.ravel() * q + q / 2, gy.ravel() * q + q / 2], 1))
    lab = cv2.resize(lab.reshape(H // q, W // q).astype(np.uint16), (W, H), interpolation=cv2.INTER_NEAREST)
    edges = (cv2.morphologyEx(lab, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0).astype(np.float32)
    shards = []
    for i in range(len(pts)):
        m = (lab == i).astype(np.uint8)
        ys, xs = np.nonzero(m)
        if len(xs) < 50: continue
        x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        ccx, ccy = xs.mean(), ys.mean(); d = math.hypot(ccx - cx, ccy - cy) + 60
        dirx, diry = (ccx - cx) / d, (ccy - cy) / d
        shards.append(dict(box=(x0, y0, x1, y1), mask=cv2.GaussianBlur(m[y0:y1, x0:x1].astype(np.float32), (0, 0), 0.8), c=(ccx, ccy),
                           v=(dirx * r.uniform(900, 1700) * (1 + 300 / d), diry * r.uniform(900, 1700) * (1 + 300 / d) - r.uniform(0, 400)),
                           rot=r.uniform(-260, 260), zoom=r.uniform(0.25, 1.1), delay=min(0.06, d / 9000)))
    dist = np.sqrt((XX - cx) ** 2 + (YY - cy) ** 2)
    _lazy[key] = (shards, edges, dist)
    return _lazy[key]


def t_shatter(t, c, A, B, a, b, cx=0.5, cy=0.40):
    """Cracks race out from an impact point, then the frame breaks into glass shards that fly at camera."""
    shards, edges, dist = _shards(cx * W, cy * H)
    if t < c:                                                     # cracks spreading, tiny shake
        u = clamp((t - a) / (c - a)); rr = 60 + ein(u) * 1.4 * max(W, H)
        f = layer(A, t, 1 + 0.03 * u)
        crack = edges * ss(rr + 30, rr - 30, dist)
        glow = cv2.GaussianBlur(crack, (0, 0), 6)
        return f * (1 - 0.5 * crack[..., None]) + (crack * 0.9 + glow * 0.8)[..., None] * np.float32([1.0, 0.98, 0.95])
    u = t - c; F = lazy(('shatter_src', A, round(c, 3)), lambda: layer(A, c - 1.0 / FPS))
    f = layer(B, t, 1 + 0.10 * (1 - eout(u / 0.3))) + 0.55 * (1 - eout(u / 0.18))   # impact flash on the next shot
    for sh in sorted(shards, key=lambda q: q['zoom']):
        uu = max(0.0, u - sh['delay'])
        x0, y0, x1, y1 = sh['box']; ccx, ccy = sh['c']
        dx = sh['v'][0] * uu; dy = sh['v'][1] * uu + 2600 * uu * uu
        z = 1 + sh['zoom'] * uu * 3.0; ang = sh['rot'] * uu
        M = cv2.getRotationMatrix2D((ccx - x0, ccy - y0), ang, z); M[0, 2] += x0 + dx; M[1, 2] += y0 + dy
        alpha = 1 - ease((uu - 0.12) / 0.2)
        if alpha <= 0.01: continue
        src = F[y0:y1, x0:x1]; m = sh['mask']
        rim = np.clip(m - cv2.erode(m, np.ones((5, 5), np.uint8)), 0, 1)
        tile = np.dstack([src * m[..., None] + rim[..., None] * 0.8, m])            # bright glass edge
        lay = cv2.warpAffine(tile, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        f = f * (1 - lay[..., 3:4] * alpha) + lay[..., :3] * alpha
    return f


# ------------------------------------------------------------ microscope dive
def _cells(N=1400, seed=33):
    def build():
        r = np.random.default_rng(seed)
        plasma = (0.80 + 0.08 * fbm(N, N, 4, seed, 6))[..., None] * np.float32([0.78, 0.84, 0.97])
        img = plasma.copy(); yy, xx = np.mgrid[0:N, 0:N].astype(np.float32); placed = []
        for _ in range(260):
            x, y, R = r.uniform(0, N), r.uniform(0, N), r.uniform(34, 52)
            if any((x - px) ** 2 + (y - py) ** 2 < (0.85 * (R + pr)) ** 2 for px, py, pr in placed): continue
            placed.append((x, y, R))
            x0, x1, y0, y1 = int(max(0, x - R - 4)), int(min(N, x + R + 4)), int(max(0, y - R - 4)), int(min(N, y + R + 4))
            d = np.sqrt((xx[y0:y1, x0:x1] - x) ** 2 + (yy[y0:y1, x0:x1] - y) ** 2) / R
            body = ss(1.02, 0.94, d)[..., None]
            shade = (0.55 + 0.30 * ss(0.15, 0.6, d) - 0.25 * ss(0.85, 1.0, d))[..., None]   # biconcave: pale centre, dark rim
            col = np.float32([0.22, 0.18, 0.80]) * shade + np.float32([0.10, 0.10, 0.22]) * (1 - shade)
            img[y0:y1, x0:x1] = img[y0:y1, x0:x1] * (1 - body) + col * body
        return img
    return lazy(('cells', N, seed), build)


def _reticle(R):
    def build():
        can = np.zeros((H, W), np.uint8); cx, cy = W // 2, int(H * 0.45)
        cv2.line(can, (cx - int(R * 0.8), cy), (cx + int(R * 0.8), cy), 255, 2, cv2.LINE_AA)
        cv2.line(can, (cx, cy - int(R * 0.8)), (cx, cy + int(R * 0.8)), 255, 2, cv2.LINE_AA)
        for k in range(-8, 9):
            L = 22 if k % 4 == 0 else 11; x = cx + int(k * R * 0.09)
            cv2.line(can, (x, cy - L), (x, cy + L), 255, 2, cv2.LINE_AA)
        cv2.circle(can, (cx, cy), int(R * 0.32), 255, 1, cv2.LINE_AA)
        cv2.putText(can, '40x', (cx + int(R * 0.45), cy + int(R * 0.62)), cv2.FONT_HERSHEY_SIMPLEX, 1.1, 255, 2, cv2.LINE_AA)
        return cv2.GaussianBlur(can.astype(np.float32) / 255, (0, 0), 0.7)
    return lazy(('reticle', R), build)


def t_scope(t, c, A, B, a, b, px=0.86, py=0.64):
    """Dive into the blood on the slide, a microscope eyepiece full of blood cells, then pull back out to the next shot."""
    cx, cy = W / 2, H * 0.45; R0 = 0.46 * W; Rbig = 1.3 * math.hypot(W, H) / 2
    D = np.sqrt((XX - cx) ** 2 + (YY - cy) ** 2)
    cells = _cells(); N = cells.shape[0]
    drift = (t - a) * 140; ox = int(N / 2 - W / 2 + drift) % (N - W); oy = int(N / 2 - H / 2) % max(1, N - H)
    def scope_view(focus):
        v = cells[oy:oy + H, ox:ox + W] if N >= H else cv2.resize(cells, (W, H))
        return blur(v, 18 * (1 - focus))
    if t < c:
        u = clamp((t - a) / (c - a)); z = math.exp(math.log(7.0) * ein(u))
        A_ = layer(A, t); K = 4; acc = np.zeros_like(A_)
        e = ease(u * 1.15)
        for k in range(K):                                           # travel to the slide's blood while zooming in (zoom blur)
            zz = z * (1 - 0.05 * k * u)
            M = np.float32([[zz, 0, 0], [0, zz, 0]])
            tx, ty = lerp(px * W, cx, e), lerp(py * H, cy, e)            # the smear drifts to the eyepiece centre
            M[0, 2] = tx - zz * px * W; M[1, 2] = ty - zz * py * H
            acc += cv2.warpAffine(A_, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        f = acc / K
        R = lerp(Rbig, R0, ease(u))
        mix = ease((u - 0.55) / 0.45)
        f = f * (1 - mix) + scope_view(ease((u - 0.7) / 0.3)) * mix
    else:
        u = clamp((t - c) / (b - c))
        R = lerp(R0, Rbig, ein(u))
        mix = 1 - ease(u / 0.6)
        Bz = layer(B, t, 1 + 1.6 * (1 - eout(u)))
        f = scope_view(1.0) * mix + Bz * (1 - mix)
    inside = ss(R + 3, R - 3, D)[..., None]
    ring = (np.exp(-((D - R) / 10) ** 2))[..., None]
    f = f * inside + np.float32([0.02, 0.02, 0.03]) * (1 - inside)            # black eyepiece surround
    f = f * (1 - 0.45 * ss(R * 0.6, R, D)[..., None])                         # lens vignette
    f = f + ring * np.float32([0.10, 0.25, 0.55]) * 0.8                        # warm chromatic rim
    ret_a = (1 - ease(abs((t - c) / (b - a)) * 3.2))
    if ret_a > 0: f = f + (_reticle(int(R0))[..., None] * 0.55 * ret_a) * np.float32([0.15, 0.15, 0.12]) - (_reticle(int(R0))[..., None] * 0.35 * ret_a)
    return f


# ------------------------------------------------------------ kill-room plastic
def _plastic_tex():
    def build():
        h, w = H + 400, W * 3
        n1 = cv2.resize(fbm(h // 8, w // 32, 4, 81, 3), (w, h), interpolation=cv2.INTER_CUBIC)      # long vertical folds
        n2 = cv2.resize(fbm(h // 4, w // 8, 4, 82, 4), (w, h), interpolation=cv2.INTER_CUBIC)       # creases
        x = np.arange(w, dtype=np.float32)[None, :]
        n3 = cv2.resize(fbm(h // 16, w // 16, 3, 83, 3), (w, h), interpolation=cv2.INTER_CUBIC)
        Hm = 0.45 * n1 + 0.35 * n2 + 0.12 * (0.5 + 0.5 * np.sin(x / (70 + 60 * n3) + 4 * n1)) + 0.08 * n3
        gx = cv2.Sobel(Hm, cv2.CV_32F, 1, 0, ksize=5); gy = cv2.Sobel(Hm, cv2.CV_32F, 0, 1, ksize=5)
        spec = (np.clip(gx * 5.0 + gy * 1.5 - 0.18, 0, 1) ** 2 + 0.4 * np.clip(-gx * 3.5 - 0.35, 0, 1) ** 2) * (0.5 + n2)
        return gx.astype(np.float32), gy.astype(np.float32), spec.astype(np.float32)
    return lazy('plastic', build)


def t_plastic(t, c, A, B, a, b):
    """A sheet of kill-room plastic drags across the frame; the next shot is behind it."""
    gx, gy, spec = _plastic_tex(); u = clamp((t - a) / (b - a))
    band = 1.1 * W
    lead = lerp(W + 120, -band - 160, 0.25 * u + 0.75 * ease(u))           # leading edge moves right -> left
    y = np.arange(H, dtype=np.float32)[:, None]
    wav = 40 * np.sin(y / 170 + 3 * t) + 18 * np.sin(y / 61 - 5 * t) + 0.16 * (y - H / 2)     # tilted, rippling edges
    L = lead + wav; T = L + band
    inside = ss(L - 6, L + 6, XX) * ss(T + 6, T - 6, XX)
    sw = (L + T) / 2 + 60 * np.sin(y / 230 + t)
    off = int(-lead) % (gx.shape[1] - W)
    sgx = gx[100:100 + H, off:off + W]; sgy = gy[100:100 + H, off:off + W]; ssp = spec[100:100 + H, off:off + W]
    k = 55
    fa = layer(A, t); fb = layer(B, t)
    under = np.where((XX < sw)[..., None], fa, fb) if inside.max() > 0 else (fa if t < c else fb)
    under = cv2.remap(under, XX + sgx * k, YY + sgy * k * 0.4, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    milky = under * 0.82 + np.float32([0.14, 0.14, 0.13]) + (ssp * 0.65)[..., None]
    rim = np.exp(-((XX - L) / 7) ** 2) * 0.9 + np.exp(-((XX - L - 22) / 30) ** 2) * 0.25 + np.exp(-((XX - T) / 9) ** 2) * 0.5
    milky = milky + rim[..., None]
    base = np.where((XX < L)[..., None], fa, fb)                                 # outside the sheet: A ahead of it, B behind it
    m = inside[..., None]
    shadow = ss(L - 40, L, XX) * (XX < L) * 0.35
    return base * (1 - m) * (1 - shadow[..., None]) + milky * m


# ------------------------------------------------------------ kill knife (shape window)
def _knife_geo():
    def build():
        Lb = 900.0; sv = np.linspace(0, 1, 60)
        spine = -90 + 50 * np.clip((sv - 0.75) / 0.25, 0, 1) ** 2
        edge = 90 - 130 * np.clip((sv - 0.45) / 0.55, 0, 1) ** 1.8
        blade = np.concatenate([np.stack([sv * Lb, spine], 1), np.stack([sv[::-1] * Lb, edge[::-1]], 1)])
        bolster = np.array([[-46, -96], [0, -92], [0, 92], [-46, 98]], np.float64)
        th = np.linspace(-np.pi / 2, np.pi / 2, 16)
        handle = np.concatenate([[[-46, -72], [-380, -66]], np.stack([-380 - 46 * np.cos(th), -4 + 62 * np.sin(th)], 1)[::-1] * [1, 1], [[-380, 58], [-46, 66]]])
        edge_line = np.stack([sv * Lb, edge - 6], 1); spine_line = np.stack([sv * Lb * 0.97, spine + 4], 1)
        return dict(Lb=Lb, blade=blade, bolster=bolster, handle=handle, edge=edge_line, spine=spine_line, rivets=[-130, -240, -330])
    return lazy('knife', build)


def t_knife(t, c, A, B, a, b):
    """Dexter's carving knife thrusts in; the blade is a window onto the next shot; the camera dives into the blade."""
    G = _knife_geo(); Lb = G['Lb']
    if t < c:                                                   # thrust in from bottom-right, settle across the frame
        u = eout((t - a) / (c - a))
        cx, cy = lerp(1.55 * W, 0.70 * W, u), lerp(1.30 * H, 0.66 * H, u)
        ang = math.radians(lerp(-112, -122, u)); sc = lerp(0.95, 1.05, u)
        P = (0.0, 0.0); fa = layer(A, t)
    else:                                                       # pan into the wide part of the blade until it fills the frame
        u = clamp((t - c) / (b - c)); e = ease(u)
        P = (Lb * 0.30 * e, 0.0)
        cx, cy = lerp(0.70 * W, W / 2, e), lerp(0.66 * H, H / 2, e)
        ang = math.radians(lerp(-122, -90, e)); sc = 1.05 * math.exp(math.log(10.0) * ein(u) ** 0.9 + 0.0 * u)
        fa = layer(A, t)
    ca, sa = math.cos(ang) * sc, math.sin(ang) * sc
    def tf(pts):                                                # local knife coords -> screen (P lands on (cx, cy))
        p = np.asarray(pts, np.float64) - P
        return np.stack([cx + p[:, 0] * ca - p[:, 1] * sa, cy + p[:, 0] * sa + p[:, 1] * ca], 1)
    def poly_mask(pts, S=4):
        m = np.zeros((H, W), np.uint8); q = np.round(tf(pts) * S).astype(np.int64)
        q = np.clip(q, -2 ** 30, 2 ** 30).astype(np.int32)
        cv2.fillPoly(m, [q], 255, cv2.LINE_AA, shift=2)
        return m.astype(np.float32) / 255
    blade = poly_mask(G['blade']); bol = poly_mask(G['bolster']); hand = poly_mask(G['handle'])
    whole = np.clip(blade + bol + hand, 0, 1)
    # local x along the blade for every pixel (for the travelling glint and steel shading)
    inv = 1.0 / (sc * sc); lx = ((XX - cx) * ca + (YY - cy) * sa) * inv + P[0]; ly = (-(XX - cx) * sa + (YY - cy) * ca) * inv + P[1]
    fb = layer(B, t, 1 + 0.08 * (1 - clamp((t - a) / (b - a))))
    steel = fb * 0.86 + np.float32([0.07, 0.06, 0.05]) + (0.10 * ss(40, 95, np.abs(ly)))[..., None]   # cool steel tint toward the rims
    gpos = lerp(-100, Lb + 150, clamp((t - a) / (c - a + 0.12)))
    glint = np.exp(-((lx - gpos + 0.35 * ly) / 45) ** 2) * 0.55 * (1 - clamp((t - c) / 0.12))
    steel = steel + glint[..., None]
    shadow = cv2.GaussianBlur(np.roll(np.roll(whole, int(40 * min(sc, 3)), 0), int(26 * min(sc, 3)), 1), (0, 0), 18) * 0.55
    f = fa * (1 - shadow[..., None])
    f = f * (1 - blade[..., None]) + steel * blade[..., None]
    # sharpened edge bevel and spine highlight, drawn as lines scaled with the knife
    lines = np.zeros((H, W), np.uint8)
    for key, th, val in (('edge', 10, 255), ('spine', 4, 170)):
        q = np.clip(np.round(tf(G[key]) * 4), -2 ** 30, 2 ** 30).astype(np.int32)
        cv2.polylines(lines, [q], False, val, max(1, int(th * sc)), cv2.LINE_AA, shift=2)
    lf = cv2.GaussianBlur(lines.astype(np.float32) / 255, (0, 0), max(1.0, 1.5 * min(sc, 4))) * blade
    f = f + (lf * (0.55 + glint))[..., None] * np.float32([1.0, 0.98, 0.95])
    # bolster: brushed steel; handle: black grip with a top highlight and steel rivets
    bcol = (0.50 + 0.30 * ss(-60, 80, ly))[..., None] * np.float32([0.95, 0.95, 1.0])
    f = f * (1 - bol[..., None]) + bcol * bol[..., None]
    hcol = np.float32([0.045, 0.045, 0.05]) + (0.20 * np.exp(-((ly + 45) / 14) ** 2))[..., None]
    f = f * (1 - hand[..., None]) + hcol * hand[..., None]
    for rx in G['rivets']:
        p = tf([[rx, -4]])[0]; r = max(1, int(13 * sc))
        if -r < p[0] < W + r and -r < p[1] < H + r:
            cv2.circle(f, (int(p[0]), int(p[1])), r, (0.78, 0.78, 0.82), -1, cv2.LINE_AA)
            cv2.circle(f, (int(p[0] - r * 0.3), int(p[1] - r * 0.3)), max(1, r // 3), (1.0, 1.0, 1.0), -1, cv2.LINE_AA)
    return f


# ============================================================ THE FLASH kit: lightning, emblem, speed
GOLD_CORE = np.float32([0.80, 0.97, 1.00]); GOLD_GLOW = np.float32([0.08, 0.50, 1.00]); BLUE_GLOW = np.float32([1.00, 0.62, 0.25])


def bolt_path(p0, p1, rng, depth=6, disp=0.14):
    """Midpoint-displacement lightning between two points; returns an (N,2) polyline."""
    pts = [np.float64(p0), np.float64(p1)]
    for d in range(depth):
        out = [pts[0]]
        for a, b in zip(pts, pts[1:]):
            m = (a + b) / 2; v = b - a; n = np.array([-v[1], v[0]]); L = np.hypot(*v) + 1e-9
            out += [m + n / L * rng.normal(0, disp * L * 0.5), b]
        pts = out
    return np.array(pts)


def bolt_tree(p0, p1, rng, branches=4, disp=0.14, depth=6):
    main = bolt_path(p0, p1, rng, depth, disp); segs = [(main, 1.0)]
    for _ in range(branches):
        k = rng.integers(len(main) // 6, len(main) * 5 // 6); s0 = main[k]
        v = main[min(len(main) - 1, k + 8)] - main[max(0, k - 8)]; L = np.hypot(*v) + 1e-9
        ang = math.atan2(v[1], v[0]) + rng.choice([-1, 1]) * rng.uniform(0.35, 0.9)
        ln = np.hypot(*(np.float64(p1) - p0)) * rng.uniform(0.15, 0.4)
        segs.append((bolt_path(s0, s0 + ln * np.array([math.cos(ang), math.sin(ang)]), rng, depth - 2, disp * 1.2), rng.uniform(0.35, 0.6)))
    return segs


def draw_bolts(segs, width=3.0, glow=GOLD_GLOW, core=GOLD_CORE, amp=1.0, q=2):
    """Additive lightning: white-hot core, tight inner glow, wide soft glow (glows drawn at 1/q res)."""
    if amp <= 0.01 or not segs: return None
    c = np.zeros((H, W), np.uint8); g = np.zeros((H // q, W // q), np.uint8)
    for pts, w in segs:
        P = np.round(pts * 4).astype(np.int32)
        cv2.polylines(c, [P], False, 255, max(1, int(round(width * w))), cv2.LINE_AA, shift=2)
        Pq = np.round(pts / q * 4).astype(np.int32)
        cv2.polylines(g, [Pq], False, 255, max(1, int(round(width * w * 2.2 / q))), cv2.LINE_AA, shift=2)
    cf = c.astype(np.float32) / 255
    gf = g.astype(np.float32) / 255
    g1 = cv2.resize(cv2.GaussianBlur(gf, (0, 0), 3.5 / q), (W, H)); g2 = cv2.resize(cv2.GaussianBlur(gf, (0, 0), 22 / q), (W, H))
    white = np.float32([1.0, 1.0, 1.0])
    return amp * (cf[..., None] * (white * 0.6 + core * 0.9) + g1[..., None] * glow * 1.5 + g2[..., None] * glow * 0.75)


def add(f, layer_):
    return f if layer_ is None else f + layer_


# ---- overlays
def _landmarks(i, k):
    key = ('lm', i, k)
    if key in _lazy: return _lazy[key]
    import os
    det = lazy('yunet', lambda: cv2.FaceDetectorYN.create(os.path.expanduser('~/.cache/rea-video/models/face_detection_yunet_2023mar.onnx'), '', (W // 2, H // 2), 0.5, 0.3, 20))
    img = cv2.resize(clip_frame(i, k), (W // 2, H // 2)); _, faces = det.detect(img); res = None
    if faces is not None and len(faces):
        f_ = faces[int(np.argmax(faces[:, 2] * faces[:, 3]))] * 2
        res = (f_[4:6], f_[6:8], f_[2])                    # right eye, left eye, face width
    if len(_lazy) > 400: [_lazy.pop(kk) for kk in [kk for kk in _lazy if isinstance(kk, tuple) and kk[0] == 'lm'][:200]]
    _lazy[key] = res
    return res


def eyes_fx(f, i, t, s):
    """A brief gold gleam in Barry's eyes on the hit (subtle: small core, soft glow, short anamorphic glint)."""
    u = t - s['start']; amp = ease(u / 0.06) * math.exp(-max(0.0, u - 0.08) / 0.22)
    if amp < 0.02: return f
    k = round((PRE + t - s['start']) * FPS); m = META[i]; k = int(min(max(k, m['lo']), m['hi']))
    lm = None
    for kk in (k, k - 1, k - 2, k + 1, k - 4, k + 3):
        lm = _landmarks(i, kk)
        if lm: break
    if not lm: return f
    z = max(lerp(s['push'][0], s['push'][1], u / (s['end'] - s['start'])), 1.0)
    cx0, cy0 = W / 2, H * 0.42; fw = lm[2] * z; lay = np.zeros((H, W, 3), np.float32)
    for e in lm[:2]:
        x, y = cx0 + (e[0] - cx0) * z, cy0 + (e[1] - cy0) * z; r = max(3.0, fw * 0.018)
        d2 = (XX - x) ** 2 + (YY - y) ** 2
        lay += np.exp(-d2 / (2 * r * r))[..., None] * GOLD_CORE * 1.3
        lay += np.exp(-d2 / (2 * (r * 3.5) ** 2))[..., None] * GOLD_GLOW * 0.45
        lay += (np.exp(-((XX - x) / (fw * 0.35)) ** 2 - ((YY - y) / (r * 0.5)) ** 2) * 0.25)[..., None] * GOLD_GLOW
    return f + lay * amp


def crackle_fx(f, i, t, s, build=False, glow=GOLD_GLOW, region=None, amp_scale=1.0):
    """Lightning arcs crawling over the shot (re-struck every 3 frames); with build=True they grow toward the end."""
    u = (t - s['start']) / (s['end'] - s['start'])
    amp = (0.15 + 0.85 * ein(u)) if build else 0.8
    rng = np.random.default_rng(abs(int(t * FPS / 3)) + 17 * i); segs = []
    n = 2 + int((5 if build else 3) * (u if build else 1))
    for _ in range(n):
        if region is not None:                                   # arcs jumping across a region (e.g. the speed cap)
            x0, y0, x1, y1 = region
            p0 = (W * rng.uniform(x0, x1), H * rng.uniform(y0, y1)); ang = rng.uniform(0, 2 * np.pi); ln = W * rng.uniform(0.12, 0.3)
            segs += bolt_tree(p0, (p0[0] + ln * math.cos(ang), p0[1] + ln * math.sin(ang)), rng, branches=1, depth=6)
            continue
        side = rng.integers(4)
        x0, y0 = (rng.uniform(0, W), H * rng.uniform(0.55, 1.0)) if side < 2 else ((0 if side == 2 else W), H * rng.uniform(0.2, 0.9))
        x1, y1 = W * rng.uniform(0.25, 0.75), H * rng.uniform(0.35, 0.8)
        segs += bolt_tree((x0, y0), (x0 + (x1 - x0) * 0.6, y0 + (y1 - y0) * 0.6), rng, branches=2, depth=6)
    flick = 0.75 + 0.25 * math.sin(t * 157)
    return add(f, draw_bolts(segs, 3.2, glow=glow, amp=amp * amp_scale * flick))


# ---- beat-driven overlays (KICKS / SNARES: half-beat indices from the section analysis, in config.py)
def _hits(name):
    import config as _c
    return [GRID0 + k * HALF for k in getattr(_c, name, [])]


def bump_amount(t, s):
    """Zoom bump on each strong kick inside the shot: +4.5% that settles in ~0.1 s.
    config BUMP (amount) and BUMP_KICKS (which kicks; default KICKS) tune it."""
    import config as _c
    amt = getattr(_c, 'BUMP', 0.045); ks = _hits('BUMP_KICKS') if hasattr(_c, 'BUMP_KICKS') else _hits('KICKS'); dec = getattr(_c, 'BUMP_DECAY', 0.09)
    z = 0.0
    if t < s['start'] and any(abs(tk - s['start']) < 1e-6 for tk in ks):
        return amt                                       # loop tail of shot 0: hold the downbeat bump so the seam matches
    for tk in ks:
        if s['start'] - 1e-6 <= tk <= t: z = max(z, amt * math.exp(-(t - tk) / dec))
    return z


def rgbhit_fx(f, t, s):
    """Colour-split kick on each snare inside the shot: channels jump apart and snap back."""
    a = 0.0
    for tk in _hits('SNARES'):
        if s['start'] - 1e-6 <= tk <= t: a = max(a, math.exp(-(t - tk) / 0.07))
    sp = int(18 * a)
    if sp < 2: return f
    f = f.copy(); f[..., 2] = np.roll(f[..., 2], sp, 1); f[..., 0] = np.roll(f[..., 0], -sp, 1)
    return f


def echo_fx(f, i, t, s):
    """Speedster afterimages: earlier frames trail behind in gold (screen-blended, fading)."""
    out = f
    for k, (dt, al) in enumerate(((0.07, 0.55), (0.15, 0.38), (0.25, 0.24))):
        g = shot(i, t - dt) * np.float32([0.35, 0.75, 1.15])
        out = 1 - (1 - out) * (1 - np.clip(g * al, 0, 1))
    return out


def sweep_fx(f, t, s):
    """A single warm light sweep crosses the shot (diagonal band + anamorphic glint at its hot spot)."""
    u = (t - s['start']) / (s['end'] - s['start']); pos = lerp(-0.4, 1.4, ease(u))
    d = (XX / W * 0.8 + YY / H * 0.45) - pos * 1.25
    band = np.exp(-(d / 0.05) ** 2) * 0.45 + np.exp(-(d / 0.16) ** 2) * 0.18
    return f + band[..., None] * np.float32([0.35, 0.72, 1.0])


# ---- transitions
def t_speedlines(t, c, A, B, a, b, d=1):
    """Barry speeds off: the frame smears into golden speed streaks and the next shot resolves out of them."""
    u = clamp((t - a) / (b - a)); env = math.sin(math.pi * u)
    if t < c: f = layer(A, t); sh = d * W * 0.45 * ein((t - a) / (c - a))
    else: f = layer(B, t); sh = -d * W * 0.45 * (1 - eout((t - c) / (b - c)))
    k = int(260 * env) | 1
    f = np.roll(f, int(sh), axis=1)
    if k > 1: f = cv2.blur(f, (k, 1))
    f = f * (1 - 0.25 * env) + np.float32([0.03, 0.18, 0.35]) * env               # warm wash at the peak
    r = np.random.default_rng(5); can = np.zeros((H // 2, W // 2), np.uint8)
    for j in range(70):
        y = r.uniform(0, H / 2); L = r.uniform(80, 420); v = r.uniform(2500, 5200) / 2; th = int(r.uniform(1, 5))
        x = (r.uniform(-W, W) / 2 + d * v * (t - a)) % (W / 2 + L) - L / 2 if d > 0 else (r.uniform(0, 2 * W) / 2 - v * (t - a)) % (W / 2 + L) - L / 2
        cv2.line(can, (int(x), int(y)), (int(x + d * L), int(y)), int(255 * r.uniform(0.5, 1)), th, cv2.LINE_AA)
    cf = cv2.resize(can.astype(np.float32) / 255, (W, H))
    gl = cv2.GaussianBlur(cf, (0, 0), 6)
    return f + (cf[..., None] * GOLD_CORE * 0.9 + gl[..., None] * GOLD_GLOW * 1.4) * env


def t_phase(t, c, A, B, a, b):
    """Speed-Force vibration: the frame shakes apart into RGB-split ghost copies and phases into the next shot."""
    u = clamp((t - a) / (b - a)); env = math.sin(math.pi * u) ** 0.7
    src = layer(A, t) if t < c else layer(B, t)
    amp = 46 * env; acc = np.zeros_like(src)
    for k in range(5):
        ph = t * 55 * 2 * math.pi + k * 1.3
        dx = int(amp * math.sin(ph)); dy = int(amp * 0.25 * math.cos(ph * 1.7))
        acc += np.roll(np.roll(src, dx, 1), dy, 0)
    f = acc / 5
    sp = int(22 * env)
    if sp:
        f = f.copy(); f[..., 2] = np.roll(f[..., 2], sp, 1); f[..., 0] = np.roll(f[..., 0], -sp, 1)
    f = f + 0.55 * max(0.0, 1 - abs(t - c) / 0.06)                               # white-gold pop on the phase
    rng = np.random.default_rng(int(t * FPS / 2)); segs = []
    for _ in range(3):
        x0, y0 = W * rng.uniform(0.2, 0.8), H * rng.uniform(0.25, 0.75)
        segs += bolt_tree((x0, y0), (x0 + rng.uniform(-300, 300), y0 + rng.uniform(-400, 400)), rng, 1, 0.3, 6)
    return add(f, draw_bolts(segs, 2.5, amp=env))


def _emblem_ref(i, k_ref):
    """Detect the emblem once on a clean reference frame: hull, centre, radius, mask, and a grey template."""
    key = ('embref', i)
    if key in _lazy: return _lazy[key]
    best = None
    for k in [k_ref] + [k_ref - d for d in range(3, 40, 3)]:
        img = clip_frame(i, k); hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        m = ((hsv[..., 1] < 70) & (hsv[..., 2] > 165)).astype(np.uint8) * 255; m[:int(0.5 * H)] = 0
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
        for c in cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
            A_ = cv2.contourArea(c); (x, y), r = cv2.minEnclosingCircle(c)
            if A_ > 3000 and A_ / (np.pi * r * r) > 0.45 and (best is None or A_ > best[0]):   # round, solid: the disc, not lightning
                best = (A_, k, c, x, y, r, m)
        if best: break
    _, k, c, x, y, r, m = best
    hull = cv2.convexHull(c)[:, 0, :].astype(np.float64); M = cv2.moments(c); cen = np.array([M['m10'] / M['m00'], M['m01'] / M['m00']])
    g = cv2.cvtColor(clip_frame(i, k), cv2.COLOR_BGR2GRAY); q = 4
    x0, y0 = int(max(0, x - r)), int(max(0, y - r)); x1, y1 = int(min(W, x + r)), int(min(H, y + r))
    tmpl = cv2.resize(g[y0:y1, x0:x1], ((x1 - x0) // q, (y1 - y0) // q))
    disc = np.zeros((H, W), np.uint8); cv2.drawContours(disc, [c], -1, 255, -1)
    _lazy[key] = dict(hull=hull, cen=cen, r=r, tl=np.array([x0, y0]), tmpl=tmpl, q=q, mask=disc > 0)
    return _lazy[key]


def _emblem_geo(i, k, k_ref):
    """Follow the reference emblem into frame k by template matching (lightning over it doesn't fool this)."""
    key = ('emb', i, k)
    if key in _lazy: return _lazy[key]
    ref = _emblem_ref(i, k_ref); q = ref['q']
    g = cv2.resize(cv2.cvtColor(clip_frame(i, k), cv2.COLOR_BGR2GRAY), (W // q, H // q))
    th, tw = ref['tmpl'].shape; x0, y0 = ref['tl'] // q
    sx0, sy0 = max(0, x0 - W // q // 4), max(0, y0 - H // q // 10)          # search only near the emblem (not the white face)
    win = g[sy0:min(H // q, y0 + th + H // q // 10), sx0:min(W // q, x0 + tw + W // q // 4)]
    res = cv2.matchTemplate(win, ref['tmpl'], cv2.TM_CCOEFF_NORMED); _, sc, _, loc = cv2.minMaxLoc(res)
    d = (np.array(loc) + [sx0, sy0]) * q - ref['tl'] if sc > 0.3 else np.zeros(2)
    mask = np.roll(np.roll(ref['mask'], int(d[1]), 0), int(d[0]), 1)
    out = (ref['hull'] + d, ref['cen'] + d, ref['r'], mask)
    _lazy[key] = out
    return out


def t_emblem_real(t, c, A, B, a, b):
    """Built on the real chest emblem: it charges (rim glow, light sweep, arcs off its edge, camera pushes in),
    flashes on the hit, and the next shot opens out of the emblem itself as the camera dives into it."""
    s = SHOTS[A]; k = round((PRE + min(t, c) - s['start']) * FPS); m = META[A]; k = int(min(max(k, m['lo']), m['hi']))
    k_ref = int(min(max(round((PRE + c - s['start']) * FPS), m['lo']), m['hi']))
    hull, cen, rad, wmask = _emblem_geo(A, k, k_ref)
    zp = max(lerp(s['push'][0], s['push'][1], (min(t, c) - s['start']) / (s['end'] - s['start'])), 1.0)
    cx0, cy0 = W / 2, H * 0.42
    toF = lambda p: np.stack([cx0 + (p[..., 0] - cx0) * zp, cy0 + (p[..., 1] - cy0) * zp], -1)    # clip -> frame (shot push)
    E = toF(cen); hullF = toF(hull); radF = rad * zp
    charge = clamp((t - a) / (c - a))
    if t < c: Z = 1 + 0.35 * ein(charge); e = 0.0
    else:
        u = clamp((t - c) / (b - c)); Z = 1.35 * math.exp(math.log(math.hypot(W, H) / radF * 1.25 / 1.35) * ein(u ** 0.85)); e = ease(u)
    T = E.copy()                                                  # zoom about the emblem itself: never exposes the frame edge
    M = np.float32([[Z, 0, T[0] - Z * E[0]], [0, Z, T[1] - Z * E[1]]])
    fa = cv2.warpAffine(layer(A, min(t, c - 1e-3) if t >= c else t), M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    hullS = hullF * Z + (T - Z * E)
    poly = np.zeros((H, W), np.uint8); cv2.fillPoly(poly, [np.clip(np.round(hullS * 4), -2 ** 30, 2 ** 30).astype(np.int32)], 255, cv2.LINE_AA, shift=2)
    pm = poly.astype(np.float32) / 255
    rim = cv2.GaussianBlur(np.clip(cv2.dilate(pm, np.ones((9, 9), np.uint8)) - cv2.erode(pm, np.ones((9, 9), np.uint8)), 0, 1), (0, 0), 3)
    if t < c:
        f = fa
        # light sweep across the emblem's own white surface
        wm = cv2.warpAffine(cv2.resize(wmask.astype(np.float32), (W, H)) if wmask.shape != (H, W) else wmask.astype(np.float32), np.float32([[zp, 0, cx0 - zp * cx0], [0, zp, cy0 - zp * cy0]]), (W, H))
        wm = cv2.warpAffine(wm, M, (W, H))
        sw = np.exp(-(((XX - YY * 0.6) - lerp(T[0] - radF * Z * 1.6 - 0.6 * T[1], T[0] + radF * Z * 1.6 - 0.6 * T[1], charge)) / (radF * Z * 0.22)) ** 2)
        f = f + (sw * wm * 0.75)[..., None] * np.float32([0.85, 0.95, 1.0])
        f = f + (rim * (0.5 + 1.3 * charge))[..., None] * GOLD_GLOW + cv2.GaussianBlur(rim, (0, 0), 18)[..., None] * GOLD_GLOW * 1.2 * charge
        rng = np.random.default_rng(int(t * FPS / 2)); segs = []
        for _ in range(2 + int(5 * charge)):                     # arcs leaping off the emblem's rim
            p0 = hullS[rng.integers(len(hullS))]; d = p0 - T; d = d / (np.hypot(*d) + 1e-9)
            ang = math.atan2(d[1], d[0]) + rng.uniform(-0.7, 0.7); ln = radF * Z * rng.uniform(0.8, 2.2)
            segs += bolt_tree(p0, (p0[0] + ln * math.cos(ang), p0[1] + ln * math.sin(ang)), rng, branches=1, depth=6)
        f = add(f, draw_bolts(segs, 3.2, amp=0.5 + 0.8 * charge))
        return f + 0.9 * ein(clamp((t - (c - 0.05)) / 0.05))     # flash into the hit
    u = clamp((t - c) / (b - c))
    fb = layer(B, t, 1 + 0.25 * (1 - eout(u)))
    f = fa * (1 - pm[..., None]) + fb * pm[..., None]
    f = f + (rim * (1 - u))[..., None] * GOLD_GLOW * 1.8 + cv2.GaussianBlur(rim, (0, 0), 24)[..., None] * GOLD_GLOW * (1 - u)
    return f + 0.9 * (1 - eout(u / 0.18))


_STRIKE = {}


def t_strike(t, c, A, B, a, b):
    """A lightning bolt strikes top to bottom and tears the frame open along its path; the next shot is in the tear."""
    if 'main' not in _STRIKE:
        rng = np.random.default_rng(9)
        _STRIKE['main'] = bolt_path((W * 0.62, -40), (W * 0.40, H + 40), rng, 8, 0.16)
        m = np.full((H, W), 255, np.uint8); cv2.polylines(m, [np.round(_STRIKE['main'] * 4).astype(np.int32)], False, 0, 3, cv2.LINE_8, shift=2)
        _STRIKE['dist'] = cv2.distanceTransform(m, cv2.DIST_L2, 5)
        _STRIKE['noise'] = fbm(H // 4, W // 4, 4, 13, 8)
    main, dist = _STRIKE['main'], _STRIKE['dist']
    rng = np.random.default_rng(int(t * FPS / 2))
    if t < c:                                                     # leader: the bolt draws itself downward, flickering
        u = clamp((t - a) / (c - a)); n = max(2, int(len(main) * eout(u)))
        segs = [(main[:n], 1.0)] + [(bolt_path(main[k], main[k] + rng.normal(0, 140, 2), rng, 5, 0.3), 0.4) for k in rng.integers(0, n, 3)]
        f = layer(A, t) * (1 - 0.25 * u)
        return add(f, draw_bolts(segs, 5.0, amp=0.6 + 0.6 * u))
    u = clamp((t - c) / (b - c))
    w = (40 + 1700 * ein(u ** 0.8)) * (0.85 + 0.3 * cv2.resize(_STRIKE['noise'], (W, H)))
    m = ss(w + 8, w - 8, dist)[..., None]
    f = layer(A, t) * (1 - m) + layer(B, t, 1 + 0.12 * (1 - u)) * m
    rim = np.exp(-((dist - w) / 10) ** 2) * (1 - ease(u / 0.6))
    f = f + rim[..., None] * GOLD_GLOW * 1.8 + 0.85 * (1 - eout(u / 0.25))          # torn glowing edges + strike flash
    segs = [(main, 1.0)] + [(bolt_path(main[k], main[k] + rng.normal(0, 220, 2), rng, 5, 0.3), 0.5) for k in rng.integers(0, len(main), 5)]
    return add(f, draw_bolts(segs, 7.0, amp=1.4 * (1 - ease(u / 0.35))))           # bolt burns out fast (no leftover)


_TUN = {}


def t_tunnel(t, A, B, a, mid, end):
    """Speed Force tunnel: blue-white and gold streaks rush past, the centre opens onto the first shot (the loop)."""
    if t >= end: return layer(B, t - DUR)
    cx, cy = W / 2, H * 0.44
    if 'th' not in _TUN:
        _TUN['th'] = np.arctan2(YY - cy, XX - cx); _TUN['r'] = np.sqrt((XX - cx) ** 2 + (YY - cy) ** 2) + 1
        r_ = np.random.default_rng(3); ang = np.zeros(1440, np.float32)
        for _ in range(170): ang[r_.integers(1440)] = r_.uniform(0.4, 1.0)
        ang = np.convolve(np.concatenate([ang, ang[:6]]), np.ones(3) / 1.5, 'same')[:1440]
        col = r_.uniform(0, 1, 1440) < 0.35
        _TUN['ang'] = ang; _TUN['gold'] = col.astype(np.float32)
    th, r = _TUN['th'], _TUN['r']; idx = ((th + np.pi) / (2 * np.pi) * 1440).astype(np.int32) % 1440
    s_ = _TUN['ang'][idx]; goldness = _TUN['gold'][idx]
    u = clamp((t - a) / (end - a)); env = ease(u / 0.35)
    dash = 0.5 + 0.5 * np.sin(np.log(r) * 9 - (t - a) * 55 + idx * 0.37)
    streak = s_ * ss(60, 360, r) * dash ** 3
    col = (BLUE_GLOW * (1 - goldness[..., None]) + GOLD_GLOW * goldness[..., None]) * 1.4 + 0.25
    fa = layer(A, t); acc = np.zeros_like(fa)
    for k in range(4): acc += zoom(fa, 1 + (0.06 + 0.5 * ein(u)) * (k + 1) / 4, cx=cx, cy=cy)
    f = acc / 4 * (1 - 0.55 * env)
    hole = 0 if t < mid else lerp(0, math.hypot(W, H), ein((t - mid) / (end - mid)))
    open_ = (1 - ss(hole * 0.6, hole, r))[..., None] if hole > 1 else 1.0
    f = f + (streak[..., None] * col) * env * open_
    if hole > 0:
        zb = lerp(4.0, 1.0, ease((t - mid) / (end - mid)))
        fb = zoom(layer(B, t - DUR), 1.0 / zb if zb < 1 else zb, cx=cx, cy=cy) if False else layer(B, t - DUR, 1 + 0.6 * (1 - ease((t - mid) / (end - mid))))
        inside = ss(hole + 6, hole - 6, r)[..., None]
        rim = np.exp(-((r - hole) / 14) ** 2)[..., None] * BLUE_GLOW * 1.6
        f = f * (1 - inside) + fb * inside + rim
    return f


_SPLIT = {}


def t_boltsplit(t, A, B, a, c, end):
    """Loop ending: a lightning-bolt seam flashes across the last shot, the two halves fly apart along it,
    and the first shot is behind them."""
    if t >= end: return layer(B, t - DUR)
    if 'side' not in _SPLIT:
        zz = np.array([(0.80, -0.05), (0.38, 0.40), (0.64, 0.46), (0.22, 1.05)]) * [W, H]
        line = []
        for p, q in zip(zz, zz[1:]): line += list(bolt_path(p, q, np.random.default_rng(int(p[0])), 5, 0.05))
        line = np.array(line); _SPLIT['line'] = line
        poly = np.vstack([line, [[W + 400, H + 400], [W + 400, -400]]])
        m = np.zeros((H, W), np.uint8); cv2.fillPoly(m, [np.round(poly * 4).astype(np.int32)], 255, cv2.LINE_AA, shift=2)
        _SPLIT['side'] = m.astype(np.float32) / 255
        d = np.full((H, W), 255, np.uint8); cv2.polylines(d, [np.round(line * 4).astype(np.int32)], False, 0, 3, cv2.LINE_8, shift=2)
        _SPLIT['dist'] = cv2.distanceTransform(d, cv2.DIST_L2, 5)
    side, dist, line = _SPLIT['side'], _SPLIT['dist'], _SPLIT['line']
    fa = layer(A, t)
    if t < c:                                                     # the seam draws itself down the frame and burns brighter
        u = clamp((t - a) / (c - a)); n = max(2, int(len(line) * eout(u / 0.6)))
        seam = np.exp(-(dist / 6) ** 2) * (1 if u > 0.6 else 0) * 0.9
        f = fa + seam[..., None] * GOLD_CORE + cv2.GaussianBlur(seam, (0, 0), 14)[..., None] * GOLD_GLOW * (0.8 + u)
        return add(f, draw_bolts([(line[:n], 1.0)], 4.5, amp=0.8 + 0.6 * u))
    u = clamp((t - c) / (end - c)); e = ein(u ** 0.9)
    off = e * W * 1.15; k = int(1 + 120 * math.sin(math.pi * min(1, u * 1.1))) | 1
    shiftL = np.float32([[1, 0, -off], [0, 1, -off * 0.35]]); shiftR = np.float32([[1, 0, off], [0, 1, off * 0.35]])
    L = cv2.warpAffine(fa * (1 - side[..., None]), shiftL, (W, H)); mL = cv2.warpAffine(1 - side, shiftL, (W, H))
    R = cv2.warpAffine(fa * side[..., None], shiftR, (W, H)); mR = cv2.warpAffine(side, shiftR, (W, H))
    if k > 1: L = cv2.blur(L, (k, 1)); R = cv2.blur(R, (k, 1)); mL = cv2.blur(mL, (k, 1)); mR = cv2.blur(mR, (k, 1))
    fb = layer(B, t - DUR, 1 + 0.18 * (1 - eout(u)))
    m = np.clip(mL + mR, 0, 1)[..., None]
    f = fb * (1 - m) + (L + R)
    edges = (np.abs(cv2.Sobel(mL, cv2.CV_32F, 1, 0, ksize=3)) + np.abs(cv2.Sobel(mR, cv2.CV_32F, 1, 0, ksize=3)))
    return f + np.clip(edges, 0, 1)[..., None] * GOLD_GLOW * 1.6 * (1 - u) + 0.7 * (1 - eout(u / 0.2))


# ------------------------------------------------------------ drawn Flash emblem (procedural, crisp at any scale)
_EBOLT = np.array([(0.66, -1.10), (-0.42, 0.17), (-0.16, 0.17), (-0.66, 1.10), (0.42, -0.17), (0.16, -0.17)])
_GOLD = (np.float32([0.04, 0.26, 0.50]), np.float32([0.12, 0.60, 0.92]), np.float32([0.62, 0.92, 1.00]))
_LIGHT = np.array([-0.55, -0.70, 0.45]); _LIGHT = _LIGHT / np.linalg.norm(_LIGHT)


def _gold(nx, ny, nz, spec_band=None):
    lam = np.clip(nx * _LIGHT[0] + ny * _LIGHT[1] + nz * _LIGHT[2], 0, 1)
    hz = _LIGHT + np.array([0, 0, 1.0]); hz = hz / np.linalg.norm(hz)
    spec = np.clip(nx * hz[0] + ny * hz[1] + nz * hz[2], 0, 1) ** 40
    d, m, l = _GOLD
    col = d + (m - d) * np.clip(lam * 1.3, 0, 1)[..., None] + (l - m) * np.clip(lam * 1.6 - 0.7, 0, 1)[..., None]
    col = col + spec[..., None] * 0.9
    if spec_band is not None: col = col + spec_band[..., None] * np.float32([0.7, 0.9, 1.0])
    return col


def draw_emblem(f, cx, cy, R, rot, ring_u=1.0, disc_u=1.0, bolt_u=1.0, glint=None, Pshift=(0.0, 0.0), window=None, alpha=1.0):
    """Composite the Flash emblem over f. ring_u: ring draw-on (0-1); disc_u: disc pop; bolt_u: bolt strike-in;
    glint: position of a specular sweep (-1.5..1.5) or None; window: an image shown inside the bolt (the dive)."""
    ca, sa = math.cos(rot), math.sin(rot)
    ext = 1.5 * R
    x0, x1 = int(max(0, cx - ext - 60)), int(min(W, cx + ext + 60)); y0, y1 = int(max(0, cy - ext - 60)), int(min(H, cy + ext + 60))
    if x1 <= x0 or y1 <= y0: return f
    X = XX[y0:y1, x0:x1] - cx; Y = YY[y0:y1, x0:x1] - cy
    lx = (X * ca + Y * sa) / R + Pshift[0]; ly = (-X * sa + Y * ca) / R + Pshift[1]
    r = np.sqrt(lx * lx + ly * ly) + 1e-6; e = 1.2 / R
    out = f.copy(); sub_ = out[y0:y1, x0:x1]
    def to_screen(pts):
        p = np.asarray(pts) - np.asarray(Pshift)
        return np.stack([cx + R * (p[:, 0] * ca - p[:, 1] * sa) - x0, cy + R * (p[:, 0] * sa + p[:, 1] * ca) - y0], 1)
    # bolt (slides in along its own axis from the top-right)
    off = (1 - bolt_u) * 2.6; bdir = np.array([0.55, -0.84])
    bpoly = to_screen(_EBOLT + bdir * off)
    bm = np.zeros(sub_.shape[:2], np.uint8)
    cv2.fillPoly(bm, [np.clip(np.round(bpoly * 4), -2 ** 30, 2 ** 30).astype(np.int32)], 255, cv2.LINE_AA, shift=2)
    bmf = bm.astype(np.float32) / 255 * (bolt_u > 0)
    # ring with draw-on
    th = (np.arctan2(lx, -ly) / (2 * np.pi)) % 1.0                     # 0 at top, clockwise
    ring = ss(1.0 + e, 1.0 - e, r) * ss(0.84 - e, 0.84 + e, r) * ss(ring_u + 0.004, ring_u - 0.004, th) * (ring_u > 0)
    sc = np.clip((r - 0.84) / 0.16, 0, 1); slope = -np.cos(np.pi * sc) * 0.85
    nx, ny = slope * lx / r, slope * ly / r; nz = np.sqrt(np.clip(1 - nx * nx - ny * ny, 0.05, 1))
    band = None if glint is None else np.exp(-((lx + ly * 0.8 - glint) / 0.07) ** 2) * 0.9
    ring_col = _gold(nx, ny, nz, band)
    # disc (pearl, recessed under the ring)
    dr = 0.84 * (0.0 if disc_u <= 0 else (1 + 0.10 * math.sin(math.pi * min(1, disc_u)) * (disc_u < 1)) * min(1, eout(disc_u) * 1.0))
    disc = ss(dr + e, dr - e, r) * (disc_u > 0)
    pearl = np.float32([0.86, 0.90, 0.93]) * (0.82 + 0.18 * ss(1.0, -0.6, lx + ly))[..., None]
    pearl = pearl * (1 - 0.40 * ss(0.62, 0.84, r))[..., None]                 # inner shadow under the rim
    if band is not None: pearl = pearl + band[..., None] * 0.35
    # drop shadow + soft outer glow
    shape = np.clip(np.maximum(np.maximum(ring, disc), bmf), 0, 1)
    shd = cv2.GaussianBlur(np.roll(np.roll(shape, int(min(0.07 * R, 60)), 0), int(min(0.05 * R, 45)), 1), (0, 0), float(np.clip(0.06 * R, 2, 40))) * 0.55 * alpha if R < 2500 else np.zeros_like(shape)
    sub_ *= (1 - shd[..., None])
    sub_[:] = sub_ * (1 - disc[..., None] * alpha) + pearl * disc[..., None] * alpha
    sub_[:] = sub_ * (1 - ring[..., None] * alpha) + ring_col * ring[..., None] * alpha
    # bolt: bevelled gold, or a window with a gold rim during the dive
    if bmf.max() > 0:
        dist = cv2.distanceTransform(cv2.copyMakeBorder(bm, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0), cv2.DIST_L2, 5)[1:-1, 1:-1]; bw = max(2.0, 0.07 * R)
        gx = cv2.Sobel(dist, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(dist, cv2.CV_32F, 0, 1, ksize=3); gn = np.sqrt(gx * gx + gy * gy) + 1e-6
        s_ = np.clip(dist / bw, 0, 1); k = (1 - s_) * 0.85
        bnx, bny = -gx / gn * k, -gy / gn * k
        bnx, bny = bnx * ca + bny * sa, -bnx * sa + bny * ca          # into emblem-local orientation for consistent lighting
        bnz = np.sqrt(np.clip(1 - bnx * bnx - bny * bny, 0.05, 1))
        bcol = _gold(bnx, bny, bnz, band) * 1.05
        outline = np.clip(cv2.dilate(bmf, np.ones((5, 5), np.uint8)) - bmf, 0, 1) * disc     # dark seam where the bolt sits on the disc
        sub_[:] = sub_ * (1 - outline[..., None] * 0.5 * alpha)
        if window is not None:
            rimw = max(2.0, min(bw, 14.0)); rimm = (dist < rimw).astype(np.float32) * bmf
            win = window[y0:y1, x0:x1]
            sub_[:] = sub_ * (1 - bmf[..., None]) + (win * (1 - rimm[..., None]) + bcol * rimm[..., None]) * bmf[..., None]
        else:
            sub_[:] = sub_ * (1 - bmf[..., None] * alpha) + bcol * bmf[..., None] * alpha
    # spark at the ring's drawing head
    if 0 < ring_u < 1:
        a_ = ring_u * 2 * np.pi; hx, hy = 0.92 * math.sin(a_), -0.92 * math.cos(a_)
        p = to_screen([(hx, hy)])[0]; d2 = (XX[y0:y1, x0:x1] - x0 - p[0]) ** 2 + (YY[y0:y1, x0:x1] - y0 - p[1]) ** 2
        sub_[:] += (np.exp(-d2 / (2 * (0.05 * R) ** 2)) * 1.6)[..., None] * GOLD_CORE + (np.exp(-d2 / (2 * (0.18 * R) ** 2)) * 0.7)[..., None] * GOLD_GLOW
    return out


def t_emblem(t, c, A, B, a, b):
    """Drawn emblem: the gold ring draws itself on, the pearl disc pops in, the bevelled bolt strikes down into place,
    a glint crosses the metal, then the camera dives through the bolt into the next shot."""
    cx, cy, R0 = W / 2, H * 0.44, 0.36 * W
    if t < c:
        p = clamp((t - a) / (c - a))
        fa = layer(A, t, 1 + 0.06 * p) * (1 - 0.35 * ease(p / 0.3))
        ring_u = eout(clamp(p / 0.40)); disc_u = clamp((p - 0.18) / 0.25); bolt_u = ein(clamp((p - 0.40) / 0.20))
        R = R0 * (1 + 0.05 * ease(clamp((p - 0.6) / 0.4))); rot = math.radians(lerp(-14, 0, eout(p)))
        glint = lerp(-1.8, 1.8, ease(clamp((p - 0.62) / 0.36))) if p > 0.62 else None
        f = draw_emblem(fa, cx, cy, R, rot, ring_u, disc_u, bolt_u, glint)
        land = clamp((p - 0.60) / 0.12)
        if 0 < land < 1:                                              # the strike lands: flash + a quick ring of arcs
            f = f + 0.55 * (1 - land)
            rng = np.random.default_rng(int(t * FPS)); segs = []
            for _ in range(6):
                ang = rng.uniform(0, 2 * np.pi); p0 = (cx + R * math.cos(ang), cy + R * math.sin(ang))
                segs.append((bolt_path(p0, (cx + R * 1.5 * math.cos(ang + rng.uniform(-.3, .3)), cy + R * 1.5 * math.sin(ang + rng.uniform(-.3, .3))), rng, 5, 0.12), 0.7))
            f = add(f, draw_bolts(segs, 3.5, amp=1.2 * (1 - land)))
        return f + 0.6 * ein(clamp((t - (c - 0.04)) / 0.04))
    u = clamp((t - c) / (b - c)); e = ein(u)
    P = np.array([0.0, 0.0])                                      # dive into the bolt's wide centre
    R = R0 * 1.05 * math.exp(math.log(30.0) * e); rot = math.radians(lerp(0, -62, ease(u)))   # turn its long axis upright
    fa = layer(A, c - 1e-3, 1.06) * 0.65
    fb = layer(B, t, 1 + 0.2 * (1 - eout(u)))
    f = draw_emblem(fa, cx, cy, R, rot, 1.0, 1.0, 1.0, None, Pshift=P, window=fb)
    k = ease(clamp((u - 0.78) / 0.22))                             # settle fully onto the next shot
    return f * (1 - k) + fb * k + 0.6 * (1 - eout(u / 0.15))


# ------------------------------------------------------------ Speed Force shockwave ripples on hits
def ripple_fx(f, t, s, cx=0.5, cy=0.42):
    hits = [tk for tk in _hits('SNARES') + _hits('KICKS') if s['start'] - 1e-6 <= tk <= t and t - tk < 0.45]
    if not hits: return f
    u = (t - s['start']) / (s['end'] - s['start']); gain = 0.5 + 0.7 * u
    X = XX - cx * W; Y = YY - cy * H; r = np.sqrt(X * X + Y * Y) + 1e-6
    disp = np.zeros((H, W), np.float32); edge = np.zeros((H, W), np.float32)
    for tk in hits:
        rr = (t - tk) * 2600; dec = math.exp(-(t - tk) / 0.22) * gain
        g = np.exp(-((r - rr) / 55) ** 2) * dec
        disp += 22 * g * np.sin((r - rr) / 18); edge += np.exp(-((r - rr) / 6) ** 2) * dec
    g2 = cv2.remap(f, XX + disp * X / r, YY + disp * Y / r, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return g2 + edge[..., None] * BLUE_GLOW * 0.35


# ============================================================ Dexter kit II (The Less I Know The Better)
# ---- overlays
_DUST = {}


def dust_fx(f, i, t, s):
    """Floating dust motes catching the light: three depth layers, visible where the frame is bright."""
    if 'p' not in _DUST:
        r = np.random.default_rng(404)
        _DUST['p'] = [dict(n=n, x=r.uniform(0, W, n), y=r.uniform(0, H, n), vx=r.uniform(-14, 30, n), vy=r.uniform(-34, -6, n),
                           ph=r.uniform(0, 6.28, n), sz=r.uniform(a, b, n), blur=bl, gain=gn)
                      for n, a, b, bl, gn in ((70, 1.2, 2.6, 0.6, 0.55), (26, 3.0, 5.0, 2.2, 0.35), (7, 9, 16, 9.0, 0.16))]
    q = 2; lum = cv2.resize(luma(f), (W // 16, H // 16), interpolation=cv2.INTER_AREA)
    lum = cv2.resize(cv2.GaussianBlur(lum, (0, 0), 2), (W // q, H // q))
    acc = np.zeros((H // q, W // q), np.float32)
    for L in _DUST['p']:
        can = np.zeros((H // q, W // q), np.float32)
        x = (L['x'] + L['vx'] * t + 18 * np.sin(L['ph'] + t * 1.3)) % W; y = (L['y'] + L['vy'] * t + 10 * np.cos(L['ph'] + t)) % H
        for xx, yy, sz in zip(x, y, L['sz']):
            cv2.circle(can, (int(xx / q), int(yy / q)), max(1, int(sz / q)), 1.0, -1, cv2.LINE_AA)
        acc += cv2.GaussianBlur(can, (0, 0), L['blur'] / q + 0.3) * L['gain']
    acc = acc * (0.25 + 1.2 * np.clip(lum - 0.15, 0, 1))
    return f + cv2.resize(acc, (W, H))[..., None] * np.float32([0.92, 0.96, 1.0])


def glint_fx(f, t, s):
    """Four-point star glints on the brightest specular spots (catchlights, lenses, metal), twinkling on the beat."""
    q = 4; l = cv2.GaussianBlur(cv2.resize(luma(f), (W // q, H // q), interpolation=cv2.INTER_AREA), (0, 0), 1.0)
    mx = cv2.dilate(l, np.ones((31, 31), np.uint8))
    ys, xs = np.nonzero((l >= mx) & (l > 0.80))
    if len(xs) == 0: return f
    order = np.argsort(-l[ys, xs])[:2]; out = f
    pulse = 0.0
    for tk in _hits('KICKS') + _hits('SNARES'):
        if s['start'] - 1e-6 <= tk <= t: pulse = max(pulse, math.exp(-(t - tk) / 0.12))
    for n, k in enumerate(order):
        x, y = xs[k] * q + q / 2, ys[k] * q + q / 2; strength = (l[ys[k], xs[k]] - 0.78) * 4
        tw = 0.55 + 0.45 * math.sin(t * 7.3 + n * 2.1); L = 110 * (0.6 + 0.6 * pulse) * tw * min(1.0, strength)
        x0, x1, y0, y1 = int(max(0, x - L - 20)), int(min(W, x + L + 20)), int(max(0, y - L - 20)), int(min(H, y + L + 20))
        X = XX[y0:y1, x0:x1] - x; Y = YY[y0:y1, x0:x1] - y
        star = (np.exp(-(X / L) ** 2 - (Y / 1.6) ** 2) + np.exp(-(Y / L) ** 2 - (X / 1.6) ** 2)
                + 0.35 * np.exp(-((X + Y) / (0.45 * L)) ** 2 - ((X - Y) / 1.6) ** 2) + 0.35 * np.exp(-((X - Y) / (0.45 * L)) ** 2 - ((X + Y) / 1.6) ** 2)
                + 0.9 * np.exp(-(X * X + Y * Y) / 30.0))
        out = out.copy() if out is f else out
        out[y0:y1, x0:x1] += (star * (0.75 + 0.6 * pulse))[..., None] * np.float32([0.95, 0.98, 1.0])
    return out


def ghost_fx(f, i, t, s, col=(0.98, 0.97, 1.0)):
    """Glowing outline double: the frame's edges, drawn as white light, drifting off the subject (voidfx-style)."""
    g = cv2.GaussianBlur(cv2.resize(luma(f), (W // 2, H // 2)), (0, 0), 1.4)
    e = cv2.Canny((np.clip(g, 0, 1) * 255).astype(np.uint8), 30, 90).astype(np.float32) / 255
    e = cv2.resize(cv2.GaussianBlur(e, (0, 0), 0.7), (W, H))
    pulse = 0.0
    for tk in _hits('SNARES') + _hits('KICKS'):
        if s['start'] - 1e-6 <= tk <= t: pulse = max(pulse, math.exp(-(t - tk) / 0.16))
    u = (t - s['start']) / max(1e-3, s['end'] - s['start'])
    out = f
    for sgn, a0 in ((1, 0.55), (-1, 0.30)):
        z = 1.04 + 0.05 * u + 0.03 * pulse; dx = sgn * (16 + 30 * u) ; dy = -10 * sgn
        M = cv2.getRotationMatrix2D((W / 2, H * 0.42), sgn * 1.5 * u, z); M[0, 2] += dx; M[1, 2] += dy
        ed = cv2.warpAffine(e, M, (W, H))
        glow = cv2.GaussianBlur(ed, (0, 0), 7)
        out = out + ((ed * 0.9 + glow * 1.4) * a0 * (0.45 + 0.8 * pulse))[..., None] * np.float32(col)
    return out


def trails_fx(f, i, t, s):
    """Psychedelic trails: earlier frames echo behind, screen-blended in cool white."""
    out = f
    for dt, al in ((0.06, 0.45), (0.13, 0.30), (0.22, 0.18)):
        g = shot(i, t - dt) * np.float32([1.0, 0.95, 0.9])
        out = 1 - (1 - out) * (1 - np.clip(g * al, 0, 1))
    return out


# ---- transitions
_INK = {}


def _ink_field(name, seeds, seed, noise=0.55):
    if name not in _INK:
        h, w = H // 4, W // 4; n = fbm(h, w, 5, seed, 6)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32); F = np.full((h, w), 99.0, np.float32)
        for sx, sy, r in seeds: F = np.minimum(F, np.sqrt((xx - sx * w) ** 2 + (yy - sy * h) ** 2) / (r * w))
        F = F + noise * (n - 0.5)
        Fu = cv2.resize(F, (W, H), interpolation=cv2.INTER_CUBIC)
        gy, gx = np.gradient(cv2.GaussianBlur(Fu, (0, 0), 3))
        gn = np.sqrt(gx * gx + gy * gy) + 1e-6
        _INK[name] = (Fu, float(Fu.max()) + 0.05, (-gx / gn * -0.6 + -gy / gn * -0.8))      # facing the top-left light
    return _INK[name]


def _ink_paint(f, Fu, g, facing, inside_is_ink=True):
    e = 0.006
    inside = ss(g + e, g - e, Fu)
    rim = np.exp(-((Fu - g) / 0.010) ** 2) * np.clip(facing, 0, 1)
    ink = np.float32([0.012, 0.012, 0.016]) + (0.05 * fbm(H // 8, W // 8, 3, 77, 4))[..., None].repeat(3, 2).repeat(8, 0).repeat(8, 1)[:H, :W] * 0.4
    m = inside[..., None] if inside_is_ink else (1 - inside)[..., None]
    out = f * (1 - m) + ink * m
    return out + (rim * 0.42)[..., None] * np.float32([0.85, 0.88, 0.95])


def t_ink(t, c, A, B, a, b):
    """Black ink blooms in from the edges with droplets and a wet rim until the frame is ink,
    then the ink parts from the centre to reveal the next shot."""
    if t < c:
        seeds = [(0.02, 0.10, 0.55), (0.98, 0.60, 0.60), (0.35, 1.02, 0.50), (0.12, 0.55, 0.05), (0.85, 0.22, 0.04), (0.70, 0.93, 0.035), (0.25, 0.30, 0.03)]
        Fu, Fmax, facing = _ink_field('in', seeds, 31)
        p = clamp((t - a) / (c - a)); g = Fmax * (0.12 * p + 0.88 * ein(p) ** 0.8)
        return _ink_paint(layer(A, t), Fu, g, facing, True)
    Fu, Fmax, facing = _ink_field('out', [(0.5, 0.44, 0.75), (0.30, 0.62, 0.10), (0.70, 0.30, 0.08)], 32, 0.5)
    u = clamp((t - c) / (b - c)); h = Fmax * eout(u) ** 1.1
    fb = layer(B, t, 1 + 0.08 * (1 - eout(u)))
    return _ink_paint(fb, Fu, h, -facing, False)


def t_slices(t, c, A, B, a, b, n=6):
    """Knife cuts slash down the frame, then the strips fall away up and down, revealing the next shot."""
    r = np.random.default_rng(7); xs = np.linspace(0, 1, n + 1); xs[1:-1] += r.uniform(-0.04, 0.04, n - 1)
    slant = math.tan(math.radians(7)); bx = lambda x, y: x * W + slant * (y - H / 2)
    if t < c:
        f = layer(A, t, 1 + 0.03 * clamp((t - a) / (c - a)))
        can = np.zeros((H, W), np.uint8)
        for k in range(1, n):
            pk = eout(clamp((t - a - 0.018 * k) / 0.09))
            if pk <= 0: continue
            y1 = pk * H
            cv2.line(can, (int(round(bx(xs[k], 0) * 4)), 0), (int(round(bx(xs[k], y1) * 4)), int(y1 * 4)), 255, 2, cv2.LINE_AA, shift=2)
        cf = can.astype(np.float32) / 255
        return f + cf[..., None] * np.float32([1.0, 1.0, 1.0]) + cv2.GaussianBlur(cf, (0, 0), 6)[..., None] * np.float32([0.25, 0.25, 1.4])
    u = t - c; fa = layer(A, c - 1e-3); fb = layer(B, t, 1 + 0.10 * (1 - eout(u / 0.3))) * (0.75 + 0.25 * eout(u / 0.25))
    out = fb.copy(); edges = np.zeros((H, W), np.float32)
    for k in range(n):
        dl = 0.022 * k; v = ein(clamp((u - dl) / 0.24)); d = -1 if k % 2 == 0 else 1
        off = d * v * H * 1.15
        poly = np.array([[bx(xs[k], 0), 0], [bx(xs[k + 1], 0), 0], [bx(xs[k + 1], H), H], [bx(xs[k], H), H]], np.float64)
        m = np.zeros((H, W), np.uint8); cv2.fillPoly(m, [np.round((poly + [0, off]) * 4).astype(np.int32)], 255, cv2.LINE_AA, shift=2)
        mf = m.astype(np.float32) / 255
        if mf.max() == 0: continue
        sh = cv2.warpAffine(fa, np.float32([[1, 0, 0], [0, 1, off]]), (W, H), borderMode=cv2.BORDER_REFLECT)
        kb = int(1 + 140 * (ein(clamp((u - dl + 1 / FPS) / 0.24)) - v) * 12) | 1
        if kb > 1: sh = cv2.blur(sh, (1, min(kb, 301)))
        out = out * (1 - mf[..., None]) + sh * mf[..., None]
        edges += np.clip(cv2.morphologyEx(mf, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)), 0, 1)
    out = out + (edges * (1 - ease(u / 0.3)))[..., None] * np.float32([0.35, 0.35, 1.3])
    return out + 0.5 * (1 - eout(u / 0.08))


# blood orange
_ORG = {}


def _orange_tex():
    if 'tex' not in _ORG:
        A_, R_ = 2048, 512; r = np.random.default_rng(55)
        def streaks(ny, nx, seed):
            g = np.random.default_rng(seed).random((ny, nx)).astype(np.float32)
            g = np.concatenate([g, g[:, :8]], 1)
            return cv2.resize(g, (A_ + int(8 * A_ / nx), R_), interpolation=cv2.INTER_CUBIC)[:, :A_]
        v = 0.6 * streaks(24, 900, 1) + 0.4 * streaks(60, 1600, 2)              # radially elongated juice cells
        v = (v - v.min()) / (v.max() - v.min())
        seg = 10; segmix = r.uniform(0.15, 0.95, seg)
        ang = (np.arange(A_) / A_ * seg).astype(int) % seg
        mix = segmix[ang][None, :] * np.ones((R_, 1), np.float32)
        mix = np.clip(mix + 0.35 * (streaks(8, 120, 3) - 0.5), 0, 1)
        _ORG['tex'] = (v.astype(np.float32), mix.astype(np.float32), seg)
    return _ORG['tex']


def draw_orange(f, cx, cy, R, rot, sx=1.0, sy=1.0, window=None, open_u=None):
    """Blood orange cross-section: rind with pores, pith, 10 segments of crimson-to-orange juice cells with
    white membranes, wet highlights. window/open_u: segments open (staggered) onto `window`."""
    v_tex, mix_tex, seg = _orange_tex(); A_, R_ = v_tex.shape[1], v_tex.shape[0]
    ext = 1.3 * R * max(sx, sy)
    x0, x1, y0, y1 = int(max(0, cx - ext)), int(min(W, cx + ext)), int(max(0, cy - ext)), int(min(H, cy + ext))
    if x1 <= x0 or y1 <= y0: return f
    X = (XX[y0:y1, x0:x1] - cx) / (R * sx); Y = (YY[y0:y1, x0:x1] - cy) / (R * sy)
    r = np.sqrt(X * X + Y * Y) + 1e-6; th = (np.arctan2(Y, X) - rot) % (2 * np.pi)
    e = 1.4 / R
    mapx = (th / (2 * np.pi) * A_).astype(np.float32); mapy = np.clip(r / 0.86, 0, 0.999).astype(np.float32) * (R_ - 1)
    v = cv2.remap(v_tex, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    mx = cv2.remap(mix_tex, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    crimson = np.float32([0.04, 0.03, 0.46]); blood = np.float32([0.07, 0.06, 0.72]); orng = np.float32([0.10, 0.36, 0.95])
    flesh = crimson * (1 - v[..., None]) + blood * v[..., None]
    flesh = flesh * (1 - mx[..., None] * 0.6) + orng * (mx[..., None] * 0.6) * (0.75 + 0.5 * v[..., None])
    flesh = flesh * (1 - 0.35 * ss(0.70, 0.86, r))[..., None]                    # darker toward the pith
    flesh += (ss(0.82, 0.95, v) * ss(0.2, 0.8, r) * 0.55)[..., None]              # wet sparkle on juice cells
    sa = (th * seg / (2 * np.pi)) % 1.0; dang = np.minimum(sa, 1 - sa) * (2 * np.pi / seg) * r
    memb = ss(0.016 + e, 0.010, dang) * ss(0.05, 0.10, r)
    pith_c = np.float32([0.80, 0.88, 0.95])
    out = f.copy(); sub = out[y0:y1, x0:x1]
    disc = ss(1.0 + e, 1.0 - e, r)
    shd = cv2.GaussianBlur(np.roll(np.roll(disc, int(min(0.06 * R, 60)), 0), int(min(0.04 * R, 40)), 1), (0, 0), float(np.clip(0.05 * R, 2, 40))) * 0.5 if R < 2500 else 0
    sub *= (1 - (shd[..., None] if R < 2500 else 0))
    rind = ss(0.93 - e, 0.93 + e, r) * disc
    pores = ss(0.78, 0.9, cv2.remap(cv2.resize(v_tex, (1024, 64)), mapx / 2, mapy / 8, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP))
    rind_c = np.float32([0.05, 0.27, 0.80]) * (0.85 - 0.25 * pores)[..., None] + (0.35 * ss(1.0, -1.0, X + Y) * ss(0.95, 0.99, r))[..., None]
    pith = ss(0.86 - e, 0.86 + e, r) * (1 - rind) * disc
    core = ss(0.06 + e, 0.06 - e, r)
    fl = disc * (1 - rind) * (1 - pith) * (1 - core) * (1 - memb)
    if window is not None and open_u is not None:                                  # segments open one after another
        sidx = np.floor(th * seg / (2 * np.pi)); op = np.clip((open_u - sidx * 0.06) / 0.25, 0, 1)
        win = window[y0:y1, x0:x1]; opn = (op * fl)[..., None]
        sub[:] = sub * (1 - fl[..., None]) + flesh * (fl[..., None] - opn) + win * opn
    else:
        sub[:] = sub * (1 - fl[..., None]) + flesh * fl[..., None]
    mm = disc * memb * (1 - rind) * (1 - pith)
    sub[:] = sub * (1 - mm[..., None]) + pith_c * mm[..., None]
    sub[:] = sub * (1 - (pith + core * disc)[..., None]) + pith_c * (0.9 + 0.1 * v)[..., None] * (pith + core * disc)[..., None]
    sub[:] = sub * (1 - rind[..., None]) + rind_c * rind[..., None]
    sub += (np.exp(-((X + 0.35) ** 2 + (Y + 0.40) ** 2) / 0.05) * 0.18 * disc)[..., None]     # broad wet gloss
    return out


def t_orange(t, c, A, B, a, b):
    """A blood orange slice (Dexter's breakfast) spins in and lands with a burst of juice; its segments open
    onto the next shot and the camera dives into the fruit."""
    cx, cy, R0 = W / 2, H * 0.45, 0.40 * W
    if t < c:
        p = clamp((t - a) / (c - a)); land = 0.58
        if p < land:
            u = eout(p / land); x = lerp(W * 1.45, cx, u); y = lerp(H * 0.95, cy, u); rot = lerp(-3.6, 0.0, u); R = R0 * lerp(0.55, 1.0, u); sx = sy = 1.0
        else:
            k = (p - land) / (1 - land); x, y, rot, R = cx, cy, 0.0, R0 * (1 + 0.04 * k)
            sq = math.exp(-k * 5) * math.cos(k * 14); sx, sy = 1 + 0.08 * sq, 1 - 0.07 * sq
        f = layer(A, t) * (1 - 0.25 * ease(p / 0.5))
        f = draw_orange(f, x, y, R, rot, sx, sy)
        if p > land:                                                                # juice spray
            k = (p - land) * (c - a); rng = np.random.default_rng(12); can = np.zeros((H, W, 3), np.float32); al = np.zeros((H, W), np.float32)
            for _ in range(26):
                ang = rng.uniform(0, 2 * np.pi); sp = rng.uniform(900, 2400); rr = rng.uniform(5, 16)
                px = cx + math.cos(ang) * (R * 0.95 + sp * k); py = cy + math.sin(ang) * (R * 0.95 + sp * k) + 2600 * k * k
                cv2.circle(al, (int(px), int(py)), int(rr), 1.0, -1, cv2.LINE_AA)
                cv2.circle(can, (int(px - rr * .3), int(py - rr * .3)), max(1, int(rr * .3)), (0.9, 0.9, 1.0), -1, cv2.LINE_AA)
            al = al * clamp(1 - k / 0.3); f = f * (1 - al[..., None]) + np.float32([0.04, 0.03, 0.55]) * al[..., None] + can * al[..., None]
        return f
    u = clamp((t - c) / (b - c)); R = R0 * 1.04 * math.exp(math.log(14.0) * ein(u)); rot = lerp(0, 0.5, ease(u))
    fb = layer(B, t, 1 + 0.25 * (1 - eout(u)))
    f = draw_orange(layer(A, c - 1e-3) * 0.75, cx, cy, R, rot, window=fb, open_u=u * 2.2)
    k = ease(clamp((u - 0.75) / 0.25))
    return f * (1 - k) + fb * k + 0.4 * (1 - eout(u / 0.12))


# M99 syringe, blackout and waking blink
def draw_syringe(f, tip, ang, plunge, alpha=1.0):
    """A glass syringe pointing at `tip` from direction `ang` (radians, toward the plunger), plunger at `plunge` (0 full..1 empty)."""
    eu = np.array([math.cos(ang), math.sin(ang)]); ev = np.array([-eu[1], eu[0]])
    X = XX - tip[0]; Y = YY - tip[1]; u = X * eu[0] + Y * eu[1]; v = X * ev[0] + Y * ev[1]; av = np.abs(v)
    Ln, Lh, Lb, Wb = 330.0, 42.0, 640.0, 62.0
    ub0 = Ln + Lh; ub1 = ub0 + Lb; us = lerp(ub1 - 70, ub0 + 50, plunge)
    def box(u0, u1, hw): return ss(-0.7, 0.7, u - u0) * ss(0.7, -0.7, u - u1) * ss(0.7, -0.7, av - hw)
    out = f.copy()
    shape = np.clip(box(0, ub1 + 22, Wb + 4) + box(ub1, ub1 + 22, 130) + box(us, us + 620, 16), 0, 1)
    shd = cv2.GaussianBlur(np.roll(np.roll(shape, 26, 0), 18, 1), (0, 0), 12) * 0.35 * alpha
    out *= (1 - shd[..., None])
    def lay(m, col):
        nonlocal out
        m = (m * alpha)[..., None]; out = out * (1 - m) + col * m
    # plunger rod + thumb disc + rubber stopper
    rod = box(us + 28, us + 600, 15); lay(rod, (0.80 + 0.12 * (1 - av / 15))[..., None] * np.float32([0.95, 0.96, 0.97]))
    lay(box(us + 600, us + 620, 78), np.float32([0.88, 0.89, 0.90]) - 0.12 * ss(40, 78, av)[..., None])
    stop = box(us, us + 28, Wb - 6)
    lay(stop, np.float32([0.05, 0.05, 0.06]) + (0.10 * (np.exp(-((u - us - 7) / 2.5) ** 2) + np.exp(-((u - us - 20) / 2.5) ** 2)))[..., None])
    # liquid (amber M99) between the hub and the stopper, then the glass barrel over everything
    liq = box(ub0 + 8, us, Wb - 7)
    lay(liq * 0.62, np.float32([0.22, 0.58, 0.88]) + (0.25 * np.exp(-((u - us + 5) / 3) ** 2))[..., None])
    glass = box(ub0, ub1, Wb)
    walls = glass * ss(Wb - 9, Wb - 3, av)
    hi = glass * (0.55 * np.exp(-((v + Wb * 0.55) / 7) ** 2) + 0.35 * np.exp(-((v - Wb * 0.72) / 2.5) ** 2))
    out = out * (1 - (glass * 0.10 * alpha)[..., None]) + (glass * 0.10 * alpha)[..., None] * np.float32([0.95, 0.97, 0.98])
    lay(walls * 0.55, np.float32([0.85, 0.88, 0.90]))
    out = out + ((hi + walls * 0.15) * alpha)[..., None]
    ticks = glass * ss(-0.5, 0.5, -(v + Wb * 0.25)) * ss(-Wb - 1, -Wb + 3, v)
    tk = np.abs(((u - ub0 - 40) % 42) - 21) > 19.6; tick_long = (((u - ub0 - 40) // 42) % 5 == 0)
    lay(ticks * tk * np.where(tick_long, 1.0, ss(-Wb * 0.62, -Wb * 0.58, v)) * (u > ub0 + 30) * (u < ub1 - 30), np.float32([0.10, 0.10, 0.12]))
    lay(box(ub1, ub1 + 22, 130), np.float32([0.90, 0.91, 0.92]) - 0.15 * ss(100, 130, av)[..., None])     # finger flange
    hub = ss(-0.7, 0.7, u - Ln) * ss(0.7, -0.7, u - ub0) * ss(0.7, -0.7, av - (10 + 14 * np.clip((u - Ln) / Lh, 0, 1)))
    lay(hub * 0.9, np.float32([0.86, 0.90, 0.92]))
    wn = 3.6 * np.clip(u / 26, 0.05, 1); needle = ss(-0.6, 0.6, u) * ss(0.6, -0.6, u - Ln) * ss(0.6, -0.6, av - wn)
    lay(needle, (0.55 + 0.4 * np.exp(-((v + 1.0) / 1.2) ** 2))[..., None] * np.float32([0.95, 0.96, 1.0]))
    return out


def _woozy(f, k, t):
    if k <= 0.01: return f
    r = np.sqrt((XX - W / 2) ** 2 + (YY - H * 0.45) ** 2)
    sw = k * 0.22 * np.exp(-(r / (0.7 * H)) ** 2) * math.sin(t * 4.0)
    ca, sa = np.cos(sw), np.sin(sw)
    X = XX - W / 2; Y = YY - H * 0.45
    g = cv2.remap(f, X * ca - Y * sa + W / 2, X * sa + Y * ca + H * 0.45, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    dx = int(36 * k * (0.6 + 0.4 * math.sin(t * 9)))
    g = 0.5 * (np.roll(g, dx, 1) + np.roll(g, -dx, 1))
    g = blur(g, 9 * k)
    return g * (1 - 0.55 * k * np.clip(r / (0.75 * H), 0, 1.4)[..., None] ** 2)


def _lids(f, o, inside_glow=0.0):
    """Eyelids: o = openness 0..1. Closed lids are dark, with blood-red light glowing through near the edge."""
    if o >= 0.999: return f
    xn = (XX - W / 2) / (W * 0.62); yc = H * 0.45
    half = o * H * 0.60 * np.clip(1 - xn * xn, 0, 1) ** 0.55
    d = np.abs(YY - yc) - half
    lid = ss(-26, 26, d)
    col = np.float32([0.008, 0.008, 0.03]) + (np.exp(-np.clip(d, 0, None) / 90.0) * (0.10 + inside_glow))[..., None] * np.float32([0.10, 0.10, 0.85])
    return f * (1 - lid[..., None]) + col * lid[..., None]


def t_syringe(t, c, A, B, a, b):
    """The M99 syringe is pushed at the viewer, the plunger goes down, the world swims and the eyes close;
    then the eyes blink open on the next shot."""
    T = (W * 0.50, H * 0.43); ang = math.radians(58)
    if t < c:
        p = (t - a) / (c - a)                                        # 0..1 over the pre-cut part
        D = 1500 * (1 - eout(clamp(p / 0.38)))
        tip = (T[0] + math.cos(ang) * D, T[1] + math.sin(ang) * D)
        plunge = ease(clamp((p - 0.42) / 0.40))
        f = layer(A, t)
        hit = clamp((p - 0.38) / 0.4)
        if 0 < hit < 1:                                              # the needle touches the lens: a ripple
            r = np.sqrt((XX - T[0]) ** 2 + (YY - T[1]) ** 2); rr = hit * 900; amp = 14 * (1 - hit)
            dd = amp * np.exp(-((r - rr) / 60) ** 2) * np.sin((r - rr) / 14)
            f = cv2.remap(f, XX + dd * (XX - T[0]) / (r + 1), YY + dd * (YY - T[1]) / (r + 1), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        f = draw_syringe(f, tip, ang, plunge)
        f = _woozy(f, ease(clamp((p - 0.55) / 0.45)), t)
        return _lids(f, 1 - ease(clamp((p - 0.84) / 0.16)))
    u = t - c
    o = 0.0
    if u < 0.10: o = 0.35 * eout(u / 0.10)
    elif u < 0.17: o = lerp(0.35, 0.08, ease((u - 0.10) / 0.07))
    else: o = lerp(0.08, 1.0, eout((u - 0.17) / (b - c - 0.17)))
    f = _woozy(layer(B, t, 1 + 0.06 * (1 - o)), (1 - o) * 0.8, t)
    return _lids(f, o)


# ============================================================ Blue-sky kit (The Flash x Mr. Blue Sky)
_SUNK = {}


def _hexagon(cx, cy, r, rot=0.3):
    return np.array([[cx + r * math.cos(rot + k * math.pi / 3), cy + r * math.sin(rot + k * math.pi / 3)] for k in range(6)])


def draw_sun(f, sx, sy, R, rot, rays, flare=1.0):
    """Drawn sun: white-hot disc, warm corona, 16 tapered rotating rays (two lengths), an anamorphic streak
    and hexagonal lens ghosts on the line through the frame centre. Additive light."""
    q = 2; h, w = H // q, W // q
    X = (XX[::q, ::q] - sx) / R; Y = (YY[::q, ::q] - sy) / R; r = np.sqrt(X * X + Y * Y) + 1e-6; th = np.arctan2(Y, X)
    disc = ss(1.02, 0.96, r); corona = np.exp(-(np.clip(r - 1, 0, None) / 1.4) ** 1.2)
    n = 16; ang = (th - rot) % (2 * np.pi / n) - math.pi / n                 # angle to the nearest ray axis
    k = np.floor(((th - rot) % (2 * np.pi)) / (2 * np.pi / n)).astype(int)
    length = np.where(k % 2 == 0, 7.5, 4.5) * rays
    width = 0.20 * np.clip(1 - (r - 1) / np.maximum(length, 1e-3), 0, 1)     # tapered toward the tip
    ray = ss(width + 0.015, width - 0.015, np.abs(ang) * r / np.maximum(r, 1)) * (r > 0.95) * (r < 1 + length)
    ray = cv2.GaussianBlur(ray.astype(np.float32), (0, 0), 1.2)
    light = (disc * 1.6)[..., None] * np.float32([0.92, 0.98, 1.0]) \
        + (corona * 0.55)[..., None] * np.float32([0.45, 0.82, 1.0]) \
        + (ray * 0.75 * np.clip(1.2 - (r - 1) / (length + 1e-3), 0, 1))[..., None] * np.float32([0.55, 0.88, 1.0])
    streak = np.exp(-((YY[::q, ::q] - sy) / (R * 0.10)) ** 2) * np.exp(-((XX[::q, ::q] - sx) / (W * 0.9)) ** 2) * 0.6
    light = light + streak[..., None] * np.float32([1.0, 0.85, 0.6])           # cool-blue anamorphic streak
    out = f + cv2.resize(light, (W, H)) * flare
    # lens ghosts along the axis from the sun through the frame centre
    g = np.zeros((h, w, 3), np.float32)
    for pos, size, col in ((0.55, 0.20, (0.55, 0.75, 1.0)), (1.25, 0.11, (1.0, 0.65, 0.4)), (1.55, 0.34, (0.6, 1.0, 0.7)), (1.9, 0.07, (1.0, 0.9, 0.9))):
        gx = sx + (W / 2 - sx) * pos; gy = sy + (H * 0.45 - sy) * pos
        poly = _hexagon(gx / q, gy / q, size * R * 2.2 / q)
        m = np.zeros((h, w), np.uint8); cv2.fillPoly(m, [np.round(poly * 4).astype(np.int32)], 255, cv2.LINE_AA, shift=2)
        mf = m.astype(np.float32) / 255
        rim = np.clip(mf - cv2.erode(mf, np.ones((5, 5), np.uint8)), 0, 1)
        g += (mf * 0.10 + rim * 0.12)[..., None] * np.float32(col)
    return out + cv2.resize(cv2.GaussianBlur(g, (0, 0), 1.5), (W, H)) * flare


def t_sunburst(t, c, A, B, a, b, sx=0.80, sy=0.13):
    """The sun rises out of the sky in the corner, its rays spin out, the light floods the frame white,
    and the next shot develops out of the glare."""
    SX, SY = sx * W, sy * H
    if t < c:
        p = clamp((t - a) / (c - a)); e = eout(p)
        f = layer(A, t, 1 + 0.04 * p) * (1 + 0.35 * ease(p))
        f = draw_sun(f, SX, SY + 160 * (1 - e), 60 + 90 * e, t * 1.6, 0.3 + 0.9 * e, 0.6 + 0.6 * e)
        wr = ein(clamp((p - 0.72) / 0.28)) * 1.6 * math.hypot(W, H)                # white flood from the sun outward
        d = np.sqrt((XX - SX) ** 2 + (YY - SY) ** 2)
        return f + (ss(wr + 220, wr - 220, d) * 1.2)[..., None]
    u = clamp((t - c) / (b - c))
    f = layer(B, t, 1 + 0.12 * (1 - eout(u))) * (1 + 0.6 * (1 - eout(u)))
    f = draw_sun(f, SX, SY, 150, t * 1.6, 1.2, 1.1 * (1 - ease(u)))
    return f + (1 - eout(u / 0.45)) * 1.1


def _cloud_bank():
    """Cumulus bank: many billows of mixed size; shaded as a height field lit from the upper left,
    bright tops, blue-grey bellies, ambient occlusion in the folds, silver lining on the edges."""
    if 'cloud' not in _SUNK:
        q = 2; w, h = int(W * 2.6) // q, int(H * 1.0) // q; r = np.random.default_rng(23)
        Hh = np.zeros((h, w), np.float32)
        for _ in range(260):                                                     # big billows low, small ones on top
            cx = r.uniform(0.16, 0.84) * w; big = r.random() < 0.35
            rad = (r.uniform(0.09, 0.16) if big else r.uniform(0.03, 0.08)) * h * (1.5 - abs(cx / w - 0.5)) * min(1.0, (0.5 - abs(cx / w - 0.5)) / 0.12)
            cy = h * (0.56 + (0.10 if big else -0.02) + 0.17 * r.normal())
            yy0, yy1, xx0, xx1 = int(max(0, cy - rad)), int(min(h, cy + rad)), int(max(0, cx - rad)), int(min(w, cx + rad))
            if yy1 <= yy0 or xx1 <= xx0: continue
            Y, X = np.mgrid[yy0:yy1, xx0:xx1].astype(np.float32)
            d2 = 1 - ((X - cx) ** 2 + (Y - cy) ** 2) / rad ** 2
            Hh[yy0:yy1, xx0:xx1] = np.maximum(Hh[yy0:yy1, xx0:xx1], np.sqrt(np.clip(d2, 0, 1)) * rad / h * 6)
        n = fbm(h, w, 5, 24, 18); n2 = fbm(h, w, 4, 25, 60)
        Hh = Hh + 0.10 * (n - 0.5) * (Hh > 0.02) + 0.04 * (n2 - 0.5) * (Hh > 0.02)
        xw = np.linspace(0, 1, w, dtype=np.float32)[None, :]
        dens = ss(0.02, 0.10, Hh)
        Hs = cv2.GaussianBlur(Hh, (0, 0), 1.5)
        gy, gx = np.gradient(Hs * 60)
        nrm = np.sqrt(gx * gx + gy * gy + 1); nx, ny, nz = -gx / nrm, -gy / nrm, 1 / nrm
        L = np.array([-0.55, -0.75, 0.45]); L /= np.linalg.norm(L)
        lam = np.clip(nx * L[0] + ny * L[1] + nz * L[2], 0, 1)
        ao = np.clip(Hs / (cv2.GaussianBlur(Hs, (0, 0), 25) + 1e-3), 0.4, 1.3) ** 0.6   # folds darker than crowns
        vert = np.linspace(1.0, 0.0, h, dtype=np.float32)[:, None]                 # bellies (lower) bluer and darker
        lit = (0.45 + 0.65 * lam) * ao * (0.82 + 0.25 * vert)
        hi = np.float32([0.93, 0.98, 1.0]); sh = np.float32([0.78, 0.62, 0.50])    # warm-white light, blue-grey shadow (BGR)
        k = np.clip(lit, 0, 1.2)[..., None]
        col = sh * (1 - np.clip(k, 0, 1)) + hi * np.clip(k, 0, 1) + np.clip(k - 1, 0, 0.2) * 0.8
        rimm = np.clip(dens - cv2.erode(dens, np.ones((5, 5), np.uint8)), 0, 1)
        col = col + cv2.GaussianBlur(rimm, (0, 0), 1.5)[..., None] * 0.35
        _SUNK['cloud'] = (cv2.resize(np.clip(col, 0, 1.3).astype(np.float32), (w * q, h * q), interpolation=cv2.INTER_CUBIC),
                          cv2.resize(dens.astype(np.float32), (w * q, h * q), interpolation=cv2.INTER_CUBIC))
    return _SUNK['cloud']


def t_clouds(t, c, A, B, a, b):
    """A sunlit cumulus bank rolls across the frame (front layer fast, back layer slower and hazier);
    the next shot is behind it."""
    col, dens = _cloud_bank(); cw = col.shape[1]
    u = clamp((t - a) / (b - a)); e = 0.25 * u + 0.75 * ease(u)
    fa, fb = layer(A, t), layer(B, t)
    edge = lerp(-0.25 * W, 1.25 * W, e) + 70 * np.sin(np.arange(H, dtype=np.float32) / 130 + t * 3)[:, None]
    m = ss(60, -60, XX - edge)[..., None]                                        # B left of the bank's middle
    f = fa * (1 - m) + fb * m
    for sp, sc, haze in ((0.85, 0.85, 0.35), (1.0, 1.0, 0.0)):                   # back layer, then front layer
        eb = e if sp == 1.0 else 0.15 * u + 0.85 * ease(clamp(u * 1.08))
        x0 = int(lerp(-cw * sc - 0.1 * W, W + 0.1 * W, eb)); y0 = int(H * (0.5 - 0.5 * sc))
        Mx = np.float32([[sc, 0, x0], [0, sc, y0]])
        cl = cv2.warpAffine(col, Mx, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        al = cv2.warpAffine(dens, Mx, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)[..., None]
        cl = cl * (1 - haze) + np.float32([0.95, 0.85, 0.75]) * haze
        f = f * (1 - al * (0.85 if haze else 1.0)) + cl * al * (0.85 if haze else 1.0)
    return f


def t_runby(t, c, A, B, a, b, y0=0.52):
    """The Flash runs across the frame as a lightning comet: white-hot head, red-gold tail, lightning
    crackling in the wake, wind smear behind him; the next shot is revealed in his wake."""
    u = clamp((t - a) / (b - a)); e = 0.15 * u + 0.85 * ease(u)
    hx = lerp(-0.35 * W, 1.35 * W, e); slope = -0.12
    yh = lambda x: H * y0 + slope * (x - W / 2) + 30 * np.sin(x / 140.0)
    hy = float(yh(np.float32(hx)))
    fa, fb = layer(A, t), layer(B, t)
    path_y = yh(np.arange(W, dtype=np.float32))[None, :]
    dy = YY - path_y; behind = hx - XX                                            # >0 behind the head
    m = ss(-40, 40, behind - 60 - 50 * np.sin(YY / 90 + t * 9))[..., None]
    base = fa * (1 - m) + fb * m
    wake = np.exp(-(dy / (H * 0.10)) ** 2) * np.exp(-np.clip(behind, 0, None) / (W * 0.5)) * (behind > -60)
    sm = cv2.blur(base, (151, 1))                                                 # wind smear in the wake
    base = base * (1 - wake[..., None] * 0.85) + sm * (wake[..., None] * 0.85)
    tail = np.exp(-(dy / (9 + 0.03 * np.clip(behind, 0, None))) ** 2) * np.exp(-np.clip(behind, 0, None) / (W * 0.45)) * (behind > -20)
    core = np.exp(-(dy / 7) ** 2) * np.exp(-np.clip(behind, 0, None) / (W * 0.18)) * (behind > -10)
    head = np.exp(-(((XX - hx) / 46) ** 2 + (dy / 20) ** 2))
    light = (tail * 0.9)[..., None] * np.float32([0.05, 0.25, 1.0]) + (tail * 0.5 + core)[..., None] * np.float32([0.25, 0.75, 1.0]) \
        + (head * 2.2 + core * 0.8)[..., None] * np.float32([0.85, 0.97, 1.0])
    f = base + light
    rng = np.random.default_rng(int(t * FPS / 2)); segs = []
    for _ in range(5):                                                            # lightning crackling along the wake
        x1 = hx - rng.uniform(0.05, 0.9) * W; x2 = x1 - rng.uniform(80, 300)
        if x1 < -50 or x1 > W + 50: continue
        segs += bolt_tree((x1, float(yh(np.float32(x1))) + rng.normal(0, 20)), (x2, float(yh(np.float32(x2))) + rng.normal(0, 90)), rng, branches=1, depth=6)
    return add(f, draw_bolts(segs, 3.0, amp=1.0))


# clock with lightning-bolt hands
def _clock_face(Rs=700):
    if 'face' in _SUNK: return _SUNK['face']
    S = 2 * Rs + 40; c = S / 2
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32); r = np.sqrt((xx - c) ** 2 + (yy - c) ** 2) / Rs
    face = np.stack([0.80 + 0.12 * (1 - r), 0.90 + 0.06 * (1 - r), 0.95 + 0.03 * (1 - r)], -1).astype(np.float32)   # warm ivory
    a = ss(1.0, 0.995, r).astype(np.float32)
    bez = ss(0.88, 0.90, r) * ss(1.0, 0.995, r)
    lam = np.clip(((xx - c) * -0.6 + (yy - c) * -0.8) / (Rs * r + 1e-3), -1, 1) * np.where(r > 0.94, 1, -1)
    gold = np.float32([0.10, 0.55, 0.92]) * (0.55 + 0.45 * lam[..., None]) + (np.clip(lam, 0, 1) ** 8 * 0.6)[..., None]
    face = face * (1 - bez[..., None]) + gold * bez[..., None]
    img = (np.clip(face, 0, 1) * 255).astype(np.uint8); ink = (25, 22, 20)
    for k in range(60):                                                           # minute and hour ticks
        th = k * math.pi / 30; r0 = 0.80 if k % 5 == 0 else 0.84; r1 = 0.87
        p0 = (int(c + Rs * r0 * math.sin(th)), int(c - Rs * r0 * math.cos(th))); p1 = (int(c + Rs * r1 * math.sin(th)), int(c - Rs * r1 * math.cos(th)))
        cv2.line(img, p0, p1, ink, 9 if k % 5 == 0 else 3, cv2.LINE_AA)
    nums = ['XII', 'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI']
    for k, s in enumerate(nums):
        th = k * math.pi / 6; px, py = c + Rs * 0.67 * math.sin(th), c - Rs * 0.67 * math.cos(th)
        sc = 2.4; (tw, thh), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_TRIPLEX, sc, 5)
        cv2.putText(img, s, (int(px - tw / 2), int(py + thh / 2)), cv2.FONT_HERSHEY_TRIPLEX, sc, ink, 5, cv2.LINE_AA)
    cv2.circle(img, (int(c), int(c)), int(Rs * 0.5), (90, 150, 200), 3, cv2.LINE_AA)       # inner gold ring
    f = img.astype(np.float32) / 255 * a[..., None]
    _SUNK['face'] = (np.dstack([f, a]), Rs)
    return _SUNK['face']


_HAND = np.array([(0.0, 0.0), (0.10, -0.42), (0.035, -0.40), (0.12, -0.92), (-0.06, -0.36), (0.01, -0.38), (-0.07, 0.0)])


def draw_clock(f, cx, cy, R, ang_h, ang_m, ang_s, open_r=0.0, window=None, swirl=0.0):
    face, Rs = _clock_face(); sc = R / Rs; S = face.shape[0]
    M = np.float32([[sc, 0, cx - sc * S / 2], [0, sc, cy - sc * S / 2]])
    lay = cv2.warpAffine(face, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    shd = cv2.GaussianBlur(np.roll(np.roll(lay[..., 3], int(min(0.06 * R, 50)), 0), int(min(0.04 * R, 35)), 1), (0, 0), float(np.clip(0.05 * R, 2, 40))) * 0.55 if R < 2500 else 0
    out = f * (1 - (shd[..., None] if R < 2500 else 0))
    out = out * (1 - lay[..., 3:4]) + lay[..., :3]
    if window is not None and open_r > 0:                                        # the face opens from the centre onto `window`
        d = np.sqrt((XX - cx) ** 2 + (YY - cy) ** 2) / R
        if swirl:
            th = swirl * np.clip(1 - d / max(open_r, 1e-3), 0, 1) ** 2
            ca, sa = np.cos(th), np.sin(th); X, Y = XX - cx, YY - cy
            window = cv2.remap(window, X * ca - Y * sa + cx, X * sa + Y * ca + cy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        m = ss(open_r + 0.01, open_r - 0.01, d)[..., None]
        rim = np.exp(-((d - open_r) / 0.012) ** 2)[..., None] * np.float32([0.25, 0.75, 1.0])
        out = out * (1 - m) + window * m + rim * (open_r < 0.88)
    if open_r < 0.85:                                                             # lightning-bolt hands (gold), red second hand
        for ang, L, wdt, col in ((ang_h, 0.45, 1.25, (0.10, 0.62, 0.95)), (ang_m, 0.70, 1.0, (0.10, 0.62, 0.95)), (ang_s, 0.80, 0.35, (0.10, 0.10, 0.85))):
            ca, sa = math.cos(ang), math.sin(ang); P = _HAND * [wdt, L / 0.92]
            pts = np.stack([cx + R * (P[:, 0] * ca - P[:, 1] * sa), cy + R * (P[:, 0] * sa + P[:, 1] * ca)], 1)
            mk = np.zeros((H, W), np.uint8); cv2.fillPoly(mk, [np.clip(np.round(pts * 4), -2 ** 30, 2 ** 30).astype(np.int32)], 255, cv2.LINE_AA, shift=2)
            mf = mk.astype(np.float32)[..., None] / 255
            out = out * (1 - mf) + np.float32(col) * mf + np.clip(mf - cv2.erode(mf[..., 0], np.ones((5, 5), np.uint8))[..., None], 0, 1) * 0.35
        hub = np.exp(-(((XX - cx) ** 2 + (YY - cy) ** 2) / (2 * (0.035 * R) ** 2)))
        out = out * (1 - np.clip(hub * 3, 0, 1)[..., None]) + (np.clip(hub * 3, 0, 1))[..., None] * np.float32([0.15, 0.6, 0.9])
    return out


def t_clock(t, A, B, a, op, end):
    """Loop ending: a clock with lightning-bolt hands pops in over the last shot, its hands spin backwards
    faster and faster (time rewinds), the world swirls, then the face opens and the camera falls through
    it into the first shot."""
    if t >= end: return layer(B, t - DUR)
    cx, cy = W / 2, H * 0.44
    p = clamp((t - a) / (end - a))
    spin = -(2 * np.pi) * (0.4 * (t - a) + 6.0 * (t - a) ** 2.2)                 # accelerating backwards
    fa = layer(A, t)
    sw = 1.4 * ein(clamp((t - a) / (op - a)))                                      # the world swirls around the clock
    if sw > 0.01:
        X, Y = XX - cx, YY - cy; d = np.sqrt(X * X + Y * Y) / (0.9 * H); th = sw * np.exp(-d * 1.4)
        ca, sa = np.cos(th), np.sin(th)
        fa = cv2.remap(fa, X * ca - Y * sa + cx, X * sa + Y * ca + cy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        fa = fa * (1 - 0.35 * ease(sw / 1.4))
    if t < op:
        k = clamp((t - a) / 0.16); pop = eout(k) * (1 + 0.12 * math.sin(math.pi * k))
        R = 0.40 * W * pop
        return draw_clock(fa, cx, cy, max(R, 1.0), spin / 12, spin, spin * 6) if R > 2 else fa
    u = clamp((t - op) / (end - op))
    R = 0.40 * W * math.exp(math.log(5.2) * ein(u))
    fb = layer(B, t - DUR, 1 + 0.25 * (1 - eout(u)))
    return draw_clock(fa, cx, cy, R, spin / 12, spin, spin * 6, open_r=0.05 + 0.95 * eout(u ** 0.8), window=fb, swirl=2.0 * (1 - u))


# ============================================================ Duality kit (Dexter x New Person, Same Old Mistakes)
_DK = {}


def _polar(cx, cy):
    key = ('polar', int(cx), int(cy))
    if key not in _DK:
        X = XX - cx; Y = YY - cy; _DK[key] = (np.sqrt(X * X + Y * Y), np.arctan2(Y, X))
    return _DK[key]


def kaleido(f, k, rot, zoom_, n=6, cx=W / 2, cy=H * 0.42):
    """Fold the frame into an n-way mirrored kaleidoscope (k = blend 0..1)."""
    if k <= 0.01: return f
    rr, th = _polar(cx, cy); seg = 2 * np.pi / n
    a = (th + rot) % seg; a = np.where(a > seg / 2, seg - a, a)               # mirror inside each wedge
    base = -np.pi / 2 - seg / 4                                               # wedge samples upward, over the face
    r2 = rr * zoom_
    mx = (cx + r2 * np.cos(a + base)).astype(np.float32); my = (cy + r2 * np.sin(a + base)).astype(np.float32)
    g = cv2.remap(f, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return f * (1 - k) + g * k if k < 0.999 else g


def t_kaleido(t, c, A, B, a, b):
    """The frame folds into a turning 6-way kaleidoscope, blooms with colour fringes, and unfolds into the next shot."""
    u = clamp((t - a) / (b - a)); k = math.sin(math.pi * u) ** 0.6
    mix = ease(clamp((u - 0.42) / 0.16))
    src = layer(A, t) * (1 - mix) + layer(B, t) * mix if 0 < mix < 1 else (layer(B, t) if mix >= 1 else layer(A, t))
    f = kaleido(src, k, rot=2.4 * ease(u) + 0.6 * t, zoom_=1.0 + 0.55 * k, n=8)
    sp = int(16 * k)
    if sp:
        f = f.copy(); f[..., 2] = np.roll(f[..., 2], sp, 1); f[..., 0] = np.roll(f[..., 0], -sp, 1)
    rr, _ = _polar(W / 2, H * 0.42)
    return f * (1 + 0.15 * k)


# crime-scene tape
def _tape():
    if 'tape' not in _DK:
        from PIL import Image, ImageDraw, ImageFont
        Lw, Ht = 3600, 150; img = Image.new('RGB', (Lw, Ht), (14, 196, 247))      # BGR order handled below: stored as BGR
        d = ImageDraw.Draw(img); f = ImageFont.truetype('/usr/share/fonts/opentype/inter/Inter-Black.otf', 74)
        txt = 'CRIME SCENE  DO NOT CROSS   '; x = 20
        while x < Lw:
            d.text((x, Ht / 2 - 46), txt, font=f, fill=(10, 10, 12)); x += f.getlength(txt)
        a = np.asarray(img).astype(np.float32) / 255
        a[:9] *= 0.25; a[-9:] *= 0.25                                              # dark edge stripes
        n = fbm(Ht // 2, Lw // 8, 3, 91, 6); a *= (0.88 + 0.18 * cv2.resize(n, (Lw, Ht)))[..., None]   # plastic sheen/wrinkles
        _DK['tape'] = a
    return _DK['tape']


_TAPES = [(-14, 0.18, 0.00), (9, 0.36, 0.05), (-6, 0.55, 0.10), (12, 0.73, 0.03), (-10, 0.90, 0.08), (4, 0.08, 0.12), (-3, 0.46, 0.15), (7, 0.64, 0.02), (-12, 0.99, 0.13)]


def t_tape(t, c, A, B, a, b):
    """Crime-scene tape whips across the frame in layers until it covers it, then rips away to reveal the next shot."""
    T = _tape(); Ht, Lw = T.shape[:2]
    f = layer(A, t) if t < c else layer(B, t)
    al_all = np.zeros((H, W), np.float32); col_all = np.zeros((H, W, 3), np.float32)
    for k, (ang, yf, dl) in enumerate(_TAPES):
        if t < c:
            p = eout(clamp((t - a - dl * (c - a)) / ((c - a) * 0.6))); d = 1 if k % 2 == 0 else -1
            x_off = d * (1 - p) * (Lw * 1.9 / 2 + W * 0.6)                       # leading end starts just off-screen
        else:
            p = ein(clamp((t - c - dl * 0.4 * (b - c)) / ((b - c) * 0.7))); d = -1 if k % 2 == 0 else 1
            x_off = d * p * (Lw * 1.9 / 2 + W * 0.6)                             # pulled fully clear lengthwise
        th = math.radians(ang + 4 * math.sin(t * 7 + k)); sc = 1.9
        cx, cy = W / 2 + x_off, yf * H
        M = cv2.getRotationMatrix2D((Lw / 2, Ht / 2), -math.degrees(th), sc); M[0, 2] += cx - Lw / 2; M[1, 2] += cy - Ht / 2
        col = cv2.warpAffine(T, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        al = cv2.warpAffine(np.ones((Ht, Lw), np.float32), M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        sh = cv2.GaussianBlur(np.roll(al, 22, 0), (0, 0), 12) * 0.45
        f = f * (1 - sh[..., None]); f = f * (1 - al[..., None]) + col * al[..., None]
    return f


# Rorschach blot
def _blot(seed=17):
    """Symmetric ink-blot distance field: a few big mirrored lobes on the fold, ragged symmetric edges."""
    if 'blot' not in _DK:
        h, w = H // 4, W // 4; r = np.random.default_rng(seed); yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        F = np.sqrt(((xx / w) - 0.5) ** 2 * 25 + 0.0 * yy) / 1.0                    # starts as a line on the fold
        for sx, sy, rad in ((0.0, 0.42, 0.20), (0.16, 0.30, 0.15), (0.20, 0.55, 0.13), (0.10, 0.72, 0.14), (0.28, 0.40, 0.08), (0.0, 0.88, 0.09)):
            for X in (0.5 - sx, 0.5 + sx):
                F = np.minimum(F, np.sqrt(((xx / w) - X) ** 2 + ((yy / h) - sy) ** 2 * 0.40) / rad)
        n = fbm(h, w, 5, seed, 10); n = (n + n[:, ::-1]) / 2
        F = F + 0.32 * (n - 0.5)
        _DK['blot'] = (cv2.resize(F, (W, H), interpolation=cv2.INTER_CUBIC), float(np.percentile(F, 99.8)) + 0.2)
    return _DK['blot']


def t_rorschach(t, c, A, B, a, b):
    """A symmetric ink blot blooms out from the fold like a Rorschach card; the next shot lives inside it."""
    Fu, Fmax = _blot(); u = clamp((t - a) / (b - a))
    g = lerp(0.05, Fmax, 0.15 * u + 0.85 * ein(u) ** 0.9)
    inside = ss(g + 0.03, g - 0.03, Fu)[..., None]
    rim = np.exp(-((Fu - g) / 0.05) ** 2) + 0.6 * ss(g + 0.12, g, Fu) * ss(g - 0.02, g + 0.02, Fu)
    fa = layer(A, t); fb = layer(B, t, 1 + 0.12 * (1 - u))
    paper = fa * (1 - 0.35 * ease(u * 2))                                          # the old shot darkens like aged paper
    f = paper * (1 - inside) + fb * inside
    ink = np.float32([0.02, 0.02, 0.03])
    m = np.clip(rim * 1.4, 0, 1)[..., None]
    f = f * (1 - m) + ink * m
    sheen = np.exp(-((Fu - g + 0.03) / 0.02) ** 2) * 0.25
    return f + sheen[..., None] * np.float32([0.9, 0.85, 0.8])


# two faces: mirror, then doors
def t_mirror(t, c, A, B, a, b):
    """His face folds into a perfect mirror image (the two Dexters), a seam of light splits it, and the halves
    swing open like doors onto the next shot."""
    s = SHOTS[A]; fa = layer(A, min(t, c))
    mid = W // 2
    mirrored = fa.copy(); mirrored[:, mid:] = fa[:, :mid][:, ::-1][:, :W - mid]
    if t < c:
        k = ease(clamp((t - a) / ((c - a) * 0.5)))
        f = fa * (1 - k) + mirrored * k
        seam = np.exp(-((XX - mid) / 3) ** 2) * ease(clamp((t - a) / (c - a)))
        return f + (seam * 1.2 + cv2.GaussianBlur(seam, (0, 0), 18) * 0.6)[..., None] * np.float32([0.9, 0.95, 1.0])
    u = clamp((t - c) / (b - c)); ang = ein(u) * math.radians(88)
    fb = layer(B, t, 1 + 0.1 * (1 - u))
    out = fb * (0.55 + 0.45 * ease(u))
    wl = max(2, int(mid * math.cos(ang))); dpt = H * 0.10 * math.sin(ang)
    for side in (0, 1):
        half = mirrored[:, :mid] if side == 0 else mirrored[:, mid:]
        hw = half.shape[1]
        src = np.float32([[0, 0], [hw, 0], [hw, H], [0, H]])
        if side == 0: dst = np.float32([[0, 0], [wl, dpt], [wl, H - dpt], [0, H]])
        else: dst = np.float32([[W - wl, dpt], [W, 0], [W, H], [W - wl, H - dpt]])
        P = cv2.getPerspectiveTransform(src, dst)
        door = cv2.warpPerspective(half * (1 - 0.6 * math.sin(ang)), P, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        m = cv2.warpPerspective(np.ones((H, hw), np.float32), P, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)[..., None]
        out = out * (1 - m) + door * m
    return out + (1 - eout(u / 0.25)) * 0.35


# ---- Venetian blinds (noir) and the VHS rewind loop
def t_blinds(t, c, A, B, a, b, n=13):
    """Venetian blinds: drawn slats tilt shut from top to bottom (sun bars sliding through the gaps), then tilt
    open from the bottom onto the next shot."""
    pitch = H / n; k = np.floor(YY[:, :1] / pitch); v = (YY[:, :1] - (k + 0.5) * pitch) / (pitch / 2)    # -1..1 inside a slat
    if t < c:
        p = (t - a) / (c - a); lag = k / n * 0.45; phi = np.clip((p - lag) / 0.55, 0, 1) ** 3 * (math.pi / 2)
        f = layer(A, t)
    else:
        p = (t - c) / (b - c); lag = (n - 1 - k) / n * 0.45; phi = (1 - np.clip((p - lag) / 0.55, 0, 1)) ** 3 * (math.pi / 2)
        f = layer(B, t, 1 + 0.06 * (1 - p))
    sphi = np.sin(phi)
    cover = ss(sphi + 0.02, sphi - 0.02, np.abs(v)) * (sphi > 0.01)
    shade = 0.30 + 0.62 * np.clip(np.cos(phi - math.radians(35)), 0, 1)                     # lit from above
    curve = 1 - 0.18 * (v / np.maximum(sphi, 1e-3)) ** 2
    hi = np.exp(-((v + sphi) / 0.06) ** 2) * 0.35; lo = np.exp(-((v - sphi) / 0.10) ** 2) * 0.35
    slat = np.float32([0.80, 0.88, 0.93]) * (shade * curve)[..., None] + hi[..., None] - lo[..., None] * 0.5
    slat = slat + (0.04 * np.sin(XX[:1, :] / 3.0))[..., None] * 0.2                          # faint grain
    gap = (1 - cover) * (sphi > 0.05)
    bars = (np.exp(-((XX - (YY * 0.35) - lerp(-0.4, 1.4, clamp((t - a) / (b - a))) * W) / 260) ** 2) * 0.45)[..., None]
    f = f * (1 - 0.45 * (sphi * (1 - cover))[..., None]) + bars * gap[..., None] * np.float32([0.55, 0.80, 1.0])   # sun through the gaps
    shadow = np.clip(np.roll(cover, 14, 0) - cover, 0, 1) * 0.5
    f = f * (1 - shadow[..., None])
    return f * (1 - cover[..., None]) + slat * cover[..., None]


def _vhs(f, k, t):
    """VHS rewind look: jittered rows, a tracking band, chroma bleed, scanlines and noise, strength k."""
    if k <= 0.01: return f
    rng = np.random.default_rng(int(t * FPS))
    rows = rng.normal(0, 1, H).astype(np.float32); rows = np.convolve(rows, np.ones(9) / 9, 'same') * 26 * k
    band = (lerp(1.1, -0.1, (t * 3.1) % 1.0)) * H
    bandm = np.exp(-((np.arange(H) - band) / 40) ** 2).astype(np.float32)
    rows = rows + bandm * 90 * k
    mx = (XX + rows[:, None]).astype(np.float32)
    g = cv2.remap(f, mx, YY, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    sp = int(10 * k); g = g.copy(); g[..., 2] = np.roll(g[..., 2], sp, 1); g[..., 0] = np.roll(g[..., 0], -sp, 1)
    l = luma(g)[..., None]; g = l + (g - l) * (1 - 0.35 * k)
    g = g * (1 - 0.18 * k * ((np.arange(H) % 4) < 2)[:, None, None])
    g = g + (bandm[:, None] * rng.random((H, W)).astype(np.float32) * 0.6 * k)[..., None]
    g = g + rng.normal(0, 0.035 * k, (H // 4, W // 4)).astype(np.float32).repeat(4, 0).repeat(4, 1)[..., None]
    return g


def t_rewind(t, a, end):
    """Loop ending: the whole edit rewinds at speed like a VHS tape (tracking noise, chroma bleed) and lands
    exactly on the first frame. Same old mistakes, again."""
    if t >= end: return layer(0, t - DUR)
    u = clamp((t - a) / (end - a)); e = 0.15 * u + 0.85 * ease(u)
    tau = a * (1 - e); dt_ = a * 0.012
    acc = None
    for j in range(3):                                                                         # motion blur over the rewind
        fr_ = frame(max(0.0, min(a - 1e-3, tau + j * dt_)))
        acc = fr_ if acc is None else acc + fr_
    f = acc / 3
    k = math.sin(math.pi * u) ** 0.5
    f = _vhs(f, k, t) * (1 + 0.25 * k)
    return f


# ============================================================ Dexter kit III: strings, evidence bag, crown splash, blood in water
BLOOD = np.float32([0.04, 0.03, 0.42]); BLOOD_HI = np.float32([0.35, 0.35, 0.95])


def _stain(al, x, y, ang, s, rng):
    """Drawn spatter stain: elongated drop pointing away along `ang`, tail drops and satellites."""
    ca, sa = math.cos(ang), math.sin(ang)
    cv2.ellipse(al, (int(x), int(y)), (int(22 * s), int(13 * s)), math.degrees(ang), 0, 360, 1.0, -1, cv2.LINE_AA)
    for j in range(3):
        d = (30 + 16 * j) * s; rr = (7 - 2 * j) * s
        cv2.circle(al, (int(x + ca * d), int(y + sa * d)), max(1, int(rr)), 1.0, -1, cv2.LINE_AA)
    for _ in range(5):
        aa = ang + rng.normal(0, 0.6); d = rng.uniform(26, 60) * s
        cv2.circle(al, (int(x + math.cos(aa) * d), int(y + math.sin(aa) * d)), max(1, int(rng.uniform(1.5, 4) * s)), 1.0, -1, cv2.LINE_AA)


def t_strings(t, c, A, B, a, b, px=0.5, py=0.40):
    """Blood-spatter string analysis: stains splat around the frame, pinned red strings shoot from each stain and
    converge on the point of origin, the strings twang taut, then the frame bursts open from that point."""
    P = np.array([W * px, H * py]); rng = np.random.default_rng(71); n = 9
    O = []
    for k in range(n):
        ang = -math.pi / 2 + k * 2 * math.pi / n + rng.uniform(-0.18, 0.18)
        O.append(np.clip(P + np.array([math.cos(ang) * W * rng.uniform(0.38, 0.48), math.sin(ang) * H * rng.uniform(0.30, 0.44)]), [50, 80], [W - 50, H - 80]))
    if t < c:
        p = (t - a) / (c - a); f = layer(A, t, 1 + 0.04 * p); hole = 0.0; ret = 0.0
    else:
        u = clamp((t - c) / (b - c)); p = 1.0; hole = eout(u); ret = ease(clamp(u / 0.18))
        fb = layer(B, t, 1 + 0.18 * (1 - eout(u)))
        rr, th = _polar(P[0], P[1])
        R = hole * 1.25 * math.hypot(W, H) * (1 + 0.10 * np.sin(th * 7 + 1.3) + 0.06 * np.sin(th * 13))
        m = ss(R + 12, R - 12, rr)[..., None]
        f = layer(A, c - 1e-3, 1.04) * (1 - m) + fb * m
        f = f + (np.exp(-((rr - R) / 18) ** 2) * (1 - u))[..., None] * BLOOD_HI * 0.9
    stain = np.zeros((H, W), np.float32); lines = np.zeros((H, W), np.float32); hil = np.zeros((H, W), np.float32)
    pins = []
    for k, o in enumerate(O):
        sk = 0.03 + 0.32 * k / (n - 1)
        if p < sk: continue
        pop = eout(clamp((p - sk) / 0.06)); d = P - o; ang = math.atan2(-d[1], -d[0])
        _stain(stain, o[0], o[1], ang, 0.9 + 0.7 * pop, np.random.default_rng(k))
        pk = eout(clamp((p - sk - 0.06) / 0.30)) * (1 - ret)
        if pk <= 0.01: continue
        tt = max(0.0, (p - sk - 0.36) * (c - a))                                   # time since the string went taut
        sag = 60 * (1 - pk) + 22 * math.exp(-tt * 9) * math.sin(tt * 70) * (pk > 0.99)
        E = o + d * pk; nrm = np.array([-d[1], d[0]]) / (np.hypot(*d) + 1e-6)
        s_ = np.linspace(0, 1, 24)[:, None]; pts = o + (E - o) * s_ + nrm * sag * np.sin(np.pi * s_)
        q = np.round(pts * 4).astype(np.int32)
        cv2.polylines(lines, [q], False, 1.0, 8, cv2.LINE_AA, shift=2)
        cv2.polylines(hil, [q + [-4, -4]], False, 1.0, 1, cv2.LINE_AA, shift=2)
        pins.append(o)
    stain *= (1 - ret)
    sh = cv2.GaussianBlur(np.roll(np.roll(np.maximum(lines, stain), 10, 0), 7, 1), (0, 0), 6) * 0.45
    f = f * (1 - sh[..., None])
    f = f * (1 - stain[..., None]) + (BLOOD * (0.8 + 0.4 * cv2.GaussianBlur(stain, (0, 0), 3)[..., None])) * stain[..., None]
    f = f * (1 - lines[..., None]) + np.float32([0.08, 0.07, 0.70]) * lines[..., None] + (hil * lines)[..., None] * 0.35
    for o in (pins if ret < 0.99 else []):                                         # white map pins in the stains
        X = XX - o[0]; Y = YY - o[1]; d2 = X * X + Y * Y
        pin = ss(110, 70, d2)
        f = f * (1 - pin[..., None]) + np.float32([0.88, 0.90, 0.92]) * pin[..., None] + (np.exp(-((X + 3) ** 2 + (Y + 3) ** 2) / 6) * 0.8)[..., None]
    if p >= 0.70 and hole < 0.3:                                                   # the knot at the point of origin glows
        g = np.exp(-(((XX - P[0]) ** 2 + (YY - P[1]) ** 2) / (2 * 26 ** 2))) * clamp((p - 0.70) / 0.3) * (1 - hole / 0.3)
        f = f + g[..., None] * BLOOD_HI * 1.4
    return f


# evidence bag
def _bag_overlay(bw, bh, head):
    key = ('bag', bw, bh)
    if key in _DK: return _DK[key]
    from PIL import Image, ImageDraw, ImageFont
    col = np.zeros((bh, bw, 3), np.float32); al = np.zeros((bh, bw), np.float32)
    yy, xx = np.mgrid[0:bh, 0:bw].astype(np.float32)
    n = cv2.resize(fbm(bh // 8, bw // 8, 4, 41, 4), (bw, bh), interpolation=cv2.INTER_CUBIC)
    dg = (xx * 0.8 + yy * 0.6) / (bw + bh)
    sheen = 0.55 * np.exp(-((dg - 0.30) / 0.05) ** 2) + 0.30 * np.exp(-((dg - 0.62) / 0.025) ** 2) + 0.10 * (n - 0.5)
    al[:] = np.clip(0.08 + sheen * 0.6, 0, 0.6)                                                     # milky clear plastic with streaky sheen
    col[:] = np.float32([0.97, 0.98, 1.0])
    edge = np.minimum(np.minimum(xx, bw - 1 - xx), np.minimum(yy, bh - 1 - yy))
    seam = ss(26, 10, edge); al = np.maximum(al, seam * 0.55)
    img = Image.new('RGB', (bw, head), (255, 255, 255)); d = ImageDraw.Draw(img)
    fb = ImageFont.truetype('/usr/share/fonts/opentype/inter/Inter-Black.otf', 92)
    fr = ImageFont.truetype('/usr/share/fonts/opentype/inter/Inter-SemiBold.otf', 30)
    d.rectangle([0, 40, bw, 170], fill=(205, 30, 34))
    tw = d.textlength('EVIDENCE', font=fb); d.text(((bw - tw) / 2, 50), 'EVIDENCE', font=fb, fill=(255, 255, 255))
    for k, lab in enumerate(('CASE NO.', 'ITEM', 'COLLECTED BY')):
        y = 192 + k * 46; d.text((40, y), lab, font=fr, fill=(30, 30, 34)); d.line([(270, y + 32), (bw - 40, y + 32)], fill=(40, 40, 44), width=3)
    d.text((300, 190), '0915-07', font=fr, fill=(40, 40, 160)); d.text((300, 236), 'SLIDE', font=fr, fill=(40, 40, 160))
    hd = np.asarray(img).astype(np.float32)[..., ::-1] / 255
    col[:head] = hd; al[:head] = 0.97
    zip_ = ss(4, 0, np.abs(yy - 22)); col[zip_ > 0] = np.float32([0.75, 0.78, 0.80]); al = np.maximum(al, zip_)
    _DK[key] = (col, al)
    return _DK[key]


def t_bag(t, c, A, B, a, b):
    """The frame shrinks like a photo going into evidence, a drawn EVIDENCE bag drops around it and zips shut,
    then the bagged evidence falls away onto the next shot."""
    sc = 0.60; cw, ch = int(W * sc), int(H * sc); pad, head = 46, 330; bw, bh = cw + 2 * pad, ch + pad + head + 20
    bx0, by0 = (W - bw) / 2, H * 0.53 - bh / 2 + head / 2 - 40
    col, al = _bag_overlay(bw, bh, head)
    if t < c:
        p = (t - a) / (c - a)
        k = ease(clamp(p / 0.45)); s = lerp(1.0, sc, k)
        cx = lerp(W / 2, bx0 + pad + cw / 2, k); cy = lerp(H / 2, by0 + head + ch / 2, k); rot = 0.0; drop = 0.0
        bag_dy = -(1 - eout(clamp((p - 0.25) / 0.45))) * (H * 1.1); seal = clamp((p - 0.78) / 0.18)
        bg = blur(layer(B, t), 22 * (1 - 0.3 * p)) * (0.45 + 0.1 * p)
    else:
        u = clamp((t - c) / (b - c)); s = sc; cx = bx0 + pad + cw / 2; cy = by0 + head + ch / 2
        drop = ein(u) * H * 1.25; rot = 14 * ein(u); bag_dy = 0.0; seal = 1.0
        bg = blur(layer(B, t), 22 * (1 - eout(u))) * lerp(0.55, 1.0, eout(u))
    fa = layer(A, min(t, c - 1e-3))
    Mc = cv2.getRotationMatrix2D((W / 2, H / 2), -rot, s); Mc[0, 2] += cx - W / 2; Mc[1, 2] += cy - H / 2 + drop
    cont = cv2.warpAffine(fa, Mc, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    cm = cv2.warpAffine(np.ones((H, W), np.float32), Mc, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    Mb = cv2.getRotationMatrix2D((bw / 2, bh / 2), -rot, 1.0)
    Mb[0, 2] += bx0 + (cx - (bx0 + pad + cw / 2)); Mb[1, 2] += by0 + bag_dy + drop
    if t >= c:                                                                     # rotate the bag about the same centre as the content
        Mb = cv2.getRotationMatrix2D((bw / 2, pad + head + ch / 2 - pad), -rot, 1.0); Mb[0, 2] += bx0; Mb[1, 2] += by0 + drop
        Mc = cv2.getRotationMatrix2D((W / 2, H / 2), -rot, s); Mc[0, 2] += cx - W / 2; Mc[1, 2] += cy - H / 2 + drop
        cont = cv2.warpAffine(fa, Mc, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        cm = cv2.warpAffine(np.ones((H, W), np.float32), Mc, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    bc = cv2.warpAffine(col, Mb, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    ba = cv2.warpAffine(al, Mb, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    bodyshape = cv2.warpAffine(np.ones((bh, bw), np.float32), Mb, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    shape = np.maximum(cm, bodyshape)
    sh = cv2.GaussianBlur(np.roll(np.roll(shape, 30, 0), 20, 1), (0, 0), 22) * 0.6
    f = bg * (1 - sh[..., None])
    f = f * (1 - cm[..., None]) + cont * cm[..., None]
    f = f * (1 - ba[..., None]) + bc * ba[..., None]
    if seal > 0:                                                                   # red tamper seal sweeps across the zip
        sm = np.zeros((bh, bw), np.float32); cv2.rectangle(sm, (0, 6), (int(bw * eout(seal)), 38), 1.0, -1)
        sw = cv2.warpAffine(sm, Mb, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        f = f * (1 - sw[..., None]) + np.float32([0.15, 0.12, 0.82]) * sw[..., None]
    return f + (0.35 * (1 - eout(clamp((t - c) / 0.12))) if t >= c else 0)


# crown splash
def t_crown(t, c, A, B, a, b, px=0.5, py=0.46):
    """A glossy blood drop falls and hits; a crown of spikes and flying droplets rises from the impact and the
    spreading pool becomes the next shot."""
    P = np.array([W * px, H * py]); tc = c - 0.08
    rr, th = _polar(P[0], P[1])
    if t < tc:
        p = clamp((t - a) / (tc - a)); y = lerp(-180, P[1], ein(p)); v = p                 # accelerating fall
        f = layer(A, t)
        X = XX - P[0]; Y = YY - y; R = 46
        tail = ss(0, -R * (1.2 + 2.2 * v), Y) * (Y < 0)
        width = R * np.sqrt(np.clip(1 + Y / (R * (1.2 + 2.2 * v)), 0, 1))
        drop = np.maximum(ss(R + 1.5, R - 1.5, np.sqrt(X * X + Y * Y)), (np.abs(X) < width) * tail)
        dcol = BLOOD * (0.7 + 0.5 * ss(-R, R, -X - Y))[..., None] + (np.exp(-((X + R * .35) ** 2 + (Y + R * .35) ** 2) / (2 * 7 ** 2)) * 1.2)[..., None]
        sh = cv2.GaussianBlur(np.roll(drop, 30, 0), (0, 0), 10) * 0.35 * p
        return (f * (1 - sh[..., None])) * (1 - drop[..., None]) + dcol * drop[..., None]
    u = clamp((t - tc) / (b - tc))
    Rp = (40 + 1.3 * math.hypot(W, H) * ein(u ** 0.85)) * (1 + 0.05 * np.sin(th * 9 + 2) + 0.03 * np.sin(th * 17))
    inside = ss(Rp + 3, Rp - 3, rr)[..., None]
    fa, fb = layer(A, min(t, c - 1e-3) if t < c else c - 1e-3), layer(B, t, 1 + 0.15 * (1 - u))
    fb = fb * (1 - 0.55 * (1 - eout(u))) + BLOOD * 0.55 * (1 - eout(u))                     # the pool clears from red to the shot
    f = fa * (1 - inside) + fb * inside
    band = np.exp(-((rr - Rp) / 26) ** 2)
    f = f * (1 - band[..., None] * 0.9) + BLOOD * band[..., None] * 0.9 + (np.exp(-((rr - Rp + 14) / 4) ** 2) * 0.6)[..., None]
    rng = np.random.default_rng(5); can = np.zeros((H, W), np.float32); hl = np.zeros((H, W), np.float32)
    k = math.sin(math.pi * min(1.0, u * 1.6))
    for j in range(20):                                                                     # crown spikes with beads
        ang = j * 2 * math.pi / 20 + rng.uniform(-0.1, 0.1); L = rng.uniform(70, 150) * k; r0 = 40 + 1.3 * math.hypot(W, H) * ein(u ** 0.85)
        base = P + np.array([math.cos(ang), math.sin(ang)]) * r0; tip = base + np.array([math.cos(ang), math.sin(ang)]) * L
        nrm = np.array([-math.sin(ang), math.cos(ang)]) * 16
        cv2.fillConvexPoly(can, np.round(np.array([base + nrm, base - nrm, tip]) * 4).astype(np.int32), 1.0, cv2.LINE_AA, shift=2)
        cv2.circle(can, (int(tip[0]), int(tip[1])), int(10 + 6 * k), 1.0, -1, cv2.LINE_AA)
        cv2.circle(hl, (int(tip[0] - 3), int(tip[1] - 3)), 3, 1.0, -1, cv2.LINE_AA)
    tt = t - tc
    for j in range(34):                                                                     # droplets flung out, falling
        ang = rng.uniform(0, 2 * math.pi); sp = rng.uniform(900, 2600); rad = rng.uniform(5, 17)
        x = P[0] + math.cos(ang) * sp * tt; y = P[1] + math.sin(ang) * sp * tt * 0.8 + 2200 * tt * tt
        cv2.circle(can, (int(x), int(y)), int(rad), 1.0, -1, cv2.LINE_AA); cv2.circle(hl, (int(x - rad * .3), int(y - rad * .3)), max(1, int(rad * .28)), 1.0, -1, cv2.LINE_AA)
    sh = cv2.GaussianBlur(np.roll(can, 16, 0), (0, 0), 8) * 0.4
    f = f * (1 - sh[..., None])
    f = f * (1 - can[..., None]) + BLOOD * (0.85 + 0.4 * cv2.GaussianBlur(can, (0, 0), 4)[..., None]) * can[..., None] + (hl * 0.9)[..., None]
    return f + 0.4 * max(0.0, 1 - tt / 0.06)


# blood in the water
def _water_fields():
    if 'wf' not in _DK:
        h, w = H // 4, W // 4
        _DK['wf'] = (tile_fbm(h, w, octaves=5, seed=81, base=4), tile_fbm(h, w, octaves=4, seed=82, base=7), tile_fbm(h, w, octaves=3, seed=83, base=3))
    return _DK['wf']


def t_bloodwater(t, c, A, B, a, b):
    """The frame goes underwater (blue tint, caustic light, wobble); crimson blood clouds bloom and swirl across it,
    then thin out onto the next shot. Dexter's ocean."""
    N1, N2, N3 = _water_fields(); h, w = N1.shape; u = clamp((t - a) / (b - a)); wk = math.sin(math.pi * u) ** 0.6
    src = layer(A, t) if t < c else layer(B, t)
    wob = 9 * wk; g = cv2.remap(src, XX + wob * np.sin(YY / 45 + t * 6), YY + wob * 0.6 * np.sin(XX / 60 + t * 5), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    tint = g * np.float32([1.10, 0.78, 0.38]) + np.float32([0.14, 0.07, 0.0])
    f = g * (1 - wk) + tint * wk
    o1 = int(t * 40) % h; o2 = int(t * 25) % w
    ca = np.roll(np.roll(N2, o1, 0), o2, 1); cst = (1 - np.abs(2 * ((ca * 6 + 0.3 * np.roll(N3, -o1, 0)) % 1) - 1)) ** 10
    f = f + cv2.resize(cst, (W, H))[..., None] * np.float32([0.45, 0.42, 0.25]) * 0.35 * wk
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dens = np.zeros((h, w), np.float32)
    for (sx, sy, d0) in ((0.42, 0.40, 0.0), (0.66, 0.62, 0.06), (0.30, 0.70, 0.11)):
        gu = clamp((u - d0) / 0.55); R = 0.08 + 0.95 * eout(gu)
        wx = xx + 18 * (np.roll(N1, int(t * 30), 0) - 0.5) * 4; wy = yy + 18 * (np.roll(N3, int(t * 22), 1) - 0.5) * 4 - 30 * u * 4 * 0.25
        dd = np.sqrt(((wx / w) - sx) ** 2 + ((wy / h) - sy) ** 2 * 0.55) / max(R, 1e-3)
        q = dd + 0.75 * (np.roll(N1, int(t * 12), 1) - 0.5) + 0.35 * (N2 - 0.5)
        dens = np.maximum(dens, ss(0.95, 0.70, q) * (0.35 + 0.65 * ss(0.4, 0.8, N3)) * (gu > 0))
    fade = 1 - ease(clamp((u - 0.62) / 0.38))
    dens = dens * fade
    D = cv2.resize(cv2.GaussianBlur(dens, (0, 0), 0.8), (W, H))[..., None]
    edge = np.clip(cv2.resize(cv2.GaussianBlur(dens, (0, 0), 3), (W, H))[..., None] - D, 0, 1)
    col = np.float32([0.03, 0.02, 0.48]) * (1 - D) + np.float32([0.015, 0.01, 0.22]) * D      # bright wispy edges, dark dense cores
    f = f * (1 - D * 0.9) + col * D * 0.9 + edge * np.float32([0.05, 0.05, 0.6]) * 0.8
    return bubbles(f, t, wk * 0.7, seed=9, n=22, t0=a)


# ============================================================ Dexter kit IV: forensic graphics (photo, case file, stamps, markers, ID lock)
def _font_px(path, size):
    from PIL import ImageFont
    key = ('font', path, size)
    if key not in _DK: _DK[key] = ImageFont.truetype(path, size)
    return _DK[key]


INTER_BLACK = '/usr/share/fonts/opentype/inter/Inter-Black.otf'
INTER_SEMI = '/usr/share/fonts/opentype/inter/Inter-SemiBold.otf'
MONO = '/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf'


def _text_mask(txt, path, size, pad=8):
    from PIL import Image, ImageDraw
    key = ('tm', txt, path, size)
    if key in _DK: return _DK[key]
    f = _font_px(path, size); l, t_, r, b = f.getbbox(txt)
    img = Image.new('L', (r - l + 2 * pad, b - t_ + 2 * pad), 0); ImageDraw.Draw(img).text((pad - l, pad - t_), txt, font=f, fill=255)
    _DK[key] = np.asarray(img, np.float32) / 255
    return _DK[key]


def _blit(f, mask, x, y, col, alpha=1.0):
    """Paint a float mask (h, w) with colour at top-left (x, y), clipped to the frame."""
    h, w = mask.shape; x, y = int(x), int(y)
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0: return f
    m = mask[y0 - y:y1 - y, x0 - x:x1 - x][..., None] * alpha
    f[y0:y1, x0:x1] = f[y0:y1, x0:x1] * (1 - m) + np.float32(col) * m
    return f


def _shot_hits(s, after=0.12):
    return sorted(tk for tk in _hits('KICKS') + _hits('SNARES') if s['start'] + after <= tk <= s['end'] - 0.15)


# ---- rubber stamp
def _stamp_sprite(txt):
    key = ('stamp', txt)
    if key in _DK: return _DK[key]
    tm = _text_mask(txt, INTER_BLACK, 132, pad=0); th, tw = tm.shape; bw, bh = tw + 110, th + 90
    m = np.zeros((bh, bw), np.float32); m[45:45 + th, 55:55 + tw] = tm
    cv2.rectangle(m, (10, 10), (bw - 11, bh - 11), 1.0, 9, cv2.LINE_AA); cv2.rectangle(m, (26, 26), (bw - 27, bh - 27), 1.0, 3, cv2.LINE_AA)
    n = cv2.resize(fbm(bh // 3, bw // 3, 4, hash(txt) % 997, 10), (bw, bh)); n2 = np.random.default_rng(3).random((bh, bw)).astype(np.float32)
    ink = m * np.clip((n - 0.18) * 2.6, 0, 1) * (n2 > 0.08)                          # worn, uneven ink
    _DK[key] = cv2.GaussianBlur(ink, (0, 0), 0.8)
    return _DK[key]


def stamp_fx(f, t, s, txt, k=0):
    hits = _shot_hits(s); t0 = hits[min(k, len(hits) - 1)] if hits else s['start'] + 0.25
    if t < t0: return f
    u = t - t0; spr = _stamp_sprite(txt); rng = np.random.default_rng(len(txt))
    ang = rng.uniform(-9, 9); sc = (1 + 0.9 * (1 - eout(u / 0.09))) * min(1.35, 0.85 * W / _stamp_sprite(txt).shape[1]); al = min(1.0, u / 0.04) * 0.95
    h, w = spr.shape; cx, cy = W * rng.uniform(0.40, 0.60), H * 0.70
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, sc); M[0, 2] += cx - w / 2; M[1, 2] += cy - h / 2
    m = cv2.warpAffine(spr, M, (W, H), flags=cv2.INTER_LINEAR)[..., None] * al
    if u < 0.09: m = cv2.GaussianBlur(m[..., 0], (0, 0), 6 * (1 - u / 0.09) + 0.1)[..., None]
    out = f * (1 - m * 0.9) + np.float32([0.10, 0.08, 0.78]) * m * 0.9
    return out * (1 + 0.15 * max(0.0, 1 - u / 0.07))                                  # impact flash


# ---- evidence markers
def _marker_sprite(num):
    key = ('marker', num)
    if key in _DK: return _DK[key]
    w, h = 150, 170; col = np.zeros((h, w, 3), np.float32); al = np.zeros((h, w), np.float32)
    front = np.array([[18, 62], [132, 62], [146, 166], [4, 166]], np.int32); top = np.array([[30, 8], [120, 8], [132, 62], [18, 62]], np.int32)
    cv2.fillConvexPoly(al, front, 1.0, cv2.LINE_AA); cv2.fillConvexPoly(al, top, 1.0, cv2.LINE_AA)
    cv2.fillConvexPoly(col, front, (0.05, 0.80, 0.98), cv2.LINE_AA); cv2.fillConvexPoly(col, top, (0.03, 0.58, 0.78), cv2.LINE_AA)
    cv2.line(col, (18, 62), (132, 62), (0.02, 0.35, 0.50), 2, cv2.LINE_AA)
    tm = _text_mask(str(num), INTER_BLACK, 78, pad=0); th, tw = tm.shape; y0, x0 = 75 + (85 - th) // 2, (w - tw) // 2
    col[y0:y0 + th, x0:x0 + tw] *= (1 - tm[..., None])
    _DK[key] = (col, al)
    return _DK[key]


def markers_fx(f, i, t, s):
    hits = _shot_hits(s)[:3] or [s['start'] + 0.2]
    base = 1 + 3 * sum(1 for j in range(i) if 'markers' in SHOTS[j].get('fx', ()))
    spots = [(0.20, 0.86), (0.78, 0.83), (0.50, 0.92)]
    out = f
    for k, t0 in enumerate(hits):
        if t < t0: continue
        u = t - t0; col, al = _marker_sprite(base + k); h, w = al.shape
        sc = 1.35 * (1 + 0.35 * math.sin(math.pi * min(1, u / 0.12)) * (u < 0.12)) * min(1.0, eout(u / 0.08) + 0.01)
        x, y = spots[k][0] * W, spots[k][1] * H - 40 * (1 - eout(u / 0.10))
        M = np.float32([[sc, 0, x - sc * w / 2], [0, sc, y - sc * h]])
        a = cv2.warpAffine(al, M, (W, H))[..., None]; c_ = cv2.warpAffine(col, M, (W, H))
        sh = cv2.GaussianBlur(np.roll(np.roll(a[..., 0], 10, 0), 16, 1), (0, 0), 9)[..., None] * 0.5
        out = out * (1 - sh) * (1 - a) + c_ * a
    return out


# ---- face lock HUD
def idbox_fx(f, i, t, s, label='SUBJECT 01'):
    k = round((PRE + t - s['start']) * FPS); m = META[i]; k = int(min(max(k, m['lo']), m['hi']))
    lm = None
    for kk in (k, k - 1, k - 2, k + 1, k - 3):
        lm = _landmarks(i, kk)
        if lm: break
    if not lm: return f
    hits = _shot_hits(s, 0.05); t0 = hits[0] if hits else s['start'] + 0.1
    if t < t0: return f
    u = t - t0
    z = max(lerp(s['push'][0], s['push'][1], (t - s['start']) / (s['end'] - s['start'])), 1.0) * (1 + bump_amount(t, s) if 'bump' in s.get('fx', ()) else 1)
    ex, ey = (lm[0][0] + lm[1][0]) / 2, (lm[0][1] + lm[1][1]) / 2; fw = lm[2]
    cx, cy = W / 2 + (ex - W / 2) * z, H * 0.42 + (ey + 0.20 * fw - H * 0.42) * z
    snap = 1 + 0.6 * (1 - eout(u / 0.14)); bw, bh = fw * z * 0.72 * snap, fw * z * 0.95 * snap
    can = np.zeros((H, W), np.float32); L = int(min(bw, bh) * 0.28)
    for sx in (-1, 1):
        for sy in (-1, 1):
            px, py = int(cx + sx * bw), int(cy + sy * bh)
            cv2.line(can, (px, py), (px - sx * L, py), 1.0, 5, cv2.LINE_AA); cv2.line(can, (px, py), (px, py - sy * L), 1.0, 5, cv2.LINE_AA)
    blink = 1.0 if u > 0.14 else (0.5 + 0.5 * math.sin(u * 120))
    out = f * (1 - can[..., None] * blink) + np.float32([0.95, 0.97, 1.0]) * (can * blink)[..., None]
    out = out + cv2.GaussianBlur(can, (0, 0), 6)[..., None] * np.float32([0.2, 0.25, 0.6]) * 0.6 * blink
    lines = [label, 'MATCH %d%%' % min(98, int(40 + 600 * max(0.0, u - 0.15)))]
    ty = int(cy + bh + 18)
    for j, txt in enumerate(lines):
        n = int(len(txt) * clamp((u - 0.12 - 0.12 * j) / 0.22))
        if n <= 0: continue
        tm = _text_mask(txt[:n], MONO, 38, pad=4); x = int(cx - bw)
        out = _blit(out, cv2.dilate(tm, np.ones((5, 5), np.uint8)), x, ty + j * 52, (0, 0, 0), 0.55)
        out = _blit(out, tm, x, ty + j * 52, (0.95, 0.97, 1.0))
    return out


# ---- forensic camera opener
def _viewfinder(f, k, t, case='CASE 0915-07'):
    if k <= 0.01: return f
    can = np.zeros((H, W), np.float32); m, L = 70, 170
    for (x, y, dx, dy) in ((m, m + 40, 1, 1), (W - m, m + 40, -1, 1), (m, H - m - 40, 1, -1), (W - m, H - m - 40, -1, -1)):
        cv2.line(can, (x, y), (x + dx * L, y), 1.0, 11, cv2.LINE_AA); cv2.line(can, (x, y), (x, y + dy * L), 1.0, 11, cv2.LINE_AA)
    cx, cy = W // 2, int(H * 0.42)
    cv2.line(can, (cx - 60, cy), (cx + 60, cy), 1.0, 5, cv2.LINE_AA); cv2.line(can, (cx, cy - 60), (cx, cy + 60), 1.0, 5, cv2.LINE_AA)
    cv2.circle(can, (cx, cy), 150, 1.0, 4, cv2.LINE_AA)
    out = f * (1 - can[..., None] * k * 0.95) + np.float32([0.95, 0.97, 1.0]) * (can * k * 0.95)[..., None]
    for txt, x, y in ((case, m + 10, m + 70), ('ISO 800  1/250  f2.8', m + 10, H - m - 110)):
        tm = _text_mask(txt, MONO, 58, pad=4); out = _blit(out, cv2.dilate(tm, np.ones((7, 7), np.uint8)), x, y, (0, 0, 0), 0.5 * k); out = _blit(out, tm, x, y, (0.95, 0.97, 1.0), k)
    if int(t * 4) % 2 == 0:
        dot = np.zeros((H, W), np.float32); cv2.circle(dot, (W - m - 200, m + 100), 22, 1.0, -1, cv2.LINE_AA)
        out = out * (1 - dot[..., None] * k) + np.float32([0.1, 0.1, 0.95]) * (dot * k)[..., None]
    tm = _text_mask('REC', MONO, 58, pad=4); out = _blit(out, tm, W - m - 165, m + 70, (0.95, 0.97, 1.0), k)
    # ABFO-style L scale along the bottom
    sx0, sy = m + 20, H - m - 170; Lr = int((W - 2 * m - 40) * k)
    sc = np.zeros((H, W), np.float32); cv2.rectangle(sc, (sx0, sy - 34), (sx0 + Lr, sy), 1.0, -1)
    out = out * (1 - sc[..., None] * 0.85) + np.float32([0.92, 0.93, 0.93]) * sc[..., None] * 0.85
    for j in range(0, Lr, 24):
        h_ = 22 if j % 120 == 0 else 12; cv2.line(out, (sx0 + j, sy - 34), (sx0 + j, sy - 34 + h_), (0.05, 0.05, 0.05), 2, cv2.LINE_AA)
    return out


def _aperture(f, open_, cx=W / 2, cy=H * 0.42, n=6):
    """Camera aperture: n overlapping blades; open_ = 1 fully open, 0 closed."""
    if open_ >= 0.999: return f
    rr, th = _polar(cx, cy); R = open_ * 0.75 * math.hypot(W, H)
    seg = 2 * math.pi / n; rot = (1 - open_) * 1.1
    a = (th + rot) % seg - seg / 2
    poly_r = R * math.cos(seg / 2) / np.maximum(np.cos(a), 1e-3)                     # n-gon opening
    blade = ss(poly_r - 2, poly_r + 2, rr)
    edges = np.exp(-(((th + rot + seg * 0.5) % seg - seg / 2) * rr / 4) ** 2) * blade
    shade = 0.10 + 0.08 * np.cos(th * n + rot * 3)
    col = np.float32([1, 1, 1]) * shade[..., None] + (edges * 0.25)[..., None]
    rim = np.exp(-((rr - poly_r) / 3) ** 2) * 0.5
    return f * (1 - blade[..., None]) + col * blade[..., None] + rim[..., None]


def t_photo(t, c, A, B, a, b):
    """Forensic camera: a viewfinder (brackets, case number, REC, scale) frames the shot, focus pulls in, the
    aperture blades snap shut, flash, and they open on the next shot."""
    if t < c:
        p = (t - a) / (c - a); k = eout(clamp(p / 0.35))
        f = blur(layer(A, t, 1 + 0.05 * p), 10 * (1 - ease(clamp(p / 0.5))) * (p < 0.5))
        f = _viewfinder(f, k, t)
        return _aperture(f, 1 - ein(clamp((p - 0.72) / 0.28)))
    u = clamp((t - c) / (b - c)); f = layer(B, t, 1 + 0.08 * (1 - u))
    f = _viewfinder(f, 1 - ease(clamp(u / 0.6)), t)
    f = _aperture(f, eout(clamp(u / 0.45)))
    return f + 1.1 * (1 - eout(clamp(u / 0.35)))


# ---- case file
def _folder_sprite():
    if 'folder' in _DK: return _DK['folder']
    from PIL import Image, ImageDraw
    img = Image.new('RGB', (W, H), (222, 196, 140)); d = ImageDraw.Draw(img)               # manila
    d.rectangle([W - 380, 140, W - 60, 230], fill=(205, 176, 118)); d.rectangle([W - 360, 160, W - 80, 210], fill=(245, 245, 240))
    d.text((W - 345, 162), 'MORGAN, DEXTER', font=_font_px(MONO, 34), fill=(30, 30, 30))
    d.text((90, 300), 'CASE FILE', font=_font_px(INTER_BLACK, 120), fill=(70, 55, 35))
    d.text((92, 450), 'MIAMI  0915-07  HOMICIDE', font=_font_px(MONO, 40), fill=(70, 55, 35))
    for k in range(6): d.line([(90, 620 + 70 * k), (W - 90, 620 + 70 * k)], fill=(170, 145, 95), width=3)
    a = np.asarray(img).astype(np.float32)[..., ::-1] / 255
    a *= (0.9 + 0.16 * cv2.resize(fbm(H // 8, W // 8, 4, 64, 6), (W, H)))[..., None]       # paper fibre
    st = _stamp_sprite('CONFIDENTIAL'); sh, sw = st.shape
    M = cv2.getRotationMatrix2D((sw / 2, sh / 2), 12, 0.95); M[0, 2] += W * 0.5 - sw / 2; M[1, 2] += H * 0.66 - sh / 2
    m = cv2.warpAffine(st, M, (W, H))[..., None]; a = a * (1 - m * 0.85) + np.float32([0.12, 0.10, 0.78]) * m * 0.85
    _DK['folder'] = a
    return a


def t_casefile(t, c, A, B, a, b):
    """A manila case-file cover (typed tab, CONFIDENTIAL stamp) swings shut over the frame on its left hinge,
    then swings open onto the next shot."""
    cov = _folder_sprite()
    if t < c: base = layer(A, t); ang = (1 - ein(clamp((t - a) / (c - a)))) * math.radians(88)
    else: base = layer(B, t, 1 + 0.08 * (1 - clamp((t - c) / (b - c)))); ang = eout(clamp((t - c) / (b - c))) * math.radians(88)
    wl = max(2.0, W * math.cos(ang)); dpt = H * 0.08 * math.sin(ang)
    P = cv2.getPerspectiveTransform(np.float32([[0, 0], [W, 0], [W, H], [0, H]]), np.float32([[0, 0], [wl, -dpt], [wl, H + dpt], [0, H]]))
    shade = 1 - 0.45 * math.sin(ang)
    door = cv2.warpPerspective(cov * shade, P, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    m = cv2.warpPerspective(np.ones((H, W), np.float32), P, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)[..., None]
    shadow = np.clip((wl + 140 - XX) / 140, 0, 1) * (XX > wl) * 0.55 * (1 - abs(math.cos(ang)) ** 4)
    base = base * (1 - shadow[..., None])
    return base * (1 - m) + door * m + (0.25 * max(0.0, 1 - abs(t - c) / 0.05))


# ============================================================ Dexter kit V: ID badge opener, blood-slide trophy box loop
def _badge_sprite():
    """Forensics ID badge at 2x (premultiplied RGBA) plus the photo-window rect (in sprite px)."""
    if 'badge' in _DK: return _DK['badge']
    from PIL import Image, ImageDraw
    S = 2; bw, bh = 620 * S, 1040 * S; img = Image.new('RGBA', (bw, bh), (0, 0, 0, 0)); d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, bw - 1, bh - 1], radius=40 * S, fill=(246, 246, 242, 255))
    d.rounded_rectangle([0, 0, bw - 1, 150 * S], radius=40 * S, fill=(22, 44, 92, 255)); d.rectangle([0, 100 * S, bw, 150 * S], fill=(22, 44, 92, 255))
    d.rectangle([bw // 2 - 70 * S, 30 * S, bw // 2 + 70 * S, 58 * S], fill=(200, 205, 215, 255))          # clip slot
    f1 = _font_px(INTER_BLACK, 50 * S); f2 = _font_px(INTER_SEMI, 34 * S); f3 = _font_px(INTER_BLACK, 56 * S); f4 = _font_px(MONO, 30 * S)
    tw = d.textlength('MIAMI METRO POLICE', font=f2); d.text(((bw - tw) / 2, 78 * S), 'MIAMI METRO POLICE', font=f2, fill=(240, 240, 240, 255))
    d.rectangle([0, 150 * S, bw, 220 * S], fill=(178, 32, 38, 255))
    tw = d.textlength('FORENSICS', font=f1); d.text(((bw - tw) / 2, 157 * S), 'FORENSICS', font=f1, fill=(255, 255, 255, 255))
    px0, py0, pw, ph = 40 * S, 250 * S, 342 * S, 608 * S                                            # 9:16 photo window
    d.rectangle([px0 - 6 * S, py0 - 6 * S, px0 + pw + 6 * S, py0 + ph + 6 * S], fill=(30, 30, 34, 255))
    x = px0 + pw + 30 * S
    for k, (txt, f, col) in enumerate((('DEXTER', f3, (20, 20, 24)), ('MORGAN', f3, (20, 20, 24)))):
        d.text((x, 270 * S + k * 70 * S), txt, font=f, fill=col + (255,))
    for k, line in enumerate(('BLOOD', 'SPATTER', 'ANALYST')):
        d.text((x, 450 * S + k * 42 * S), line, font=f2, fill=(178, 32, 38, 255))
    d.text((x, 640 * S), 'ID 4471-D', font=f4, fill=(40, 40, 44, 255)); d.text((x, 685 * S), 'EXP 09/08', font=f4, fill=(40, 40, 44, 255))
    r = np.random.default_rng(4); bx = 40 * S
    while bx < bw - 40 * S:                                                                        # barcode
        w_ = int(r.choice([2, 3, 5, 7])) * S; d.rectangle([bx, 900 * S, bx + w_, 990 * S], fill=(15, 15, 18, 255)); bx += w_ + int(r.choice([2, 3, 4])) * S
    a = np.asarray(img).astype(np.float32) / 255; rgb = a[..., :3][..., ::-1].copy(); al = a[..., 3:4]
    _DK['badge'] = (np.dstack([rgb * al, al]), (px0, py0, pw, ph), S)
    return _DK['badge']


def t_badge(t, c, A, B, a, b):
    """Dexter's lanyard ID badge drops in and swings to rest, a glint crosses the laminate, then the camera pushes
    into the badge photo (which plays the next shot) until it fills the screen."""
    spr, (px0, py0, pw, ph), S = _badge_sprite(); bh, bw = spr.shape[:2]
    p = clamp((t - a) / (c + 0.30 - a)); tz = c - 0.12
    drop = eout(clamp((t - a) / 0.30)); swing = math.radians(16) * math.exp(-(t - a) * 5.0) * math.cos((t - a) * 14) * (t >= a)
    base_s = 0.5                                                                              # sprite px -> screen px at rest
    z = 1.0 if t < tz else math.exp(math.log((W / (pw * base_s)) / 1.0) * ease(clamp((t - tz) / (c + 0.30 - tz))))
    pv = np.array([W / 2, lerp(-bh * base_s - 60, H * 0.12, drop)])                       # pivot: badge top-centre
    ang = swing * (1 - ease(clamp((t - tz) / 0.2)))
    cphoto = np.array([px0 + pw / 2, py0 + ph / 2])                                          # photo centre in sprite px
    s = base_s * z; ca, sa = math.cos(ang), math.sin(ang)
    pv_rel = cphoto - np.array([bw / 2, 0]); rest = pv + s * np.array([ca * pv_rel[0] - sa * pv_rel[1], sa * pv_rel[0] + ca * pv_rel[1]])
    k = ease(clamp((t - tz) / (c + 0.30 - tz)))
    target = np.array([W / 2, H / 2]); shift = (target - rest) * k                              # camera centres the photo as it zooms
    M = np.float32([[s * ca, -s * sa, 0], [s * sa, s * ca, 0]]); M[:, 2] = pv + shift - M[:, :2] @ np.array([bw / 2, 0])
    fa = layer(A, min(t, c - 1e-3)) * (1 - 0.30 * drop) if t < c + 0.2 else None
    lay = cv2.warpAffine(spr, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    # photo window: the next shot, mapped so that at full zoom it is the frame 1:1
    corners = np.array([[px0, py0], [px0 + pw, py0], [px0 + pw, py0 + ph], [px0, py0 + ph]], np.float64)
    scr = corners @ M[:, :2].T + M[:, 2]
    P = cv2.getPerspectiveTransform(np.float32([[0, 0], [W, 0], [W, H], [0, H]]), np.float32(scr))
    fb = layer(B, t)
    win = cv2.warpPerspective(fb, P, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    wm = cv2.warpPerspective(np.ones((H, W), np.float32), P, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)[..., None]
    if fa is None: fa = fb
    strap = np.zeros((H, W), np.float32)                                                      # lanyard strap from the clip up off-screen
    top = M[:, :2] @ np.array([bw / 2, 30 * S]) + M[:, 2]
    for sx in (-1, 1):
        cv2.line(strap, (int(top[0] + sx * 30 * s / base_s), int(top[1])), (int(W / 2 + sx * 150), -40), 1.0, max(2, int(34 * s)), cv2.LINE_AA)
    sh = cv2.GaussianBlur(np.roll(np.roll(np.maximum(lay[..., 3], strap), 26, 0), 18, 1), (0, 0), 16) * 0.55
    f = fa * (1 - sh[..., None])
    f = f * (1 - strap[..., None]) + np.float32([0.45, 0.14, 0.08]) * strap[..., None]
    f = f * (1 - lay[..., 3:4]) + lay[..., :3]
    f = f * (1 - wm) + win * wm
    X = XX - top[0]; g = np.exp(-(((X + (YY - top[1]) * 0.5) - lerp(-500, 900, clamp((t - a - 0.25) / 0.35))) / 50) ** 2)
    f = f + (g * lay[..., 3] * 0.35 * (t < tz))[..., None]                                     # laminate glint
    return f


# ---- blood-slide trophy box loop
_BOX = {}


def _slide_sprite(k, img):
    """Glass blood slide (landscape, 2x): drop circle on the left shows `img`; frosted label 'No. k' on the right."""
    from PIL import Image, ImageDraw
    S = 2; sw, sh = 860 * S, 290 * S; R = 118 * S; cx, cy = 190 * S, sh // 2
    al = np.full((sh, sw), 0.22, np.float32); col = np.ones((sh, sw, 3), np.float32) * np.float32([0.96, 0.97, 0.98])
    yy, xx = np.mgrid[0:sh, 0:sw].astype(np.float32)
    edge = np.minimum(np.minimum(xx, sw - 1 - xx), np.minimum(yy, sh - 1 - yy)); al = np.maximum(al, ss(14, 2, edge) * 0.8)
    al = al + np.exp(-(((xx * 0.4 + yy) - sh * 0.9) / 24) ** 2) * 0.25                         # glass streak
    lab = Image.new('RGB', (330 * S, 230 * S), (236, 236, 230)); d = ImageDraw.Draw(lab)
    d.text((26 * S, 30 * S), 'No. %02d' % (k + 1), font=_font_px(MONO, 54 * S), fill=(40, 40, 140))
    d.line([(26 * S, 120 * S), (300 * S, 120 * S)], fill=(80, 80, 90), width=3 * S); d.text((26 * S, 140 * S), 'D.M.', font=_font_px(MONO, 40 * S), fill=(40, 40, 140))
    lb = np.asarray(lab).astype(np.float32)[..., ::-1] / 255; ly0, lx0 = 30 * S, 500 * S
    col[ly0:ly0 + lb.shape[0], lx0:lx0 + lb.shape[1]] = lb; al[ly0:ly0 + lb.shape[0], lx0:lx0 + lb.shape[1]] = 0.97
    d2 = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2); drop = ss(R + 2, R - 2, d2)
    sq = cv2.resize(img, (2 * R, 2 * R), interpolation=cv2.INTER_AREA)
    col[cy - R:cy + R, cx - R:cx + R] = col[cy - R:cy + R, cx - R:cx + R] * (1 - drop[cy - R:cy + R, cx - R:cx + R, None]) + sq * drop[cy - R:cy + R, cx - R:cx + R, None]
    al = np.maximum(al, drop)
    rim = np.exp(-((d2 - R) / (4 * S)) ** 2); col = col * (1 - rim[..., None] * 0.8) + BLOOD * rim[..., None] * 0.8; al = np.maximum(al, rim * 0.9)
    al = np.clip(al, 0, 1)
    return np.dstack([col * al[..., None], al]), (cx, cy, R), S


def _box_assets():
    if 'slides' in _BOX: return _BOX
    n = len(SHOTS); _BOX['slides'] = []
    for k in range(n):
        s = SHOTS[k]; fr = layer(k, (s['start'] + s['end']) / 2)
        sq = fr[int(H * 0.42) - W // 2:int(H * 0.42) + W // 2, :] if H > W else fr
        _BOX['slides'].append(_slide_sprite(k, sq))
    _BOX['gap'] = 330                                                                         # screen px between slides at z=1
    hh = _BOX['gap'] * n + 2 * H
    wood = cv2.resize(fbm(64, 512, 5, 17, 6), (W, 1024), interpolation=cv2.INTER_CUBIC)       # rosewood grain (stretched along y)
    wood = cv2.resize(wood, (W, hh), interpolation=cv2.INTER_LINEAR)
    grain = 0.5 + 0.5 * np.sin(np.arange(W, dtype=np.float32)[None, :] / 9 + wood * 18)
    base = np.stack([0.05 + 0.04 * grain, 0.08 + 0.06 * grain, 0.20 + 0.10 * grain], -1).astype(np.float32)
    for k in range(n + 1):                                                                    # dark slot grooves between slides
        y = int(H + k * _BOX['gap'] - _BOX['gap'] / 2); base[max(0, y - 10):y + 10] *= 0.35
    _BOX['wood'] = base
    return _BOX


def t_slidebox(t, A, a, end):
    """Loop: the last shot shrinks into the blood drop of a glass slide; pull back to Dexter's rosewood slide box, one
    slide per shot of the edit; race up past them all to slide No. 01, and dive into its drop, which is the first shot."""
    if t >= end: return layer(0, t - DUR)
    bx = _box_assets(); n = len(SHOTS); gap = bx['gap']
    u = clamp((t - a) / (end - a))
    base_s = 0.5; spr0, (dcx, dcy, R), S = bx['slides'][0]; sh_, sw_ = spr0.shape[:2]
    Rfull = 0.5 * math.hypot(W, H) * 1.02; zmax = Rfull / (R * base_s)
    yk = lambda k: H + k * gap                                                                 # slide centre in world (z=1 screen units)
    dx_off = (dcx - sw_ / 2) * base_s                                                          # drop centre offset from slide centre
    if u < 0.25: e = ease(u / 0.25); z = math.exp(math.log(zmax) * (1 - e)); camy = yk(n - 1); camx = W / 2 + dx_off * (1 - e)
    elif u < 0.75: e = ease((u - 0.25) / 0.50); z = 1.0 - 0.10 * math.sin(math.pi * e); camy = lerp(yk(n - 1), yk(0), e); camx = W / 2
    else: e = ein((u - 0.75) / 0.25); z = math.exp(math.log(zmax) * e); camy = yk(0); camx = W / 2 + dx_off * e
    speed = abs(yk(0) - yk(n - 1)) / (0.50 * (end - a)) * (0.25 <= u < 0.75) * math.sin(math.pi * clamp((u - 0.25) / 0.50))
    # world -> screen: screen = (world - cam) * z + (W/2, H/2)
    wood = bx['wood']; Mw = np.float32([[z, 0, W / 2 - camx * z], [0, z, H / 2 - camy * z]])
    f = cv2.warpAffine(wood, Mw, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    for k in range(n):
        cy_s = (yk(k) - camy) * z + H / 2
        if abs(cy_s) > H / 2 + sh_ * base_s * z: continue
        spr, (dcx, dcy, R), S = bx['slides'][k]
        Ms = np.float32([[base_s * z, 0, 0], [0, base_s * z, 0]]); Ms[:, 2] = np.array([(W / 2 - camx) * z + W / 2, cy_s]) - Ms[:, :2] @ np.array([sw_ / 2, sh_ / 2])
        lay = cv2.warpAffine(spr, Ms, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        shd = cv2.GaussianBlur(np.roll(np.roll(lay[..., 3], 14, 0), 10, 1), (0, 0), 10) * 0.5
        f = f * (1 - shd[..., None]) * (1 - lay[..., 3:4]) + lay[..., :3]
        if k in (0, n - 1) and z > 1.5:                                                         # live full-res frame inside the zoomed drop
            c_ = Ms[:, :2] @ np.array([dcx, dcy]) + Ms[:, 2]; Rs = R * base_s * z
            live = layer(k, t) if k == n - 1 else layer(0, t - DUR)
            sc = Rs / Rfull * lerp(Rfull / (W / 2), 1.0, clamp(Rs / Rfull) ** 2); Ml = np.float32([[sc, 0, c_[0] - sc * W / 2], [0, sc, c_[1] - sc * H / 2]])
            lv = cv2.warpAffine(live, Ml, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            rr = np.sqrt((XX - c_[0]) ** 2 + (YY - c_[1]) ** 2); m = ss(Rs + 2, Rs - 2, rr)[..., None]
            f = f * (1 - m) + lv * m + (np.exp(-((rr - Rs) / max(3, 0.03 * Rs)) ** 2) * 0.8)[..., None] * BLOOD * (Rs < Rfull * 0.98)
    if speed > 50:
        kb = int(min(61, speed / 160)) | 1; f = cv2.blur(f, (1, kb))
    return f


# ---- spinning newspaper headline
SERIF_B = '/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf'


def _paper_sprite():
    """Front page at 2x (premultiplied RGBA) and its 9:16 photo window (sprite px)."""
    if 'paper' in _DK: return _DK['paper']
    from PIL import Image, ImageDraw
    S = 2; pw, ph = 930 * S, 1330 * S
    img = Image.new('RGB', (pw, ph), (232, 226, 210)); d = ImageDraw.Draw(img); ink = (28, 26, 24)
    mast = _font_px(SERIF_B, 100 * S); tw = d.textlength('MIAMI TRIBUNE', font=mast); d.text(((pw - tw) / 2, 22 * S), 'MIAMI TRIBUNE', font=mast, fill=ink)
    d.line([(40 * S, 162 * S), (pw - 40 * S, 162 * S)], fill=ink, width=5 * S); d.line([(40 * S, 206 * S), (pw - 40 * S, 206 * S)], fill=ink, width=2 * S)
    sm = _font_px(SERIF_B, 26 * S)
    d.text((44 * S, 170 * S), 'THURSDAY, NOVEMBER 15', font=sm, fill=ink); t2 = 'FINAL EDITION   50¢'; d.text((pw - 44 * S - d.textlength(t2, font=sm), 170 * S), t2, font=sm, fill=ink)
    hs = 84
    while d.textlength('BAY HARBOR BUTCHER', font=_font_px(SERIF_B, hs * S)) > pw - 110 * S: hs -= 2
    hl = _font_px(SERIF_B, hs * S)
    for k, line in enumerate(('BAY HARBOR BUTCHER', 'STRIKES AGAIN')):
        tw = d.textlength(line, font=hl); d.text(((pw - tw) / 2, (222 + k * 108) * S), line, font=hl, fill=ink)
    sub = _font_px(SERIF_B, 34 * S); t3 = 'Police baffled as more remains surface off the coast'
    tw = d.textlength(t3, font=sub); d.text(((pw - tw) / 2, 452 * S), t3, font=sub, fill=(60, 56, 52))
    px0, py0, pww, phh = 44 * S, 510 * S, 448 * S, 796 * S
    d.rectangle([px0 - 3 * S, py0 - 3 * S, px0 + pww + 3 * S, py0 + phh + 3 * S], outline=ink, width=3 * S)
    r = np.random.default_rng(12); x0 = px0 + pww + 30 * S
    for col in range(2):
        cx = x0 + col * 196 * S; y = py0
        while y < ph - 50 * S:                                                                   # text columns
            if r.random() < 0.08: y += 22 * S; continue
            w_ = int((176 if r.random() > 0.15 else r.uniform(60, 160)) * S)
            d.rectangle([cx, y, cx + w_, y + 9 * S], fill=(95, 90, 84)); y += 22 * S
    a = np.asarray(img).astype(np.float32)[..., ::-1] / 255
    n = cv2.resize(fbm(ph // 16, pw // 16, 4, 77, 6), (pw, ph)); a *= (0.90 + 0.14 * n)[..., None]   # aged newsprint
    yy, xx = np.mgrid[0:ph, 0:pw].astype(np.float32)
    a *= (1 - 0.18 * ss(0.25, 0.0, np.minimum(np.minimum(xx, pw - xx), np.minimum(yy, ph - yy)) / pw))[..., None]   # yellowed edges
    a *= (1 - 0.10 * np.exp(-((xx - pw / 2) / (6 * S)) ** 2))[..., None]                            # centre fold
    _DK['paper'] = (np.dstack([a, np.ones((ph, pw), np.float32)]), (px0, py0, pww, phh), S)
    return _DK['paper']


def _halftone(f, cell=13):
    l = np.clip(luma(f), 0, 1); X = (XX + YY) * 0.7071; Y = (YY - XX) * 0.7071
    dx = (X % cell) - cell / 2; dy = (Y % cell) - cell / 2; d = np.sqrt(dx * dx + dy * dy)
    r = np.sqrt(1 - l) * cell * 0.72
    dot = ss(r + 0.8, r - 0.8, d)
    return (1 - dot)[..., None] * np.float32([0.82, 0.88, 0.91]) + dot[..., None] * np.float32([0.10, 0.10, 0.11])


def t_newspaper(t, c, A, B, a, b, sprite=None):
    """A front page spins in (BAY HARBOR BUTCHER STRIKES AGAIN), slaps down, then the camera pushes into its
    halftone photo of the next shot, whose dots dissolve into full colour as it fills the screen."""
    spr, (px0, py0, pww, phh), S = (sprite or _paper_sprite)(); ph, pw = spr.shape[:2]
    tl = c - 0.12; base = 0.5
    if t < tl:
        p = clamp((t - a) / (tl - a)); e = eout(p)
        sc = base * lerp(0.04, 1.0, e); ang = math.radians(-4) + (1 - e) * 2.5 * 2 * math.pi; ctr = np.array([W / 2, H / 2])
        zk = 0.0
    else:
        u = clamp((t - tl) / (b - tl)); k = ease(u); bounce = 1 + 0.04 * math.sin(min(1.0, (t - tl) / 0.10) * math.pi) * (t - tl < 0.10)
        zfull = (W / (pww * base)); sc = base * math.exp(math.log(zfull) * ein(u) ** 0.9) * bounce; ang = math.radians(-4) * (1 - k)
        pc = np.array([px0 + pww / 2, py0 + phh / 2]); zk = k
        ctr = None
    ca, sa = math.cos(ang), math.sin(ang)
    R2 = np.array([[ca, -sa], [sa, ca]]) * sc
    if ctr is not None: tvec = ctr - R2 @ np.array([pw / 2, ph / 2])
    else:                                                                                       # keep the photo centre heading to the screen centre
        rest_c = np.array([W / 2, H / 2]) - R2 @ np.array([pw / 2, ph / 2]) + R2 @ (pc - np.array([pw / 2, ph / 2]))
        p_scr = R2 @ pc + (np.array([W / 2, H / 2]) - R2 @ np.array([pw / 2, ph / 2]))
        tvec = (np.array([W / 2, H / 2]) - R2 @ np.array([pw / 2, ph / 2])) + (np.array([W / 2, H / 2]) - p_scr) * zk
    M = np.float32(np.hstack([R2, tvec[:, None]]))
    fa = layer(A, min(t, c - 1e-3)); bg = blur(fa, 8 * min(1.0, (t - a) / 0.3)) * (1 - 0.45 * min(1.0, (t - a) / 0.3))
    if t < tl and (1 - eout(clamp((t - a) / (tl - a)))) > 0.25:                                     # rotational motion blur while spinning
        acc = None
        for j in range(3):
            aj = ang + j * 0.06; Rj = np.array([[math.cos(aj), -math.sin(aj)], [math.sin(aj), math.cos(aj)]]) * sc
            Mj = np.float32(np.hstack([Rj, (ctr - Rj @ np.array([pw / 2, ph / 2]))[:, None]]))
            lj = cv2.warpAffine(spr, Mj, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            acc = lj if acc is None else acc + lj
        lay = acc / 3
    else:
        lay = cv2.warpAffine(spr, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    sh = cv2.GaussianBlur(np.roll(np.roll(lay[..., 3], 24, 0), 16, 1), (0, 0), 18) * 0.6
    f = bg * (1 - sh[..., None]) * (1 - lay[..., 3:4]) + lay[..., :3] * lay[..., 3:4]
    corners = np.array([[px0, py0], [px0 + pww, py0], [px0 + pww, py0 + phh], [px0, py0 + phh]], np.float64)
    scr = corners @ M[:, :2].T + M[:, 2]
    P = cv2.getPerspectiveTransform(np.float32([[0, 0], [W, 0], [W, H], [0, H]]), np.float32(scr))
    fb = layer(B, t); photo = _halftone(fb) * (1 - zk) + fb * zk
    win = cv2.warpPerspective(photo, P, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    wm = cv2.warpPerspective(np.ones((H, W), np.float32), P, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)[..., None]
    if t < tl and (1 - eout(clamp((t - a) / (tl - a)))) > 0.25: wm = wm * 0
    f = f * (1 - wm) + win * wm
    return f + 0.25 * max(0.0, 1 - abs(t - tl) / 0.05)                                          # slap flash


# ============================================================ Dexter kit IV: "X Gon' Give It To Ya" (DMX)
# Street-crime objects cut to the hook's words: subwoofer (the bass), duct-tape rolls ("roll"), mugshot flash ("pop"),
# knife-carved X ("X gon'"), handcuffs, luminol, a print that rolls itself up ("roll"), a bare bulb that pops ("pop"),
# a spray-painted X, and a vinyl record that scratches back to the start (the loop).
_K4 = {}
RFULL = 0.5 * math.hypot(W, H) * 1.02
_L4 = np.float32([-0.42, -0.62, 0.66]); _L4 /= np.linalg.norm(_L4)
_H4 = _L4 + np.float32([0, 0, 1]); _H4 /= np.linalg.norm(_H4)


def _shade(nx, ny, nz, base, ks=0.8, pw=40, amb=0.28):
    """Blinn-shaded luminance for a surface normal field (light from the upper left)."""
    d = np.clip(nx * _L4[0] + ny * _L4[1] + nz * _L4[2], 0, 1)
    s = np.clip(nx * _H4[0] + ny * _H4[1] + nz * _H4[2], 0, 1) ** pw
    return (base * (amb + (1 - amb) * d) + ks * s).astype(np.float32)


def _nz(tilt): return np.sqrt(np.clip(1 - tilt * tilt, 0, 1))


def _live_circle(f, live, cx, cy, Rs, ang=0.0, feather=1.5, src=None):
    """Paste a live frame into the circle (cx, cy, Rs), rotated by ang degrees; at Rs = RFULL centred on screen it
    is the frame 1:1, so a dive into the circle lands exactly on the next shot."""
    Rs = min(Rs, RFULL)
    sc = Rs / RFULL * lerp(RFULL / (W / 2), 1.0, clamp(Rs / RFULL) ** 2)
    sx, sy = src if src is not None else (W / 2, H / 2)
    M = cv2.getRotationMatrix2D((sx, sy), ang, sc); M[0, 2] += cx - sx; M[1, 2] += cy - sy
    lv = cv2.warpAffine(live, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    rr = np.sqrt((XX - cx) ** 2 + (YY - cy) ** 2); m = ss(Rs + feather, Rs - feather, rr)[..., None]
    return f * (1 - m) + lv * m, rr


def _rgba(rgb, al):
    al = np.clip(al, 0, 1).astype(np.float32)
    return np.dstack([rgb * al[..., None], al]).astype(np.float32)


def _polar_text(txt, font, size, r0, r1, D, c0, fg, bg, reps=1):
    """Text running round a ring (r0..r1) of a D x D sprite centred at c0: returns (rgb, mask)."""
    from PIL import Image, ImageDraw
    L = int(2 * math.pi * (r0 + r1) / 2); bh = int(r1 - r0)
    img = Image.new('RGB', (L, bh), tuple(int(255 * v) for v in bg[::-1])); d = ImageDraw.Draw(img)
    fnt = _font_px(font, size); s = (txt + '   ') * reps; tw = d.textlength(s, font=fnt)
    x = 0
    while x < L: d.text((x, (bh - size) / 2 - 2), s, font=fnt, fill=tuple(int(255 * v) for v in fg[::-1])); x += tw
    tex = np.asarray(img).astype(np.float32)[..., ::-1] / 255
    Y, X = np.mgrid[0:D, 0:D].astype(np.float32) - c0
    r = np.sqrt(X * X + Y * Y); th = np.arctan2(Y, X)
    mx = ((th + math.pi) / (2 * math.pi) * L).astype(np.float32); my = ((r1 - r) / (r1 - r0) * bh).astype(np.float32)
    rgb = cv2.remap(tex, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return rgb, ss(r1 + 0.8, r1 - 0.8, r) * ss(r0 - 0.8, r0 + 0.8, r)


# ---- subwoofer (opener: the bass)
def _sub_assets():
    if 'sub' in _K4: return _K4['sub']
    from PIL import Image, ImageDraw
    D = 1000; R = 480.0; c0 = D / 2
    Y, X = np.mgrid[0:D, 0:D].astype(np.float32) - c0
    r = np.sqrt(X * X + Y * Y) + 1e-3; ux, uy = X / r, Y / r
    rs0, rc0, rd0 = 0.86 * R, 0.70 * R, 0.30 * R
    n = cv2.resize(fbm(D // 8, D // 8, 4, 63, 4), (D, D), interpolation=cv2.INTER_CUBIC)
    # gunmetal basket frame, bevelled at both edges, spun-metal rings
    tilt = 0.8 * (ss(R - 16, R, r) - ss(rs0 + 16, rs0, r))
    fr = _shade(ux * tilt, uy * tilt, _nz(tilt), 0.30, ks=0.55, pw=30) * (1 + 0.05 * np.sin(r * 0.9) + 0.04 * (n - 0.5))
    bolt_v = np.zeros_like(r); bolt_m = np.zeros_like(r)
    for k in range(8):                                                                        # 8 mounting bolts
        ang = math.radians(22.5 + 45 * k); bx, by = 0.93 * R * math.cos(ang), 0.93 * R * math.sin(ang)
        dx, dy = X - bx, Y - by; dd = np.sqrt(dx * dx + dy * dy) + 1e-3; q = np.clip(dd / 17, 0, 1) * 0.9
        m = ss(17.5, 16, dd)
        sh = _shade(dx / dd * q, dy / dd * q, _nz(q), 0.60, ks=1.0, pw=50) * (1 - 0.8 * ss(6.5, 5, np.maximum(np.abs(dx), np.abs(dy))))
        bolt_v = bolt_v * (1 - m) + sh * m; bolt_m = np.maximum(bolt_m, m)
    fr = fr * (1 - bolt_m) + bolt_v * bolt_m
    m_fr = ss(R + 1, R - 1, r) * ss(rs0 - 1, rs0 + 1, r)
    sr = np.clip((r - (rs0 + rc0) / 2) / ((rs0 - rc0) / 2), -1, 1) * 0.92                 # rubber surround (half-round)
    sur = _shade(ux * sr, uy * sr, _nz(sr), 0.15, ks=0.22, pw=14) * (0.95 + 0.1 * (n - 0.5))
    m_sur = ss(rs0 + 1, rs0 - 1, r) * ss(rc0 - 1, rc0 + 1, r)
    cone = _shade(ux * -0.3, uy * -0.3, _nz(0.3) + 0 * r, 0.12, ks=0.07, pw=10)             # pressed-paper cone
    cone = cone * (1 + 0.07 * np.sin(r * 0.30)) * (0.92 + 0.16 * (n - 0.5)) * (0.6 + 0.4 * ss(rc0, rc0 - 45, r))
    m_cone = ss(rc0 + 1, rc0 - 1, r) * ss(rd0 - 1, rd0 + 1, r)
    q = np.clip(r / rd0, 0, 1) * 0.85                                                        # glossy dust cap with a red X
    cap = _shade(ux * q, uy * q, _nz(q), 0.08, ks=1.1, pw=60)
    xm = np.zeros((D, D), np.float32); e = 0.46 * rd0
    cv2.line(xm, (int(c0 - e), int(c0 - e)), (int(c0 + e), int(c0 + e)), 1.0, 34, cv2.LINE_AA)
    cv2.line(xm, (int(c0 + e), int(c0 - e)), (int(c0 - e), int(c0 + e)), 1.0, 34, cv2.LINE_AA)
    xm = xm * ss(rd0, rd0 - 6, r)
    m_cap = ss(rd0 + 1, rd0 - 1, r)
    cool = np.float32([1.04, 1.0, 0.97])
    st_rgb = (fr * m_fr + sur * m_sur)[..., None] * cool; st_al = m_fr + m_sur
    img = Image.new('RGBA', (260, 50), (0, 0, 0, 0)); d = ImageDraw.Draw(img)                # model plate on the frame
    d.rounded_rectangle([0, 0, 259, 49], radius=10, fill=(196, 198, 204, 255)); d.rounded_rectangle([3, 3, 256, 46], radius=8, outline=(120, 122, 128, 255), width=2)
    fnt = _font_px(MONO, 26); tw = d.textlength('XG-15  2000W', font=fnt); d.text(((260 - tw) / 2, 9), 'XG-15  2000W', font=fnt, fill=(24, 24, 28, 255))
    pl = np.asarray(img).astype(np.float32) / 255; py0, px0 = int(c0 + 0.93 * R - 25), int(c0 - 130)
    pa = pl[..., 3]; st_rgb[py0:py0 + 50, px0:px0 + 260] = st_rgb[py0:py0 + 50, px0:px0 + 260] * (1 - pa[..., None]) + pl[..., 2::-1] * pa[..., None]
    red = np.float32([0.10, 0.08, 0.80])
    cap_rgb = cap[..., None] * cool * (1 - xm[..., None]) + np.clip(red * (0.55 + 3.0 * cap[..., None]), 0, 1.3) * xm[..., None]
    mv_rgb = cone[..., None] * cool * m_cone[..., None] + cap_rgb * m_cap[..., None]
    _K4['sub'] = (np.dstack([st_rgb, st_al]).astype(np.float32), np.dstack([mv_rgb, m_cone + m_cap]).astype(np.float32), R)
    return _K4['sub']


def t_sub(t, c, A, B, a, b):
    """A subwoofer rises in front of the shot and pumps on the beat (the frame shakes, a pressure ring runs out);
    the last kick blows the X dust cap out and the camera dives through the cone into the next shot."""
    st, mv, R = _sub_assets(); D = st.shape[0]
    rise = eout(clamp((t - a) / 0.22))
    pump = 0.0
    for tp, amp in ((c - HALF, 0.6), (c, 1.0)):
        if t >= tp - 0.03: dt = t - tp; pump += amp * clamp((dt + 0.03) / 0.03) * math.exp(-max(0.0, dt) * 9)
    base = 0.92 * W / D
    cx0, cy0 = W / 2, lerp(H * 1.30, H * 0.44, rise)
    u = clamp((t - c) / (b - c)); zmax = RFULL / (0.30 * R * base)
    z = math.exp(math.log(zmax) * ein(u) ** 0.8) if t >= c else 1.0
    cx, cy = lerp(cx0, W / 2, ease(u)), lerp(cy0, H / 2, ease(u))
    s = base * z; sp = s * (1 + 0.06 * pump)
    bg = layer(A, min(t, c - 1e-3))
    if pump > 0.02: bg = zoom(bg, 1 + 0.035 * pump, cy=cy0)
    for tp in (c - HALF, c):                                                                   # pressure rings
        dt = t - tp
        if 0 <= dt < 0.3:
            rr0 = np.sqrt((XX - cx) ** 2 + (YY - cy) ** 2) + 1e-3; rw = 300 * s / base + dt * 2600
            dd = (22 * np.exp(-((rr0 - rw) / 60) ** 2) * (1 - dt / 0.3)).astype(np.float32)
            bg = cv2.remap(bg, XX - dd * (XX - cx) / rr0, YY - dd * (YY - cy) / rr0, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    bg = blur(bg, 7 * rise) * (1 - 0.5 * rise)
    M = np.float32([[s, 0, cx - s * D / 2], [0, s, cy - s * D / 2]])
    Mm = np.float32([[sp, 0, cx - sp * D / 2], [0, sp, cy - sp * D / 2]])
    lay = cv2.warpAffine(st, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    lm = cv2.warpAffine(mv, Mm, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    sh = cv2.GaussianBlur(np.roll(np.roll(np.maximum(lay[..., 3], lm[..., 3]), 34, 0), 22, 1), (0, 0), 26) * 0.65
    f = bg * (1 - sh[..., None])
    f = over(f, lay); f = over(f, lm * np.float32([1, 1, 1, 1]) + np.float32([0.05, 0.05, 0.05, 0]) * pump * lm[..., 3:4])
    if t >= c - 0.02:                                                                         # the cap blows out: the next shot bursts from the cone
        burst = eout(clamp((t - c + 0.02) / 0.10))
        Rs = 0.30 * R * sp * (1 + 0.55 * burst * (1 - u))
        f, rr = _live_circle(f, layer(B, t), cx, cy, Rs)
        f = f + (np.exp(-((rr - Rs) / 14) ** 2) * 0.5 * (1 - u))[..., None]
        f = f + 0.35 * math.exp(-max(0.0, t - c) * 12)
    return f


# ---- duct tape ("then we gonna roll")
def _tape_assets():
    if 'tape4' in _K4: return _K4['tape4']
    hs = H // 3 + 36; L = W + 40
    yy, xx = np.mgrid[0:hs, 0:L].astype(np.float32)
    wr = cv2.resize(fbm(8, 40, 4, 91, 4), (L, hs), interpolation=cv2.INTER_CUBIC)
    rng = np.random.default_rng(91)
    weave = 0.035 * np.sin(xx * 1.7) * np.sin(yy * 1.7) + 0.045 * np.sin(yy * 0.75 + 2 * wr)     # cloth scrim and threads
    speck = cv2.GaussianBlur(rng.random((hs, L)).astype(np.float32), (0, 0), 0.8) - 0.5
    crease = np.zeros((hs, L), np.float32)
    for k in range(70):                                                                       # sharp crinkle ridges
        x0, y0 = rng.uniform(0, L), rng.uniform(0, hs); ang = rng.normal(math.radians(70), 0.5) * rng.choice([-1, 1]); ln = rng.uniform(40, 160)
        dx_, dy_ = math.cos(ang), math.sin(ang)
        d_ = (xx - x0) * dy_ - (yy - y0) * dx_; al_ = np.abs((xx - x0) * dx_ + (yy - y0) * dy_) < ln
        crease += (np.exp(-((d_ - 1.5) / 1.2) ** 2) * 0.22 - np.exp(-((d_ + 1.5) / 1.6) ** 2) * 0.18) * al_ * rng.uniform(0.5, 1.0)
    v = 0.56 + 0.12 * (wr - 0.5) + weave + 0.10 * speck + crease
    spec = 0.30 * np.exp(-((yy / hs - 0.27 - 0.06 * (wr - 0.5)) / 0.035) ** 2) + 0.10 * np.exp(-((yy / hs - 0.72) / 0.05) ** 2)
    v = (v + spec * (1 + 2.5 * np.clip(crease, 0, None))) * (0.62 + 0.38 * ss(0, 6, np.minimum(yy, hs - 1 - yy)))
    strip = (v[..., None] * np.float32([1.03, 1.0, 0.96])).astype(np.float32)
    Ro = hs / 2; Dd = int(2 * Ro + 8); c0 = Dd / 2
    Y, X = np.mgrid[0:Dd, 0:Dd].astype(np.float32) - c0; r = np.sqrt(X * X + Y * Y)
    Rc1, Rc0 = 0.60 * Ro, 0.49 * Ro
    m_ring = ss(Ro + 1, Ro - 1, r) * ss(Rc1 - 1, Rc1 + 1, r)
    m_core = ss(Rc1 + 1, Rc1 - 1, r) * ss(Rc0 - 1, Rc0 + 1, r)
    ring = (0.56 + 0.05 * np.sin(r * 1.7) + 0.04 * np.sin(r * 0.31)) * (0.75 + 0.25 * ss(Ro, Ro - 12, r)) * (0.8 + 0.2 * ss(Rc1, Rc1 + 10, r))
    ring = ring * (1 - 0.35 * ss(9, 3, np.abs(Y)) * (X > Ro - 16))                            # the tape's end on the outer wrap
    rgb = ring[..., None] * np.float32([1.03, 1.0, 0.96]) * m_ring[..., None]
    trgb, tm = _polar_text('DUCT TAPE  48mm x 55m', INTER_BLACK, 21, Rc0 + 3, Rc1 - 3, Dd, c0, (0.18, 0.13, 0.10), (0.42, 0.60, 0.74), reps=2)
    card = np.float32([0.42, 0.60, 0.74]) * (0.85 + 0.15 * (0.5 + 0.5 * np.sin(r * 2.3)))[..., None]
    card = card * (1 - tm[..., None]) + trgb * tm[..., None]
    rgb = rgb + card * m_core[..., None]
    _K4['tape4'] = (strip, _rgba(rgb / np.maximum(m_ring + m_core, 1e-3)[..., None], m_ring + m_core), Ro)
    return _K4['tape4']


def t_ducttape(t, c, A, B, a, b):
    """Three rolls of duct tape roll across the frame (alternating directions), laying silver strips until the shot
    is taped shut; on the hit the strips are ripped off one by one onto the next shot."""
    strip, roll, Ro = _tape_assets(); hs = strip.shape[0]; Dd = roll.shape[0]
    if t < c: under = layer(A, t)
    else: under = layer(B, t, 1 + 0.07 * (1 - eout((t - c) / (b - c))))
    rows = ((H / 6, 1, 0.00, 0), (H / 2, -1, 0.05, 2), (5 * H / 6, 1, 0.025, 1))
    covm = np.zeros((H, W), np.float32); col = np.zeros((H, W, 3), np.float32); rolls = []
    for yc, d, dl, order in rows:
        t0, t1 = a + dl, c - 0.05
        p = clamp((t - t0) / (t1 - t0)); xf = lerp(-Ro, W + Ro, ease(p) ** 0.9)
        dx = dy = 0.0
        if t >= c:
            uk = clamp((t - c - 0.035 * order) / (b - c - 0.08)); dx = d * 1.5 * W * ein(uk); dy = -0.10 * H * ein(uk)
            if uk >= 1: continue
        tex = strip if d > 0 else strip[:, ::-1]
        Mt = np.float32([[1, 0, dx - 20], [0, 1, yc - hs / 2 + dy]])
        cr = cv2.warpAffine(tex, Mt, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        my = ss(yc - hs / 2 + dy - 1, yc - hs / 2 + dy + 1, YY) * ss(yc + hs / 2 + dy + 1, yc + hs / 2 + dy - 1, YY)
        xr = xf if d > 0 else W - xf
        if t < c: mx = ss(xr + 1.5, xr - 1.5, XX) if d > 0 else ss(xr - 1.5, xr + 1.5, XX)
        else: mx = ss(dx - 1.5, dx + 1.5, XX) * ss(W + dx + 1.5, W + dx - 1.5, XX)
        m = my * mx
        if t >= c and abs(dx) > 30:                                                        # motion blur on the rip
            kb = int(min(91, abs(dx) / 12)) | 1; cr = cv2.blur(cr, (kb, 1)); m = cv2.blur(m, (kb, 1))
        col = col * (1 - m[..., None]) + cr * m[..., None]; covm = np.maximum(covm, m)
        if t < c and 0 < p < 1: rolls.append((xr, yc, -d * math.degrees((xf + Ro) / Ro)))
    sh = cv2.GaussianBlur(np.roll(np.roll(covm, 16, 0), 10, 1), (0, 0), 14) * 0.5
    f = under * (1 - sh[..., None]) * (1 - covm[..., None]) + col
    for xr, yc, deg in rolls:
        M = cv2.getRotationMatrix2D((Dd / 2, Dd / 2), deg, 1.0); M[0, 2] += xr - Dd / 2; M[1, 2] += yc - Dd / 2
        lr = cv2.warpAffine(roll, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        rs = cv2.GaussianBlur(np.roll(np.roll(lr[..., 3], 22, 0), 14, 1), (0, 0), 16) * 0.55
        f = over(f * (1 - rs[..., None]), lr)
    if t >= c: f = f + 0.18 * (1 - eout((t - c) / 0.10))
    return f


# ---- mugshot ("then we let it pop")
def _mug_assets():
    if 'mug' in _K4: return _K4['mug']
    from PIL import Image, ImageDraw
    img = Image.new('L', (W, H), 0); d = ImageDraw.Draw(img); lines = Image.new('L', (W, H), 0); dl = ImageDraw.Draw(lines)
    top, bot = 0.07 * H, 0.90 * H; px_in = (bot - top) / 36; fnt = _font_px(INTER_BLACK, 40)
    for k in range(37):
        y = top + k * px_in; hin = 84 - k; major = hin % 6 == 0
        Lt = 74 if major else (44 if hin % 2 == 0 else 26)
        d.line([(0, y), (Lt, y)], fill=235, width=5 if major else 3); d.line([(W - Lt, y), (W, y)], fill=235, width=5 if major else 3)
        if major:
            lab = "%d'%d\"" % (hin // 12, hin % 12); tw = d.textlength(lab, font=fnt)
            d.text((84, y - 24), lab, font=fnt, fill=245); d.text((W - 84 - tw, y - 24), lab, font=fnt, fill=245)
            dl.line([(0, y), (W, y)], fill=255, width=3)
    face = np.exp(-(((XX - W / 2) / (0.30 * W)) ** 2 + ((YY - 0.40 * H) / (0.24 * H)) ** 2) ** 2)
    chart = np.maximum(np.asarray(img, np.float32) / 255, np.asarray(lines, np.float32) / 255 * 0.22 * (1 - 0.92 * face))
    pw_, ph_ = 700, 300
    pc = Image.new('RGBA', (pw_, ph_), (0, 0, 0, 0)); d = ImageDraw.Draw(pc)
    d.rounded_rectangle([0, 0, pw_ - 1, ph_ - 1], radius=14, fill=(20, 20, 23, 255)); d.rounded_rectangle([10, 10, pw_ - 11, ph_ - 11], radius=10, outline=(225, 222, 214, 255), width=3)
    for txt, fn, y in (('MIAMI METRO POLICE', _font_px(INTER_SEMI, 34), 30), ('MORGAN, DEXTER', _font_px(INTER_BLACK, 70), 82)):
        tw = d.textlength(txt, font=fn); d.text(((pw_ - tw) / 2, y), txt, font=fn, fill=(236, 233, 225, 255))
    fm = _font_px(MONO, 42); d.text((44, 200), '0915-07', font=fm, fill=(236, 233, 225, 255))
    tw = d.textlength('11 15 08', font=fm); d.text((pw_ - 44 - tw, 200), '11 15 08', font=fm, fill=(236, 233, 225, 255))
    a = np.asarray(pc).astype(np.float32) / 255
    _K4['mug'] = (chart.astype(np.float32), _rgba(a[..., 2::-1].copy(), a[..., 3]))
    return _K4['mug']


def t_mugshot(t, c, A, B, a, b):
    """A booking camera pre-flashes, then POPs; out of the flash comes the next shot as a mugshot (harsh flash light,
    height chart on the wall, name board pushed up into frame), which settles back into the edit."""
    chart, board = _mug_assets(); tf = c - 0.04
    if t < c:
        f = layer(A, t, 1 + 0.04 * clamp((t - a) / (c - a)))
    else:
        u = clamp((t - c) / (b - c)); fb = layer(B, t, 1.0 + 0.05 * eout(u))
        k = math.exp(-(t - c) * 3.5); l = luma(fb)[..., None]
        harsh = np.clip((l + (fb - l) * 0.35 - 0.45) * 1.35 + 0.52, 0, 1.2)                   # bleached flash look
        fb = fb * (1 - 0.85 * k) + harsh * 0.85 * k
        oa = 1 - ease(clamp((t - (b - 0.14)) / 0.14))
        f = fb * (1 - chart[..., None] * oa) + chart[..., None] * oa * 0.96
        bh_, bw_ = board.shape[:2]
        y0 = lerp(H + 10, H * 0.955 - bh_, eout(clamp((t - c) / 0.12))) + (H * 0.30) * ein(clamp((t - (b - 0.16)) / 0.16))
        Mb = np.float32([[1, 0, (W - bw_) / 2], [0, 1, y0]])
        lb = cv2.warpAffine(board, Mb, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        bs = cv2.GaussianBlur(np.roll(lb[..., 3], -18, 0), (0, 0), 20) * 0.5
        f = over(f * (1 - bs[..., None]), lb)
    fl = 0.35 * math.exp(-((t - (tf - 0.17)) / 0.025) ** 2)                                    # pre-flash
    if t >= tf: fl += 1.5 * math.exp(-(t - tf) * 8)
    if fl > 0.01:
        k = min(0.66, 0.6 * fl); f = f * (1 - k) + k * 0.97                                       # blown out, but the image still shows through
        f = f + cv2.resize(cv2.GaussianBlur(np.clip(cv2.resize(f, (W // 8, H // 8)) - 0.8, 0, 2), (0, 0), 6), (W, H)) * 0.3 * min(1.0, fl)
    return f


# ---- knife X ("X gon' give it to ya")
def t_xslash(t, c, A, B, a, b):
    """X gon' give it to ya: a blade carves a giant X into the frame in two slashes (white-hot cuts cooling into
    bleeding gashes), then on the hit the four wedges of the shot fall away onto the next shot."""
    hyp = math.hypot(W, H)
    S1 = (YY * W - XX * H) / hyp; S2 = (XX * H + YY * W - W * H) / hyp                        # signed distance to the diagonals
    T1 = (XX * W + YY * H) / hyp; T2 = ((W - XX) * W + YY * H) / hyp                          # position along them
    slashes = ((c - 0.30, c - 0.21, S1, T1), (c - 0.16, c - 0.07, S2, T2))
    if t < c:
        f = layer(A, t)
        for j, (ts, te, S, T) in enumerate(slashes):
            if t < ts: continue
            p = eout(clamp((t - ts) / (te - ts))) * hyp; dist = np.abs(S); age = max(0.0, t - te)
            along = ss(p + 2, p - 2, T)
            heat = np.exp(-((T - p) / 90) ** 2) * (t < te) + math.exp(-age * 9)
            gash = ss(5.5, 2.5, dist) * along
            f = f * (1 - gash[..., None]) + np.float32([0.02, 0.02, 0.22]) * gash[..., None]
            f = f + (np.exp(-(dist / 9) ** 2) * along * heat * 1.1)[..., None] * np.float32([0.75, 0.90, 1.0])
            f = f + (np.exp(-(dist / 30) ** 2) * along * heat * 0.25)[..., None] * np.float32([0.4, 0.6, 1.0])
            if t < te:                                                                       # glint at the blade's tip
                hx = (p / hyp) * (W if j == 0 else -W) + (0 if j == 0 else W); hy = (p / hyp) * H
                dh = np.sqrt((XX - hx) ** 2 + (YY - hy) ** 2)
                g = np.exp(-(dh / 22) ** 2) * 1.6 + np.exp(-(np.abs(XX - hx) / 3) - (np.abs(YY - hy) / 60)) * 0.6 + np.exp(-(np.abs(YY - hy) / 3) - (np.abs(XX - hx) / 60)) * 0.6
                f = f + g[..., None]
            if age > 0:                                                                      # blood beads run from the cut
                dm = np.zeros((H, W), np.float32); rng = np.random.default_rng(31 + j)
                for k in range(9):
                    s_ = rng.uniform(0.08, 0.92); x0 = s_ * W if j == 0 else (1 - s_) * W; y0 = s_ * H
                    L = min(260.0, 2200 * (age * rng.uniform(0.6, 1.2)) ** 2)
                    cv2.line(dm, (int(x0), int(y0)), (int(x0), int(y0 + L)), 1.0, int(rng.integers(5, 9)), cv2.LINE_AA)
                    cv2.circle(dm, (int(x0), int(y0 + L)), int(rng.integers(6, 10)), 1.0, -1, cv2.LINE_AA)
                f = f * (1 - dm[..., None]) + BLOOD * dm[..., None]
        return f
    u = clamp((t - c) / (b - c))
    fb = layer(B, t, 1 + 0.12 * (1 - eout(u)))
    fa = layer(A, c - 1e-3)
    rim = np.exp(-(np.minimum(np.abs(S1), np.abs(S2)) / 5) ** 2)[..., None]
    fa = fa * (1 - 0.85 * rim) + BLOOD * 0.85 * rim
    m1, m2 = ss(-1, 1, S1), ss(-1, 1, S2)
    pieces = (((1 - m1) * (1 - m2), (0, -1), (W / 2, H * 0.22), 9), ((1 - m1) * m2, (1, 0), (W * 0.80, H / 2), -7),
              (m1 * m2, (0, 1), (W / 2, H * 0.78), -9), (m1 * (1 - m2), (-1, 0), (W * 0.20, H / 2), 7))
    dist = 10 + 1700 * u ** 1.7
    acc = np.zeros((H, W, 3), np.float32); am = np.zeros((H, W), np.float32)
    for m, (dx, dy), (px, py), rot in pieces:
        M = cv2.getRotationMatrix2D((px, py), rot * u, 1.0); M[0, 2] += dx * dist; M[1, 2] += dy * dist
        pc = cv2.warpAffine(fa * m[..., None], M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        pm = cv2.warpAffine(m, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        acc = acc * (1 - pm[..., None]) + pc; am = np.maximum(am, pm)
    sh = cv2.GaussianBlur(np.roll(np.roll(am, 22, 0), 14, 1), (0, 0), 18) * 0.6
    f = fb * (1 - sh[..., None]) * (1 - am[..., None]) + acc
    return f + 0.30 * (1 - eout(u * 4))


# ---- handcuffs
_CUFF = dict(Rm=300.0, hb=24.0, hs=15.0, piv=math.radians(300), lock=math.radians(100))


def _draw_cuffs(f, Cx, Cy, z, psi, beta):
    """Steel handcuff (bow, swinging toothed strand, lock box, chain) drawn analytically at any zoom.
    Returns the frame and the ring's inner radius on screen."""
    g = _CUFF; Rm, hb, hs = g['Rm'], g['hb'], g['hs']
    cp, sp_ = math.cos(-psi), math.sin(-psi)
    lx0 = (XX - Cx) / z; ly0 = (YY - Cy) / z
    lx = lx0 * cp - ly0 * sp_; ly = lx0 * sp_ + ly0 * cp                                         # undo the cuff's own rotation
    rr = np.sqrt(lx * lx + ly * ly) + 1e-3; ph = np.mod(np.arctan2(ly, lx), 2 * math.pi)
    out = np.zeros((H, W), np.float32); al = np.zeros((H, W), np.float32)
    # bow: double bar from the lock (100 deg) round the left to the pivot (300 deg)
    s = (rr - Rm) / hb; inb = ss(g['lock'] - 0.02, g['lock'] + 0.02, ph) * ss(g['piv'] + 0.02, g['piv'] - 0.02, ph)
    mb = ss(1.04, 0.96, np.abs(s)) * inb
    tilt = np.sign(s) * ss(0.55, 1.0, np.abs(s)) * 0.9
    v = _shade(lx / rr * tilt, ly / rr * tilt, _nz(tilt), 0.62, ks=1.0, pw=45) * (1 - 0.55 * np.exp(-(s / 0.10) ** 2))
    out = out * (1 - mb) + v * mb; al = np.maximum(al, mb)
    # strand: rotated about the pivot by beta (open) -> closed pose coordinates
    Px, Py = Rm * math.cos(g['piv']), Rm * math.sin(g['piv']); cb, sb = math.cos(-beta), math.sin(-beta)
    qx = (lx - Px) * cb - (ly - Py) * sb + Px; qy = (lx - Px) * sb + (ly - Py) * cb + Py
    rq = np.sqrt(qx * qx + qy * qy) + 1e-3; pq = np.mod(np.arctan2(qy, qx) - g['piv'], 2 * math.pi)
    sq = (rq - Rm) / hs; span = (2 * math.pi - g['piv'] + g['lock']) - 0.06
    ms = ss(1.04, 0.96, np.abs(sq)) * ss(span + 0.02, span - 0.02, pq)
    v = _shade(qx / rq * sq * 0.9, qy / rq * sq * 0.9, _nz(np.clip(sq, -1, 1) * 0.9), 0.66, ks=1.1, pw=55)
    teeth = (np.sin(pq * Rm / 6.5) > 0.2) & (sq > 0.35)
    v = v * (1 - 0.55 * teeth)
    out = out * (1 - ms) + v * ms; al = np.maximum(al, ms)
    # pivot rivet
    dpx, dpy = lx - Px, ly - Py; dp = np.sqrt(dpx * dpx + dpy * dpy) + 1e-3; qq = np.clip(dp / 21, 0, 1) * 0.85
    mr = ss(21.5, 20, dp); v = _shade(dpx / dp * qq, dpy / dp * qq, _nz(qq), 0.7, ks=1.2, pw=60)
    out = out * (1 - mr) + v * mr; al = np.maximum(al, mr)
    # lock box at the end of the bow
    Lx, Ly = (Rm + 4) * math.cos(g['lock']), (Rm + 4) * math.sin(g['lock'])
    tx, ty = -math.sin(g['lock']), math.cos(g['lock'])
    bu = (lx - Lx) * tx + (ly - Ly) * ty + 22; bv = (lx - Lx) * math.cos(g['lock']) + (ly - Ly) * math.sin(g['lock'])
    eb = np.minimum(70 - np.abs(bu), 48 - np.abs(bv)); mbx = ss(-0.8, 0.8, eb)
    bev = 1 - ss(0, 7, eb); nxb = np.sign(bu) * bev * 0.6 * (np.abs(bu) / 70 > np.abs(bv) / 48); nyb = np.sign(bv) * bev * 0.6 * (np.abs(bu) / 70 <= np.abs(bv) / 48)
    v = _shade(nxb * tx + nyb * math.cos(g['lock']), nxb * ty + nyb * math.sin(g['lock']), _nz(np.maximum(np.abs(nxb), np.abs(nyb))), 0.58, ks=0.7, pw=30)
    key = ss(11, 9, np.sqrt(bu ** 2 + (bv + 6) ** 2)) + ss(4, 3, np.abs(bu)) * ss(22, 20, np.abs(bv - 10)) * (bv > 0)
    v = v * (1 - 0.9 * np.clip(key, 0, 1))
    for rx in (-48, 48): v = v * (1 - 0.5 * ss(6, 4.5, np.sqrt((bu - rx) ** 2 + (bv + 26) ** 2)))
    out = out * (1 - mbx) + v * mbx; al = np.maximum(al, mbx)
    # chain off the bow to the left (alternating face-on and edge-on links)
    a0 = math.radians(195); dx_, dy_ = math.cos(a0), math.sin(a0)
    for k in range(7):
        cxk, cyk = (Rm + hb + 34 + 62 * k) * dx_, (Rm + hb + 34 + 62 * k) * dy_ + 2.5 * k * k
        ux_ = (lx - cxk) * dx_ + (ly - cyk) * dy_; uy_ = -(lx - cxk) * dy_ + (ly - cyk) * dx_
        if k % 2 == 0:
            e = np.sqrt((ux_ / 40) ** 2 + (uy_ / 24) ** 2); s2 = (e - 1) * 24 / 8
            ml = ss(1.05, 0.95, np.abs(s2)); nn = np.clip(s2, -1, 1) * 0.9
            v = _shade(ux_ / (np.abs(ux_) + np.abs(uy_) + 1) * nn, uy_ / (np.abs(ux_) + np.abs(uy_) + 1) * nn, _nz(nn), 0.6, ks=1.0, pw=40)
        else:
            s2 = uy_ / 8; ml = ss(1.05, 0.95, np.abs(s2)) * ss(42, 38, np.abs(ux_)); nn = np.clip(s2, -1, 1) * 0.9
            v = _shade(0 * nn, nn, _nz(nn), 0.6, ks=1.0, pw=40)
        out = out * (1 - ml) + v * ml; al = np.maximum(al, ml)
    sh = cv2.GaussianBlur(np.roll(np.roll(al, int(14 * z), 0), int(10 * z), 1), (0, 0), 10 * z) * 0.55
    f = f * (1 - sh[..., None])
    rgb = out[..., None] * np.float32([1.04, 1.0, 0.96])
    return f * (1 - al[..., None]) + rgb * al[..., None], (Rm - hb) * z


def t_cuffs(t, c, A, B, a, b):
    """A pair of handcuffs swings in round the face, the toothed strand ratchets shut (the world outside goes dark),
    and on the hit the camera dives through the locked cuff into the next shot."""
    sw = eout(clamp((t - a) / 0.20))
    Cx0 = lerp(-0.85 * W, W / 2, sw); Cy0 = 0.41 * H
    psi = math.radians(-32) * (1 - sw) + math.radians(4) * math.sin((t - a) * 18) * math.exp(-(t - a) * 7)
    tc = c - 0.10; bq = clamp((t - (c - 0.24)) / (tc - (c - 0.24)))
    beta = math.radians(78) * (1 - eout(bq))
    if beta < math.radians(16): beta = math.radians(4) * math.floor(math.degrees(beta) / 4)   # ratchet clicks
    u = clamp((t - c) / (b - c)); zmax = RFULL / (_CUFF['Rm'] - _CUFF['hb'])
    z = math.exp(math.log(zmax) * ein(u) ** 0.85) if t >= c else 1.0
    Cx, Cy = lerp(Cx0, W / 2, ease(u)), lerp(Cy0, H / 2, ease(u))
    lock = ease(clamp((t - tc) / 0.08))
    fa = layer(A, min(t, c - 1e-3)); rr = np.sqrt((XX - Cx) ** 2 + (YY - Cy) ** 2)
    Ri = (_CUFF['Rm'] - _CUFF['hb']) * z
    f = fa * (1 - 0.55 * lock * ss(Ri, Ri + 40, rr))[..., None]
    if t >= c: f, _ = _live_circle(f, layer(B, t), Cx, Cy, Ri, feather=2.0)
    f, _ = _draw_cuffs(f, Cx, Cy, z, psi, beta)
    if t >= tc: f = f + 0.25 * math.exp(-(t - tc) * 14) * np.exp(-((rr - _CUFF['Rm'] * z) / 30) ** 2)[..., None]
    if t >= c: f = f + 0.25 * math.exp(-(t - c) * 12)
    return f


# ---- luminol
def _lum_assets():
    if 'lum' in _K4: return _K4['lum']
    al = np.zeros((H, W), np.float32); rng = np.random.default_rng(7)
    for k in range(30):                                                                     # cast-off arc across the frame
        v = k / 29; x = W * (0.04 + 0.92 * v); y = H * (0.62 - 0.30 * math.sin(math.pi * v)) + rng.normal(0, 18)
        ang = math.atan2(-0.30 * H * math.pi * math.cos(math.pi * v), 0.92 * W) + rng.normal(0, 0.15)
        _stain(al, x, y, ang, rng.uniform(0.6, 1.5), rng)
    for k in range(22):                                                                     # drops and a pooled impact
        _stain(al, rng.uniform(0.05, 0.95) * W, rng.uniform(0.08, 0.95) * H, rng.uniform(0, 2 * math.pi), rng.uniform(0.6, 2.4), rng)
    for k in range(3):                                                                      # wiped smears
        x0, y0 = rng.uniform(0.1, 0.9) * W, rng.uniform(0.6, 0.9) * H
        cv2.line(al, (int(x0), int(y0)), (int(x0 + rng.normal(0, 160)), int(y0 + rng.uniform(80, 200))), 0.6, int(rng.integers(26, 44)), cv2.LINE_AA)
    al = np.clip(cv2.GaussianBlur(al, (0, 0), 1.6), 0, 1)
    _K4['lum'] = (al, cv2.GaussianBlur(al, (0, 0), 16) * 1.6)
    return _K4['lum']


def t_luminol(t, c, A, B, a, b):
    """The lights go out, a UV torch sweeps the frame and luminol makes hidden blood spatter glow blue where the
    beam passes; on the hit the whole scene flares and the next shot comes back up, the glow fading off it."""
    spat, bloom_ = _lum_assets(); dim = ease(clamp((t - a) / 0.08))
    if t >= c: dim *= 1 - eout(clamp((t - c) / (b - c)))
    ts0 = a + 0.07; v = clamp((t - ts0) / (c - 0.04 - ts0))
    path = lambda v_: (lerp(-0.12 * W, 1.12 * W, ease(v_)), H * (0.66 - 0.30 * math.sin(math.pi * v_)))
    bx, by = path(v); Rb = 0.27 * W
    fr = layer(A, t) if t < c else layer(B, t)
    spot = np.exp(-((XX - bx) ** 2 + (YY - by) ** 2) / (2 * Rb ** 2)) * (t >= ts0) * (t < c)
    q = 4; Xs, Ys = XX[::q, ::q], YY[::q, ::q]; rev = np.zeros_like(Xs)                      # everywhere the beam has been
    for j in range(16):
        px_, py_ = path(v * j / 15); rev = np.maximum(rev, np.exp(-((Xs - px_) ** 2 + (Ys - py_) ** 2) / (2 * (Rb * 0.8) ** 2)))
    rev = cv2.resize(rev, (W, H)) * (t >= ts0)
    violet = np.float32([1.0, 0.30, 0.78]); lumc = np.float32([1.0, 0.80, 0.30])
    l = luma(fr)[..., None]
    f = fr * (1 - 0.80 * dim) * np.float32([1.15, 0.95, 0.9]) ** dim + 0.025 * dim * violet
    f = f + spot[..., None] * dim * (l * violet * 0.95 + 0.06 * violet)
    sx, sy = 1.15 * W, 1.10 * H; vx, vy = bx - sx, by - sy; L2 = vx * vx + vy * vy + 1e-3       # faint beam haze from the torch
    pr = np.clip(((XX - sx) * vx + (YY - sy) * vy) / L2, 0, 1); dd = np.sqrt((XX - sx - pr * vx) ** 2 + (YY - sy - pr * vy) ** 2)
    f = f + (np.exp(-(dd / (40 + Rb * 0.7 * pr)) ** 2) * 0.10 * dim * (t < c))[..., None] * violet
    flare = math.exp(-((t - c) / 0.05) ** 2) * 2.5
    g = (spot * 1.0 + rev * 0.45 + flare) * (dim if t < c else (1 - clamp((t - c) / (b - c))))
    f = f + (spat * g)[..., None] * lumc * 1.5 + (bloom_ * g)[..., None] * lumc * 0.55
    if abs(t - c) < 0.08: f = f + 0.5 * math.exp(-((t - c) / 0.03) ** 2)
    return f


# ---- the print rolls itself up ("then we gonna roll", second time)
def _paper_back():
    if 'pback' in _K4: return _K4['pback']
    from PIL import Image, ImageDraw
    th = 1100; img = Image.new('RGB', (W, th), (236, 230, 214)); d = ImageDraw.Draw(img); fnt = _font_px(INTER_BLACK, 30)
    s = 'MIAMI METRO POLICE DEPT  ·  EVIDENCE PHOTOGRAPH  ·  CASE 0915-07  ·  '
    for k in range(10):
        y = 24 + k * 110; x = -((k * 173) % 600)
        while x < W: d.text((x, y), s, font=fnt, fill=(120, 116, 110) if k % 3 else (170, 40, 40)); x += d.textlength(s, font=fnt)
        d.line([(0, y + 62), (W, y + 62)], fill=(200, 204, 208), width=2)
    a = np.asarray(img).astype(np.float32)[..., ::-1] / 255
    a *= (0.94 + 0.08 * cv2.resize(fbm(th // 16, W // 16, 4, 5, 6), (W, th)))[..., None]
    _K4['pback'] = a.astype(np.float32)
    return _K4['pback']


def t_rollup(t, c, A, B, a, b):
    """The shot is a photo print that rolls itself up from the bottom like a scroll, its printed back turning on the
    roll, uncovering the next shot underneath; the roll snaps off the top on the hit."""
    tex = _paper_back(); th = tex.shape[0]
    e = ease(clamp((t - a) / (b - a))) ** 1.15
    y0, y1 = H + 60, -260
    yc = lerp(y0, y1, e); R = 80 + 0.11 * (y0 - yc)
    fa = layer(A, t); fb = layer(B, t, 1 + 0.05 * (1 - e))
    ma = ss(yc + 1, yc - 1, YY)[..., None]
    sh = (np.exp(-np.clip(YY - (yc + R), 0, None) / 70) * (YY > yc) * 0.55)[..., None]
    ao = (np.exp(-np.clip((yc - R) - YY, 0, None) / 30) * (YY < yc) * 0.6)[..., None]
    f = fb * (1 - sh) * (1 - ma) + fa * (1 - ao) * ma
    r0, r1 = int(max(0, math.floor(yc - R))), int(min(H, math.ceil(yc + R)))
    if r1 > r0:
        ys = np.arange(r0, r1, dtype=np.float32) + 0.5; s_ = np.clip((ys - yc) / R, -1, 1); thv = np.arcsin(s_)
        rows = ((y0 - yc) + R * thv).astype(np.int64) % th
        band = tex[rows]
        nzc = np.sqrt(1 - s_ ** 2); diff = np.clip(-0.45 * s_ + 0.89 * nzc, 0, 1)
        shade = (0.18 + 0.80 * diff + 0.35 * diff ** 30)[:, None, None]
        edge = ss(1.0, 0.92, np.abs(s_))[:, None, None]
        f[r0:r1] = f[r0:r1] * (1 - edge) + band * shade * edge
    return f


# ---- bare bulb pops ("then we let it pop", second time)
def _bulb_assets():
    if 'bulb' in _K4: return _K4['bulb']
    S = 600; c0x, c0y = S // 2, 380
    Y, X = np.mgrid[0:S, 0:S].astype(np.float32); X -= c0x; Y -= c0y
    rb = 130.0; r = np.sqrt(X * X + Y * Y) + 1e-3
    wneck = np.clip(lerp(150, 84, (-Y - 90) / 120), 84, 150)
    glass = np.maximum(ss(rb + 1, rb - 1, r), ss(1, -1, np.abs(X) - wneck / 2) * ss(-212, -208, Y) * ss(-80, -100, Y))
    edge = np.clip(1 - np.minimum(rb - r, 30) / 30, 0, 1) * (r < rb + 1)
    fres = np.maximum(edge, ss(wneck / 2 - 24, wneck / 2, np.abs(X)) * (Y < -90)) * glass
    g_al = glass * (0.14 + 0.55 * fres) + glass * 0.5 * np.exp(-(((X + 52) / 26) ** 2 + ((Y + 48) / 40) ** 2))
    base = ss(43, 41, np.abs(X)) * ss(-282, -278, Y) * ss(-206, -210, Y)
    thread = 0.45 + 0.25 * np.sin(Y * 0.55) + 0.25 * np.exp(-((X + 18) / 10) ** 2)
    cap_ = ss(30, 28, np.abs(X)) * ss(-300, -296, Y) * ss(-280, -284, Y)
    fil = np.zeros((S, S), np.float32)
    pts = np.int32([[c0x - 54 + 12 * k, c0y - 40 + (14 if k % 2 else -14)] for k in range(10)])
    cv2.polylines(fil, [pts], False, 1.0, 3, cv2.LINE_AA)
    sup = np.zeros((S, S), np.float32)
    for xx_ in (-54, 54): cv2.line(sup, (c0x + xx_, c0y - 40), (c0x + xx_ // 3, c0y - 200), 1.0, 2, cv2.LINE_AA)
    glow = np.exp(-((X / 95) ** 2 + ((Y + 30) / 85) ** 2)) * glass
    metal = (base * thread + cap_ * 0.25)
    _K4['bulb'] = dict(S=S, c=(c0x, c0y), glass=g_al.astype(np.float32), gm=glass, fil=fil, sup=sup, glow=glow.astype(np.float32),
                       metal=metal.astype(np.float32), mm=np.maximum(base, cap_).astype(np.float32), top=(c0x, c0y - 300))
    rng = np.random.default_rng(5); sh = []
    for k in range(46):
        ang = rng.uniform(0, 2 * math.pi); rr_ = rng.uniform(0, 1) ** 0.5 * 120
        sh.append(dict(p=(rr_ * math.cos(ang), rr_ * math.sin(ang)), v=(math.cos(ang) * rng.uniform(500, 1700), math.sin(ang) * rng.uniform(500, 1700) - 500),
                       w=rng.uniform(-14, 14), s=rng.uniform(9, 30), seed=k, spark=k % 3 == 0))
    _K4['shards'] = sh
    return _K4['bulb']


def t_bulb(t, c, A, B, a, b):
    """A bare bulb drops in on its cord and swings, buzzing and flickering on the shot; on the hit it POPs: the
    filament flashes, the glass bursts into shards and sparks, the room goes dark and the next shot comes up out of it."""
    g = _bulb_assets(); S = g['S']; tp = c - 0.02
    drop = eout(clamp((t - a) / 0.14)); Lc = lerp(0.05, 0.27, drop) * H
    th = math.radians(18) * math.exp(-(t - a) * 3.5) * math.cos((t - a) * 9.5)
    pvx, pvy = W / 2, -40.0
    sc = 0.62; popped = t >= tp
    bxc = pvx + (Lc + 300 * sc) * math.sin(th); byc = pvy + (Lc + 300 * sc) * math.cos(th)       # bulb centre at the end of the cord
    fr_ = int(round(t * FPS)); I = 0.0 if popped else (0.78 + 0.22 * ((fr_ * 2654435761) % 97) / 97) * drop
    if t < c:
        f = layer(A, t) * (0.62 + 0.42 * I)
        rr = np.sqrt((XX - bxc) ** 2 + (YY - byc) ** 2)
        f = f + (np.exp(-(rr / 420) ** 2) * 0.32 * I)[..., None] * np.float32([0.35, 0.72, 1.0])
    else:
        u = clamp((t - c) / (b - c)); lift = eout(clamp((t - c - 0.05) / 0.32))
        f = layer(B, t, 1 + 0.04 * (1 - u)) * lerp(0.10, 1.0, lift)
        rr = np.sqrt((XX - bxc) ** 2 + (YY - byc) ** 2)
    M = cv2.getRotationMatrix2D((float(g['c'][0]), float(g['c'][1])), math.degrees(th), sc); M[0, 2] += bxc - g['c'][0]; M[1, 2] += byc - g['c'][1]
    wp = lambda im: cv2.warpAffine(im, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    top = M[:, :2] @ np.array(g['top'], np.float64) + M[:, 2]
    cord = np.zeros((H, W), np.float32); cv2.line(cord, (int(pvx), int(pvy)), (int(top[0]), int(top[1])), 1.0, 7, cv2.LINE_AA)
    vis = 1.0 if t < c else 1 - clamp((t - c) / (b - c))
    f = f * (1 - cord[..., None] * vis) + np.float32([0.06, 0.06, 0.07]) * cord[..., None] * vis
    mm = wp(g['mm']); f = f * (1 - mm[..., None] * vis) + (wp(g['metal']) / np.maximum(mm, 1e-3))[..., None] * np.float32([0.62, 0.66, 0.70]) * mm[..., None] * vis
    warm = np.float32([0.45, 0.80, 1.0])
    if not popped:
        ga = wp(g['glass']); f = f * (1 - ga[..., None]) + ga[..., None] * (np.float32([0.95, 0.95, 0.92]) + warm * I * 0.4)
        f = f + (wp(g['glow']) * I * 0.8)[..., None] * warm + (wp(g['sup']) * 0.5)[..., None]
        f = f + (cv2.GaussianBlur(wp(g['fil']), (0, 0), 3) * I * 1.4 + wp(g['fil']) * (0.3 + 1.6 * I))[..., None] * np.float32([0.6, 0.9, 1.0])
    else:
        dt = t - tp
        f = f + (np.exp(-(rr / (120 + 2400 * dt)) ** 2) * 2.2 * math.exp(-dt * 16))[..., None] * np.float32([0.75, 0.92, 1.0])
        f = f + 0.8 * math.exp(-dt * 22)
        sm = np.zeros((H, W), np.float32); spk = np.zeros((H, W), np.float32)
        for s_ in _K4['shards']:
            px_ = bxc + s_['p'][0] * sc + s_['v'][0] * dt; py_ = byc + s_['p'][1] * sc + s_['v'][1] * dt + 0.5 * 2600 * dt * dt
            if s_['spark']:
                al_ = math.exp(-dt * 6); vx_, vy_ = s_['v'][0], s_['v'][1] + 2600 * dt
                cv2.line(spk, (int(px_), int(py_)), (int(px_ - vx_ * 0.02), int(py_ - vy_ * 0.02)), al_, 3, cv2.LINE_AA)
            else:
                r_ = np.random.default_rng(s_['seed']); ang0 = s_['w'] * dt
                poly = [(px_ + s_['s'] * math.cos(ang0 + k * 2.1 + r_.uniform(-0.4, 0.4)), py_ + s_['s'] * math.sin(ang0 + k * 2.1 + r_.uniform(-0.4, 0.4))) for k in range(3)]
                cv2.fillConvexPoly(sm, np.int32(poly), 0.55 + 0.45 * (math.sin(dt * 40 + s_['seed']) > 0.6), cv2.LINE_AA)
        f = f * (1 - 0.6 * sm[..., None]) + sm[..., None] * np.float32([0.95, 0.95, 0.92]) * 0.8
        f = f + (cv2.GaussianBlur(spk, (0, 0), 3) * 1.2 + spk)[..., None] * np.float32([0.45, 0.85, 1.0])
    return f


# ---- shooting-range target ("then we let it pop", second time)
_TGT_PT = 140


def _target_assets():
    if 'target' in _K4: return _K4['target']
    from PIL import Image, ImageDraw
    sw, sh = int(W * 1.04), int(H * 1.04); PT = _TGT_PT
    img = Image.new('RGBA', (sw, sh + PT), (0, 0, 0, 0)); d = ImageDraw.Draw(img)
    d.rectangle([0, PT, sw, PT + sh], fill=(236, 232, 221, 255)); ink = (24, 24, 26, 255)
    d.rectangle([40, PT - 26, sw - 40, PT - 6], fill=(150, 154, 160, 255))                        # carrier bar and clips
    for cx_ in (0.28 * sw, 0.72 * sw):
        d.rectangle([cx_ - 34, PT - 40, cx_ + 34, PT + 70], fill=(30, 30, 33, 255)); d.rectangle([cx_ - 30, PT - 36, cx_ + 30, PT - 24], fill=(90, 92, 96, 255))
    d.line([(sw / 2, 0), (sw / 2, PT - 26)], fill=(120, 122, 126, 255), width=6)
    d.rectangle([0, PT, sw, PT + 150], fill=ink)
    for txt, fn, y, col in (('MIAMI METRO POLICE', _font_px(INTER_BLACK, 66), 18, (245, 245, 245, 255)), ('FIREARMS RANGE  ·  B-27', _font_px(INTER_SEMI, 36), 100, (200, 200, 204, 255))):
        tw = d.textlength(txt, font=fn); d.text(((sw - tw) / 2, PT + y), txt, font=fn, fill=col)
    fm = _font_px(MONO, 38); d.text((44, PT + 168), 'LANE 07', font=fm, fill=ink); tw = d.textlength('25 YDS', font=fm); d.text((sw - 44 - tw, PT + 168), '25 YDS', font=fm, fill=ink)
    Y = lambda v: PT + v * sh
    d.ellipse([sw / 2 - 0.15 * sw, Y(0.17), sw / 2 + 0.15 * sw, Y(0.37)], fill=ink)                     # silhouette
    d.rectangle([sw / 2 - 0.07 * sw, Y(0.33), sw / 2 + 0.07 * sw, Y(0.42)], fill=ink)
    d.polygon([(0.13 * sw, Y(1.0)), (0.11 * sw, Y(0.56)), (0.24 * sw, Y(0.44)), (0.40 * sw, Y(0.40)), (0.60 * sw, Y(0.40)),
               (0.76 * sw, Y(0.44)), (0.89 * sw, Y(0.56)), (0.87 * sw, Y(1.0))], fill=ink)
    fr_ = _font_px(INTER_BLACK, 40)
    for k, lab in zip(range(5, 0, -1), ('7', '8', '9', '10', 'X')):
        rx, ry = 0.075 * sw * k, 0.055 * sh * k; cy_ = Y(0.68)
        d.ellipse([sw / 2 - rx, cy_ - ry, sw / 2 + rx, cy_ + ry], outline=(236, 232, 221, 255), width=5)
        d.text((sw / 2 + rx - 0.06 * sw, cy_ - 22), lab, font=fr_, fill=(236, 232, 221, 255))
    d.ellipse([sw / 2 - 0.08 * sw, Y(0.225), sw / 2 + 0.08 * sw, Y(0.315)], outline=(236, 232, 221, 255), width=5)
    a = np.asarray(img).astype(np.float32) / 255
    rgb = a[..., 2::-1].copy(); al = a[..., 3].copy()
    n = cv2.resize(fbm((sh + PT) // 16, sw // 16, 4, 23, 6), (sw, sh + PT))
    paper = (al > 0) & (np.arange(sh + PT)[:, None] >= PT)
    rgb[paper] *= (0.93 + 0.10 * n[paper])[:, None]
    xx = np.arange(sw, dtype=np.float32)[None, :]
    rgb[PT:] *= (1 - 0.10 * np.exp(-((xx - sw / 3) / 5) ** 2) - 0.08 * np.exp(-((xx - 2 * sw / 3) / 5) ** 2))[..., None]   # fold creases
    holes = ((0.42 * sw, Y(0.58), 120, 0), (0.59 * sw, Y(0.71), 105, 1), (0.50 * sw, Y(0.27), 160, 2))
    _K4['target'] = (rgb, al, holes)
    return _K4['target']


def t_target(t, c, A, B, a, b):
    """A shooting-range target (MIAMI METRO POLICE, B-27 silhouette) is run up to camera and fills the frame;
    POP, POP, POP: three shots punch ragged, powder-burned holes in it, the next shot showing through them,
    then the target is yanked back down the range and the next shot is left."""
    rgb0, al0, holes = _target_assets(); ph, pw = al0.shape; PT = _TGT_PT
    pops = (c - HALF, c - HALF / 2, c)
    rgb = rgb0.copy(); al = al0.copy(); shake = 0.0; flash = 0.0
    for (hx, hy, R, k), tp in zip(holes, pops):
        if t < tp: continue
        dt = t - tp; g = eout(clamp(dt / 0.05)); shake += 12 * math.exp(-dt * 22) * math.cos(dt * 90 + k); flash += 0.20 * math.exp(-dt * 22)
        x0, x1, y0, y1 = int(max(0, hx - 2.4 * R)), int(min(pw, hx + 2.4 * R)), int(max(0, hy - 2.4 * R)), int(min(ph, hy + 2.4 * R))
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32); dx, dy = xx - hx, yy - hy
        r = np.sqrt(dx * dx + dy * dy) + 1e-3; th = np.arctan2(dy, dx)
        jag = 1 + 0.15 * np.sin(th * 7 + 2 * k) + 0.09 * np.sin(th * 13 + 5 * k) + 0.05 * np.sin(th * 29 + k)
        Rr = R * g * jag
        hole = ss(Rr + 1.5, Rr - 1.5, r); burn = np.exp(-((r - Rr) / 10) ** 2) * (r > Rr - 2) * g
        smudge = np.exp(-(r / (1.8 * R)) ** 2) * 0.5 * g
        sub = rgb[y0:y1, x0:x1]
        sub *= (1 - smudge)[..., None]; sub[:] = sub * (1 - burn[..., None]) + 0.05 * burn[..., None]
        tear = np.exp(-((r - Rr - 6) / 4.5) ** 2) * g * (0.6 + 0.4 * np.sin(th * 17 + k) ** 2)            # torn white paper fibres
        sub[:] = sub * (1 - tear[..., None]) + np.float32([0.80, 0.85, 0.88]) * tear[..., None]
        al[y0:y1, x0:x1] *= 1 - hole
    if t < c: s = lerp(0.12, 1.0, eout(clamp((t - a) / 0.18))); u = 0.0
    else:
        u = clamp((t - c - 0.05) / (b - c - 0.05)); s = 0.06 ** ease(u)
    ky = ease(clamp((t - a) / 0.18)) if t < c else 1.0
    ctr = (W / 2, lerp(0.42 * H, H / 2, ky) if t < c else lerp(H / 2, 0.36 * H, ease(u)))
    rot = math.radians(5) * math.sin(u * math.pi * 1.5) * (t >= c)
    M = cv2.getRotationMatrix2D((pw / 2, PT + (ph - PT) / 2), math.degrees(rot), s)
    M[0, 2] += ctr[0] - pw / 2; M[1, 2] += ctr[1] - (PT + (ph - PT) / 2) + shake * s
    pre = np.dstack([rgb * al[..., None], al]) * (1 - ss(0.65, 1.0, u))
    lay = cv2.warpAffine(pre, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    rect = np.zeros_like(al0); rect[PT:] = 1.0
    rm = cv2.warpAffine(rect, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    fb = layer(B, t, 1 + 0.06 * (1 - eout(u)))
    if t < c:
        fa = layer(A, t) * (1 - 0.4 * ky)
        sh = cv2.GaussianBlur(np.roll(np.roll(rm, 30, 0), 18, 1), (0, 0), 24) * 0.55 * (s < 1.0)
        under = fa * (1 - sh[..., None]) * (1 - rm[..., None]) + fb * rm[..., None]
    else:
        sh = cv2.GaussianBlur(np.roll(np.roll(lay[..., 3], int(30 * s), 0), int(18 * s), 1), (0, 0), 6 + 18 * s) * 0.5
        under = fb * (1 - sh[..., None])
    f = over(under, lay)
    fl = np.zeros((H, W), np.float32)                                                             # paper flecks blown out of each hole
    for (hx, hy, R, k), tp in zip(holes, pops):
        dt = t - tp
        if not 0 <= dt < 0.30: continue
        p0 = M[:, :2] @ np.array([hx, hy]) + M[:, 2]; rng = np.random.default_rng(70 + k)
        for j in range(14):
            ang = rng.uniform(0, 2 * math.pi); sp = rng.uniform(500, 1400); sz = rng.uniform(6, 16) * (1 + 3 * dt)
            px_, py_ = p0[0] + math.cos(ang) * sp * dt, p0[1] + math.sin(ang) * sp * dt + 900 * dt * dt; a0 = rng.uniform(0, 6) + dt * 20
            poly = [(px_ + sz * math.cos(a0 + q * 1.9), py_ + sz * math.sin(a0 + q * 1.9)) for q in range(4)]
            cv2.fillConvexPoly(fl, np.int32(poly), 1 - dt / 0.30, cv2.LINE_AA)
    f = f * (1 - fl[..., None]) + np.float32([0.84, 0.88, 0.92]) * fl[..., None]
    return f + flash


# ============================================================ Dexter kit V: "The Kill Ritual" (fragment, slowed)
# Every transition is a step of Dexter's ritual: the bone saw (the drop), zip ties, the leather tool roll, the
# fingerprint card, the black contractor bag, the kill-room slide projector, a heart monitor that flatlines,
# and frost creeping over the lens (the Ice Truck Killer) for the loop.


def _ring_noise(seed, h=H, w=W, s=3):
    key = ('rn', seed, h, w, s)
    if key not in _K4: _K4[key] = cv2.resize(fbm(h // (8 * s) + 2, w // (8 * s) + 2, 4, seed, 4), (w, h), interpolation=cv2.INTER_CUBIC)
    return _K4[key]


def _raw(i, t, ez=1.0):
    """layer() without the projected-slide look (for the projector transitions, which apply it themselves)."""
    _K4['raw'] = True
    try: return layer(i, t, ez)
    finally: _K4['raw'] = False


# ---- bone saw (the drop)
def _saw_assets():
    if 'saw' in _K4: return _K4['saw']
    R = 330.0; D = int(2 * R + 40); c0 = D / 2
    Y, X = np.mgrid[0:D, 0:D].astype(np.float32) - c0; r = np.sqrt(X * X + Y * Y) + 1e-3; th = np.arctan2(Y, X)
    nT = 30; ph = np.mod(th / (2 * math.pi) * nT, 1.0)
    rim = R - 38 * np.where(ph < 0.8, ph / 0.8, (1 - ph) / 0.2) ** 1.2                         # saw teeth
    body = ss(rim + 1, rim - 1, r)
    tip = ((ph > 0.90) | (ph < 0.07)) & (r > R - 16)
    grind = 0.52 + 0.07 * np.sin(r * 0.9) + 0.05 * np.sin(r * 0.23)
    sheen = 0.30 * np.abs(np.cos(th - math.radians(-135))) ** 8
    v = (grind + sheen) * (0.8 + 0.2 * ss(R - 60, R - 20, r))
    v = np.where(tip, 0.80 + 0.15 * np.cos(th * nT * 2), v)
    slots = np.zeros((D, D), np.float32)
    for k in range(4):
        a0 = math.radians(20 + 90 * k)
        pts = np.int32([[c0 + (R - 40 - 22 * q) * math.cos(a0 + 0.10 * q), c0 + (R - 40 - 22 * q) * math.sin(a0 + 0.10 * q)] for q in range(5)])
        cv2.polylines(slots, [pts], False, 1.0, 5, cv2.LINE_AA); cv2.circle(slots, tuple(int(v_) for v_ in pts[-1]), 8, 1.0, -1, cv2.LINE_AA)
    al = body * (1 - slots) * ss(23, 26, r)
    fl = ss(64, 62, r); q = np.clip((r - 30) / 34, 0, 1) * 0.5                                    # flange washer
    v = v * (1 - fl) + _shade(X / r * q, Y / r * q, _nz(q), 0.45, ks=0.9, pw=40) * fl
    v = v * (1 - 0.6 * np.exp(-((r - 64) / 2) ** 2))
    trgb, tm = _polar_text('MORGAN  ·  BONE SAW  ·  7-1/4"  ·  5800 RPM', INTER_BLACK, 20, 120, 150, D, c0, (0.10, 0.10, 0.12), (0.5, 0.5, 0.5), reps=1)
    rgb = v[..., None] * np.float32([1.03, 1.0, 0.97])
    rgb = rgb * (1 - tm[..., None] * 0.85) + trgb * tm[..., None] * 0.85
    _K4['saw'] = (_rgba(rgb, al), R)
    return _K4['saw']


def t_saw(t, c, A, B, a, b):
    """The drop: a spinning bone-saw blade drops in and rips straight down through the shot (sparks and red mist
    thrown off the teeth, a glowing kerf behind it); on the drop the two halves burst apart onto the next shot."""
    spr, R = _saw_assets(); D = spr.shape[0]; bx = W / 2
    p = clamp((t - a) / (c - a)); by = lerp(-R - 60, H * 0.90, ease(p) ** 0.85)
    if t >= c: by = H * 0.90 + (t - c) * 3600
    sc = 1.0; ang = (t - a) * 2400.0
    if t < c:
        f = layer(A, t)
        vib = 3 * math.sin(t * 410); f = np.roll(f, int(vib), 1) if p > 0.3 else f
        ky = min(by, H)                                                                        # the kerf so far
        kd = np.abs(XX - bx); kin = (YY < ky) * 1.0
        f = f * (1 - (ss(5, 2, kd) * kin)[..., None]) + (ss(5, 2, kd) * kin)[..., None] * 0.02
        heat = np.exp(-np.clip(ky - YY, 0, None) / 260) * kin
        f = f + (np.exp(-(kd / 7) ** 2) * heat)[..., None] * np.float32([0.25, 0.55, 1.0]) * 1.2
    else:
        u = clamp((t - c) / (b - c)); fb = layer(B, t, 1 + 0.15 * (1 - eout(u)))
        fa = layer(A, c - 1e-3); dx = eout(u) * W * 0.75
        kd = np.abs(XX - bx); glow = np.exp(-(kd / 9) ** 2)[..., None] * np.float32([0.25, 0.55, 1.0]) * (1 - u)
        fa = fa + glow
        left = (XX < bx).astype(np.float32); acc = fb.copy()
        for side, m in ((-1, left), (1, 1 - left)):
            M = cv2.getRotationMatrix2D((bx, H / 2), side * -7 * eout(u), 1.0); M[0, 2] += side * dx
            pc = cv2.warpAffine(fa * m[..., None], M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            pm = cv2.warpAffine(m, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            sh = cv2.GaussianBlur(np.roll(pm, 24, 1 if side < 0 else 1), (0, 0), 20) * 0.5
            acc = acc * (1 - sh[..., None]) * (1 - pm[..., None]) + pc
        f = acc + 0.55 * math.exp(-(t - c) * 14)
        if t - c < 0.2: f = np.roll(f, int(18 * math.exp(-(t - c) * 18) * math.sin((t - c) * 120)), 0)
    lay = None                                                                                  # spinning blade with rotational blur
    for j in range(3):
        M = cv2.getRotationMatrix2D((D / 2, D / 2), -(ang + j * 2.5), sc); M[0, 2] += bx - D / 2; M[1, 2] += by - D / 2
        lj = cv2.warpAffine(spr, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        lay = lj if lay is None else lay + lj
    lay /= 3
    sh = cv2.GaussianBlur(np.roll(np.roll(lay[..., 3], 30, 0), 20, 1), (0, 0), 22) * 0.55
    f = over(f * (1 - sh[..., None]), lay)
    rb = np.sqrt((XX - bx) ** 2 + (YY - by) ** 2) + 1e-3; thb = np.arctan2(YY - by, XX - bx)           # fixed highlight on the spinning steel
    f = f + (lay[..., 3] * (0.35 * np.abs(np.cos(thb - math.radians(-135))) ** 10 + 0.25 * np.exp(-((rb - R * 0.86) / 10) ** 2)))[..., None]
    cy_ = by + R * 0.98                                                                        # sparks and mist off the cutting edge
    if -40 < cy_ < H + 40 and t < c + 0.15:
        sp = np.zeros((H, W), np.float32); mist = np.zeros((H, W), np.float32); rng = np.random.default_rng(1)
        for i in range(140):
            tb = a + i * 0.0065; age = t - tb
            if not 0 <= age < 0.24: continue
            pb = clamp((tb - a) / (c - a)); byb = lerp(-R - 60, H * 0.90, ease(pb) ** 0.85) + R * 0.98
            if byb < 0: continue
            ang_ = math.radians(rng.uniform(150, 215)); v_ = rng.uniform(900, 2200)
            vx, vy = math.cos(ang_) * v_, math.sin(ang_) * v_ - 300
            px_, py_ = bx + vx * age, byb + vy * age + 1500 * age * age
            if i % 4 == 0: cv2.circle(mist, (int(px_), int(py_)), int(3 + 6 * age * 10), 1 - age / 0.24, -1, cv2.LINE_AA)
            else: cv2.line(sp, (int(px_), int(py_)), (int(px_ - vx * 0.022), int(py_ - (vy + 3000 * age) * 0.022)), 1 - age / 0.24, 3, cv2.LINE_AA)
        f = f * (1 - mist[..., None] * 0.7) + BLOOD * mist[..., None] * 0.7
        f = f + (cv2.GaussianBlur(sp, (0, 0), 4) * 2.0 + sp * 1.4)[..., None] * np.float32([0.35, 0.75, 1.0])
    return f


# ---- zip tie
def _ziptie_assets():
    if 'zip' in _K4: return _K4['zip']
    sh_, L = 56, 2 * W
    yy, xx = np.mgrid[0:sh_, 0:L].astype(np.float32); s = (yy - sh_ / 2) / (sh_ / 2)
    teeth = (np.mod(xx, 9) < 4.5) * (yy < sh_ * 0.45)
    v = _shade(0 * s, s * 0.8, _nz(np.clip(s, -1, 1) * 0.8), 0.10, ks=0.35, pw=20) + 0.10 * teeth
    strap = (v[..., None] * np.float32([1.0, 1.0, 1.0])).astype(np.float32)
    hb, hw = 120, 104
    yy, xx = np.mgrid[0:hb, 0:hw].astype(np.float32); e = np.minimum(np.minimum(xx, hw - 1 - xx), np.minimum(yy, hb - 1 - yy))
    tl = 1 - ss(0, 10, e); nx = np.where(xx < hw / 2, -1, 1) * tl * (np.minimum(xx, hw - 1 - xx) < np.minimum(yy, hb - 1 - yy)); ny = np.where(yy < hb / 2, -1, 1) * tl * (np.minimum(xx, hw - 1 - xx) >= np.minimum(yy, hb - 1 - yy))
    hv = _shade(nx * 0.7, ny * 0.7, _nz(np.maximum(np.abs(nx), np.abs(ny)) * 0.7), 0.12, ks=0.5, pw=25)
    slot = (np.abs(xx - hw / 2) < 34) & (np.abs(yy - hb / 2) < 14); hv = np.where(slot, 0.01, hv)
    _K4['zip'] = (strap, hv.astype(np.float32))
    return _K4['zip']


def t_ziptie(t, c, A, B, a, b):
    """A heavy black zip tie loops round the shot's waist and ratchets tight in three clicks, cinching the frame
    in like a sack; the tail is snipped and the tie snaps off, the frame springs back as the next shot."""
    strap, head = _ziptie_assets(); sh_ = strap.shape[0]; yb = 0.58 * H; cx = W / 2
    clicks = (c - 0.30, c - 0.20, c - 0.10)
    k = 0.0
    for j, tc_ in enumerate(clicks): k += 0.15 * eout(clamp((t - tc_) / 0.05))
    k = k * (t < c) + (-0.10 * math.exp(-(t - c) * 9) * math.cos((t - c) * 40)) * (t >= c)
    swing = 0.12 * eout(clamp((t - a) / 0.10))                                                  # the loop drops on first
    sy = 1 - k * np.exp(-((YY - yb) / (0.24 * H)) ** 2)
    mx = (cx + (XX - cx) / sy).astype(np.float32)
    src = layer(A, min(t, c - 1e-3)) if t < c else layer(B, t)
    bg = np.float32([0.03, 0.03, 0.035])
    f = cv2.remap(src, mx, YY, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=tuple(float(v) for v in bg))
    if t >= c: f = f + 0.45 * math.exp(-(t - c) * 14)
    snap = t >= c - 0.02
    if not snap or t < c + 0.12:
        half = W / 2 * (1 - max(k, 0) * 1.0); y0 = int(yb - sh_ / 2); fall = 0.0 if not snap else 2600 * (t - c + 0.02) ** 2
        for side in (-1, 1):                                                                     # the strap round the waist
            x0, x1 = (cx - half, cx) if side < 0 else (cx, cx + half)
            if snap: x0, x1 = x0 + side * 900 * (t - c + 0.02), x1 + side * 900 * (t - c + 0.02)
            xs, xe = int(max(0, x0)), int(min(W, x1))
            if xe > xs and 0 <= y0 + fall < H - sh_:
                yy0 = int(y0 + fall); f[yy0:yy0 + sh_, xs:xe] = strap[:, :xe - xs]
        if not snap:
            tail = 120 + 1400 * max(k, 0); M = cv2.getRotationMatrix2D((0, sh_ / 2), -18, 1.0); M[0, 2] += cx + 40; M[1, 2] += yb - sh_ / 2
            tm = cv2.warpAffine(np.ones((sh_, int(tail)), np.float32), M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            tc = cv2.warpAffine(strap[:, :int(tail)], M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            f = f * (1 - tm[..., None]) + tc
            hb_, hw_ = head.shape; hy, hx = int(yb - hb_ / 2), int(cx - hw_ / 2)
            f[hy:hy + hb_, hx:hx + hw_] = head[..., None] * np.float32([1, 1, 1.02])
            f[hy:hy + hb_, hx:hx + hw_] += 0.35 * math.exp(-min((abs(t - tc_) for tc_ in clicks)) * 40)
    return f


# ---- the leather tool roll
def _roll_assets():
    if 'troll' in _K4: return _K4['troll']
    n = _ring_noise(19, s=1); g = cv2.resize(np.random.default_rng(2).random((H // 2, W // 2)).astype(np.float32), (W, H))
    leather = np.float32([0.10, 0.19, 0.34])[None, None] * (0.75 + 0.35 * n[..., None] + 0.10 * (g[..., None] - 0.5))
    rgb = leather.copy(); al = np.ones((H, W), np.float32)
    st = np.zeros((H, W), np.float32)
    for y in (26, H - 26):
        for x in range(10, W, 26): cv2.line(st, (x, y), (x + 13, y), 1.0, 3, cv2.LINE_AA)
    for x in (26, W - 26):
        for y in range(10, H, 26): cv2.line(st, (x, y), (x, y + 13), 1.0, 3, cv2.LINE_AA)
    rgb = rgb * (1 - st[..., None]) + np.float32([0.45, 0.62, 0.72]) * st[..., None]
    def put(mask, v, tint=(1.0, 1.0, 1.0)):
        nonlocal rgb
        sh = cv2.GaussianBlur(np.roll(np.roll(mask, 14, 0), 10, 1), (0, 0), 9) * 0.6
        rgb = rgb * (1 - sh[..., None]) * (1 - mask[..., None]) + (v * mask)[..., None] * np.float32(tint)
    def steel(mask, x0, x1, base=0.62):
        u = np.clip((XX - x0) / max(1, x1 - x0), 0, 1); nx = (u - 0.5) * 1.6
        return _shade(np.clip(nx, -0.95, 0.95), 0 * nx, _nz(np.clip(nx, -0.95, 0.95)), base, ks=1.0, pw=30) + 0.18 * np.exp(-((u - 0.3) / 0.06) ** 2)
    def poly(pts):
        m = np.zeros((H, W), np.float32); cv2.fillConvexPoly(m, np.int32(pts), 1.0, cv2.LINE_AA); return m
    m = np.zeros((H, W), np.float32); cv2.rectangle(m, (120, int(0.17 * H)), (176, int(0.58 * H)), 1.0, -1)      # scalpel
    put(m, steel(m, 120, 176) * (1 - 0.18 * (np.mod(YY, 12) < 4) * (YY > 0.30 * H) * (YY < 0.50 * H)))
    put(poly([[122, int(0.58 * H)], [174, int(0.58 * H)], [184, int(0.66 * H)], [150, int(0.74 * H)], [126, int(0.67 * H)]]), steel(m, 120, 190, 0.75))
    m = np.zeros((H, W), np.float32); cv2.rectangle(m, (300, int(0.12 * H)), (384, int(0.40 * H)), 1.0, -1)       # boning knife
    put(m, 0.05 + 0.10 * steel(m, 300, 384, 0.4))
    for y in (0.18, 0.26, 0.34): cv2.circle(rgb, (342, int(y * H)), 10, (0.75, 0.78, 0.80), -1, cv2.LINE_AA)
    put(poly([[296, int(0.40 * H)], [388, int(0.40 * H)], [384, int(0.45 * H)], [300, int(0.45 * H)]]), steel(m, 296, 388, 0.55))
    mk = poly([[302, int(0.45 * H)], [382, int(0.45 * H)], [376, int(0.72 * H)], [330, int(0.90 * H)], [306, int(0.72 * H)]])
    put(mk, steel(mk, 300, 384, 0.68) + 0.25 * np.exp(-((XX - 306) / 4) ** 2))
    bx0, bx1 = 500, 800; by0 = int(0.30 * H); by1 = by0 + int(round((bx1 - bx0) * 16 / 9))          # cleaver (blade is 9:16)
    m = np.zeros((H, W), np.float32); cv2.rectangle(m, (bx1 - 92, int(0.07 * H)), (bx1 - 20, by0), 1.0, -1)
    put(m, 0.20 + 0.12 * _ring_noise(4, s=1), (0.55, 0.80, 1.0))
    for y in (0.12, 0.19, 0.26): cv2.circle(rgb, (bx1 - 56, int(y * H)), 11, (0.72, 0.75, 0.78), -1, cv2.LINE_AA)
    mb = np.zeros((H, W), np.float32); cv2.rectangle(mb, (bx0, by0), (bx1, by1), 1.0, -1)
    brush = cv2.GaussianBlur(np.random.default_rng(6).random((H, W)).astype(np.float32), (0, 0), sigmaX=40, sigmaY=0.8)
    put(mb, steel(mb, bx0, bx1, 0.70) * (0.9 + 0.5 * (brush - brush.mean()) / (brush.std() + 1e-6) * 0.15))
    m = np.zeros((H, W), np.float32); cv2.rectangle(m, (896, int(0.28 * H)), (980, int(0.62 * H)), 1.0, -1)       # M99 syringe
    put(m, 0.55 + 0.25 * steel(m, 896, 980, 0.5), (1.0, 1.0, 1.0))
    liq = np.zeros((H, W), np.float32); cv2.rectangle(liq, (904, int(0.44 * H)), (972, int(0.61 * H)), 1.0, -1)
    rgb = rgb * (1 - 0.8 * liq[..., None]) + np.float32([0.10, 0.12, 0.70]) * 0.8 * liq[..., None]
    for yy_ in range(int(0.31 * H), int(0.60 * H), 40): cv2.line(rgb, (896, yy_), (918, yy_), (0.15, 0.15, 0.18), 2, cv2.LINE_AA)
    m = np.zeros((H, W), np.float32); cv2.rectangle(m, (928, int(0.14 * H)), (948, int(0.28 * H)), 1.0, -1); cv2.rectangle(m, (880, int(0.13 * H)), (996, int(0.145 * H)), 1.0, -1)
    cv2.rectangle(m, (870, int(0.27 * H)), (1006, int(0.285 * H)), 1.0, -1)
    put(m, steel(m, 870, 1006, 0.6))
    m = np.zeros((H, W), np.float32); cv2.line(m, (938, int(0.62 * H)), (938, int(0.80 * H)), 1.0, 4, cv2.LINE_AA); put(m, steel(m, 930, 946, 0.8))
    rgb = _blit(rgb, _text_mask('M99', INTER_BLACK, 40), 900, int(0.33 * H), (0.12, 0.10, 0.75))
    bev = np.zeros((H, W), np.float32)                                                          # cleaver spine, ground edge, hole
    bev += (np.abs(YY - (by0 + 10)) < 10) * (np.abs(XX - (bx0 + bx1) / 2) < (bx1 - bx0) / 2) * -0.35
    bev += ss(by1 - 70, by1 - 6, YY) * (YY < by1) * (np.abs(XX - (bx0 + bx1) / 2) < (bx1 - bx0) / 2) * 0.35
    bev += np.exp(-((YY - (by1 - 3)) / 2.5) ** 2) * (np.abs(XX - (bx0 + bx1) / 2) < (bx1 - bx0) / 2) * 0.6
    hole = ss(23, 21, np.sqrt((XX - (bx1 - 48)) ** 2 + (YY - (by0 + 56)) ** 2)); bev -= hole * 0.9
    _K4['troll_bev'] = bev.astype(np.float32)
    for y0 in (int(0.24 * H), int(0.70 * H)):                                                   # leather straps over the tools
        sm = np.zeros((H, W), np.float32); cv2.rectangle(sm, (60, y0), (W - 60, y0 + 64), 1.0, -1)
        sm = sm * (1 - ((XX > bx0 - 8) & (XX < bx1 + 8) & (YY > by0 - 8) & (YY < by1 + 8)))
        sh = cv2.GaussianBlur(np.roll(sm, 10, 0), (0, 0), 8) * 0.6
        rgb = rgb * (1 - sh[..., None]) * (1 - sm[..., None]) + (leather * 0.75 * sm[..., None])
        ss_ = np.zeros((H, W), np.float32)
        for x in range(70, W - 70, 22): cv2.line(ss_, (x, y0 + 8), (x + 11, y0 + 8), 1.0, 2); cv2.line(ss_, (x, y0 + 56), (x + 11, y0 + 56), 1.0, 2)
        rgb = rgb * (1 - ss_[..., None] * sm[..., None]) + np.float32([0.45, 0.62, 0.72]) * (ss_ * sm)[..., None]
    _K4['troll'] = (rgb.astype(np.float32), (bx0, by0, bx1, by1))
    return _K4['troll']


def t_toolroll(t, c, A, B, a, b):
    """Dexter's leather tool roll unrolls across the shot (scalpel, boning knife, cleaver, M99 syringe in their
    straps), a glint runs along the steel, then the camera dives into the mirror-polished cleaver, whose
    reflection is the next shot."""
    rgb, (bx0, by0, bx1, by1) = _roll_assets(); Rr = 64.0
    xr = lerp(-2 * Rr, W + 2 * Rr, eout(clamp((t - a) / 0.30)))
    fa = layer(A, min(t, c - 1e-3)); fb = layer(B, t)
    bw, bh = bx1 - bx0, by1 - by0; bcx, bcy = (bx0 + bx1) / 2, (by0 + by1) / 2
    td = c - 0.06; u = clamp((t - td) / (b - td)); zmax = W / bw
    z = math.exp(math.log(zmax) * ein(u) ** 0.8)
    ccx, ccy = lerp(bcx, W / 2, ease(u)), lerp(bcy, H / 2, ease(u))
    Mz = np.float32([[z, 0, ccx - z * bcx], [0, z, ccy - z * bcy]])
    sc = cv2.warpAffine(rgb, Mz, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE) if z > 1.001 else rgb
    q0 = Mz @ np.array([bx0, by0, 1.0]); q1 = Mz @ np.array([bx1, by1, 1.0])
    Mb = np.float32([[(q1[0] - q0[0]) / W, 0, q0[0]], [0, (q1[1] - q0[1]) / H, q0[1]]])
    refl = cv2.warpAffine(fb, Mb, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    rm = cv2.warpAffine(np.ones((H, W), np.float32), Mb, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mix = 0.12 + 0.88 * ease(u)
    sc = sc * (1 - rm[..., None] * mix) + refl * np.float32([1.04, 1.0, 0.96]) * rm[..., None] * mix
    bev = _K4['troll_bev'] if z <= 1.001 else cv2.warpAffine(_K4['troll_bev'], Mz, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    sc = sc + (bev * (1 - ease(u)))[..., None]
    gl = clamp((t - (c - 0.24)) / 0.20)                                                         # glint along the steel
    if 0 < gl < 1: sc = sc + (np.exp(-(((XX * 0.6 + YY * 0.8) - lerp(-200, W * 0.6 + H * 0.8 + 200, gl)) / 40) ** 2) * 0.5)[..., None]
    if z <= 1.001:
        m = ss(xr + 1, xr - 1, XX)[..., None]
        f = fa * (1 - m) + sc * m
        cr = (XX >= xr) & (XX < xr + 2 * Rr)                                                    # the rolled-up leather cylinder
        s_ = np.clip((XX - xr - Rr) / Rr, -1, 1); shade = (0.25 + 0.85 * np.clip(-0.5 * s_ + 0.85 * np.sqrt(1 - s_ ** 2), 0, 1))
        cyl = np.float32([0.10, 0.19, 0.34]) * (0.8 + 0.3 * _ring_noise(19, s=1)[..., None]) * shade[..., None]
        f = np.where(cr[..., None], cyl, f)
        f = f * (1 - (np.exp(-np.clip(XX - xr - 2 * Rr, 0, None) / 40) * (XX > xr + 2 * Rr) * 0.5 * (xr < W))[..., None])
    else: f = sc
    return f


# ---- fingerprint card
def _fp_assets():
    if 'fp' in _K4: return _K4['fp']
    from PIL import Image, ImageDraw
    img = Image.new('RGB', (W, H), (238, 236, 230)); d = ImageDraw.Draw(img); ink = (30, 30, 34)
    d.text((60, 70), 'MIAMI METRO POLICE DEPARTMENT', font=_font_px(INTER_BLACK, 44), fill=ink)
    d.text((60, 128), 'FINGERPRINT RECORD  ·  CASE 0915-07', font=_font_px(INTER_SEMI, 30), fill=(90, 90, 96))
    d.line([(60, 180), (W - 60, 180)], fill=ink, width=4)
    d.rectangle([110, int(0.27 * H), W - 110, int(0.73 * H)], outline=ink, width=4)
    d.text((126, int(0.27 * H) + 12), '1.  R. THUMB', font=_font_px(INTER_BLACK, 32), fill=ink)
    for k, lab in enumerate(('2. R. INDEX', '3. R. MIDDLE', '4. R. RING')):
        x0 = 60 + k * 330; d.rectangle([x0, int(0.78 * H), x0 + 300, int(0.92 * H)], outline=ink, width=3); d.text((x0 + 14, int(0.78 * H) + 10), lab, font=_font_px(INTER_SEMI, 26), fill=ink)
    a = np.asarray(img).astype(np.float32)[..., ::-1] / 255
    a *= (0.95 + 0.07 * _ring_noise(31, s=1))[..., None]
    rng = np.random.default_rng(57)
    def nz_(sig): n = cv2.GaussianBlur(rng.random((H, W)).astype(np.float32), (0, 0), sig); return ((n - n.mean()) / (n.std() + 1e-6)).astype(np.float32)
    _K4['fp'] = (a.astype(np.float32), nz_(28) * 0.5, nz_(28) * 0.5)
    _K4['fp_fine'] = nz_(4)
    return _K4['fp']


def t_fingerprint(t, c, A, B, a, b):
    """A fingerprint card slaps into frame and Dexter's inked thumb rolls onto it (the whorl spreads from the
    core); the camera dives into the whorl as the ridges spin open and the next shot shows between them."""
    card, nwx, nwy = _fp_assets(); P = np.array([W / 2, 0.52 * H])
    land = eout(clamp((t - a) / 0.14))
    td = c - 0.06; u = clamp((t - td) / (b - td)); z = math.exp(math.log(28.0) * ein(u) ** 0.9)
    C = np.array([lerp(P[0], W / 2, ease(u)), lerp(P[1], H / 2, ease(u))])
    lx = (XX - C[0]) / z; ly = (YY - C[1]) / z                                                   # card coordinates around the core
    if land < 1:
        s_ = lerp(1.6, 1.0, land); ang = math.radians(9 * (1 - land))
        ca, sa = math.cos(ang), math.sin(ang); lx, ly = (lx * ca - ly * sa) / s_, (lx * sa + ly * ca) / s_
    mx = (lx + P[0]).astype(np.float32); my = (ly + P[1]).astype(np.float32)
    paper = cv2.remap(card, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    wx = cv2.remap(nwx, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT); wy = cv2.remap(nwy, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    nf = cv2.remap(_K4['fp_fine'], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    RX, RY = 260.0, 330.0
    r = np.sqrt(lx * lx + (ly / 1.22) ** 2) + 1e-3; th = np.arctan2(ly, lx); rn = np.sqrt((lx / RX) ** 2 + (ly / RY) ** 2)
    swirl = 2.2 * ein(u) * np.exp(-r / 160)
    pw_ = 2 * math.pi * (r + 22 * wx) / 10.5 + 1.0 * (th + swirl) + 3 * wy                         # whorl core
    ya = (ly + 60) - 0.0016 * lx * lx * np.sign(ly + 60 + 1e-3)                                    # arches round it
    pa = 2 * math.pi * (np.abs(ya) + 22 * wx) / 10.5 + 3 * wy
    wb = ss(0.42, 0.70, rn)
    ridge = 0.5 + 0.5 * ((1 - wb) * np.cos(pw_) + wb * np.cos(pa))
    brk = ss(-1.5, -0.9, nf)                                                                        # ridge endings and breaks
    shape = ss(1.02, 0.92, rn)
    tp = a + 0.16; grow = clamp((t - tp) / 0.16) * 1.2
    spread = ss(grow * 1.05, grow * 1.05 - 0.15, rn)
    ink = ss(0.38, 0.62, ridge) * shape * spread * (0.85 + 0.12 * nf) * (t >= tp) * brk
    ink = np.clip(ink, 0, 1)
    f = paper * (1 - ink[..., None] * 0.92) + np.float32([0.09, 0.08, 0.10]) * ink[..., None] * 0.92
    if land < 1:
        bg = layer(A, t) * 0.55; cm = ss(0.5, -0.5, np.maximum(np.abs(mx - W / 2) - W / 2, np.abs(my - H / 2) - H / 2))[..., None]
        f = bg * (1 - cm) + f * cm
    if t >= td:
        fb = layer(B, t, 1 + 0.08 * (1 - u)); open_ = ss(0.0, 0.9, u)
        mB = np.clip(open_ * 1.4 * (1 - ridge * shape) + ss(0.55, 1.0, u), 0, 1)[..., None]
        f = f * (1 - mB) + fb * mB
    return f


# ---- black contractor bag
def t_bag2(t, c, A, B, a, b):
    """The frame is swallowed by a black contractor bag: glossy crinkled plastic gathers in from the edges in
    pleats, sucking the shot into the shrinking mouth, and the red drawstring pulls it shut; on the hit a blade
    slits the bag open and the next shot spills out of the tear."""
    P = (W / 2, 0.46 * H); n1 = _ring_noise(71, s=1); n2 = _ring_noise(72, s=1)
    dx, dy = XX - P[0], YY - P[1]; r = np.sqrt(dx * dx + dy * dy) + 1e-3; th = np.arctan2(dy, dx)
    tc = c - 0.05; cl = clamp((t - (a + 0.04)) / (tc - (a + 0.04)))
    Rm = RFULL * 1.05 * (1 - ease(cl) ** 0.9)
    gy, gx = np.gradient(n1 * 40 + n2 * 14)
    pleat = np.sin(th * 26 + n2 * 6) * np.exp(-np.clip(r - Rm, 0, None) / 220)
    nx_ = np.clip(gx * 0.28 + (-np.sin(th)) * pleat * 0.6, -0.95, 0.95); ny_ = np.clip(gy * 0.28 + np.cos(th) * pleat * 0.6, -0.95, 0.95)
    v = _shade(nx_, ny_, _nz(np.sqrt(np.clip(nx_ ** 2 + ny_ ** 2, 0, 0.95))), 0.05, ks=1.15, pw=55, amb=0.45)
    v = v + 0.05 * np.exp(-YY / (0.6 * H))                                                        # overhead light on the plastic
    plastic = v[..., None] * np.float32([1.08, 1.0, 0.96])
    if t < c:
        s_img = max(0.02, (Rm / RFULL)) ** 0.8; rot = 0.6 * (1 - Rm / RFULL)
        ca, sa = math.cos(rot), math.sin(rot)
        mx = (P[0] + (dx * ca - dy * sa) / s_img).astype(np.float32); my = (P[1] + (dx * sa + dy * ca) / s_img).astype(np.float32)
        inside = cv2.remap(layer(A, t), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        mo = ss(Rm + 2, Rm - 2, r)[..., None]
        f = plastic * (1 - mo) + inside * mo
        ring = np.exp(-((r - Rm) / 9) ** 2) * (Rm < RFULL)
        f = f * (1 - ring[..., None]) + ring[..., None] * np.float32([0.10, 0.10, 0.72]) * (0.6 + 0.6 * np.cos(th * 30) ** 2)[..., None]
        f = f * (1 - (np.exp(-np.clip(r - Rm, 0, None) / 60) * 0.5 * (Rm < RFULL))[..., None]) + 0 * f
        if t >= tc - 0.02:
            kn = np.exp(-(r / 46) ** 2); f = f * (1 - kn[..., None]) + kn[..., None] * np.float32([0.10, 0.10, 0.70]) * (0.8 + 0.4 * np.cos(th * 9) ** 2)[..., None]
    else:
        u = clamp((t - c) / (b - c)); fb = layer(B, t, 1 + 0.12 * (1 - eout(u)))
        hh = lerp(0.18 * H, 1.3 * H, eout(u)); wo = lerp(6, 1.5 * W, eout(u) ** 1.3)
        jag = 1 + 0.10 * np.sin(YY / 23 + 3) + 0.06 * np.sin(YY / 9)
        prof = np.sqrt(np.clip(1 - ((YY - P[1]) / hh) ** 2, 0, 1)) * wo / 2 * jag
        hole = ss(prof + 1.5, prof - 1.5, np.abs(XX - P[0]))
        edge = np.exp(-((np.abs(XX - P[0]) - prof) / 8) ** 2) * (prof > 1)
        f = plastic * (1 - hole[..., None]) + fb * hole[..., None] + (edge * 0.35)[..., None]
        f = f + 0.35 * math.exp(-(t - c) * 14)
    return f


# ---- kill-room slide projector (projector -> click -> lightsup)
def _proj_assets():
    if 'proj' in _K4: return _K4['proj']
    n = _ring_noise(83, s=1)
    fold = (np.sin(XX / 70 + 5 * n) * 9 + np.sin(XX / 23 + 2 * n) * 3).astype(np.float32)
    sheen = (np.exp(-((np.mod(XX + 40 * n, 140) - 70) / 6) ** 2) * 0.10).astype(np.float32)
    x0, y0, x1, y1 = 0.05 * W, 0.07 * H, 0.95 * W, 0.87 * H
    e = np.minimum(np.minimum(XX - x0, x1 - XX), np.minimum(YY - y0, y1 - YY))
    area = ss(-3, 6, e)
    hot = (1 - 0.45 * np.clip(((XX - W / 2) / (0.5 * W)) ** 2 + ((YY - 0.47 * H) / (0.45 * H)) ** 2, 0, 1)).astype(np.float32)
    _K4['proj'] = dict(fold=fold, sheen=sheen, area=area.astype(np.float32), hot=hot, rect=(x0, y0, x1, y1))
    return _K4['proj']


def proj_look(f, t, seed=0, k=1.0, shift=0.0):
    """The shot as a slide thrown onto kill-room plastic: warm lamp light, hot centre, plastic folds bending it,
    dust and scratches, lamp flicker; dark plastic wall around it."""
    g = _proj_assets(); x0, y0, x1, y1 = g['rect']
    sx, sy = (x1 - x0) / W, (y1 - y0) / H
    mx = ((XX - x0 - shift) / sx + g['fold'] * 0.6).astype(np.float32); my = ((YY - y0) / sy).astype(np.float32)
    img = cv2.remap(f, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    fr_ = int(round(t * FPS)); rng = np.random.default_rng(fr_ * 7 + seed)
    flick = 1 + 0.05 * (rng.random() - 0.5)
    img = img * np.float32([0.78, 0.95, 1.12]) * g['hot'][..., None] * 1.12 * flick + g['sheen'][..., None] * 0.6
    dust = np.zeros((H, W), np.float32)
    for _ in range(14): cv2.circle(dust, (int(rng.uniform(0, W)), int(rng.uniform(0, H))), int(rng.integers(1, 4)), 1.0, -1)
    if rng.random() < 0.6: xs = int(rng.uniform(0.1, 0.9) * W); cv2.line(dust, (xs, 0), (xs + int(rng.normal(0, 8)), H), 0.6, 1)
    img = img * (1 - 0.6 * dust[..., None])
    am = (g['area'] * k)[..., None]
    wall = (0.025 + g['sheen'] * 0.35)[..., None] * np.float32([1.0, 1.0, 1.05])
    return wall * (1 - am) + img * am


def t_projector(t, c, A, B, a, b):
    """The lights cut out (a fluorescent stutter), the slide projector's lamp comes on and the next shot is
    thrown onto the plastic sheeting of the kill room."""
    if t < c:
        u = clamp((t - a) / (c - 0.05 - a)); fl = 1.0 if u < 0.35 else (0.25 if int(t * FPS) % 3 else 0.7) * (1 - ease((u - 0.35) / 0.65))
        fa_ = layer(A, t); f = fa_ * max(0.05, fl) + (1 - max(0.05, fl)) * (0.10 * fa_ + np.float32([0.05, 0.03, 0.02]))
        lamp = clamp((t - (c - 0.05)) / 0.05)
        if lamp > 0: f = f * (1 - lamp) + proj_look(_raw(B, t), t, B, k=lamp) * lamp
        return f
    u = clamp((t - c) / (b - c)); return proj_look(_raw(B, t), t, B, k=1.0) * (1 + 0.25 * (1 - u))


def t_click(t, c, A, B, a, b):
    """Slide change: the carousel clacks, the projection blinks dark, and the next slide drops into the gate."""
    if t < c:
        u = clamp((t - a) / (c - a)); return proj_look(_raw(A, t), t, A, k=1 - 0.85 * ease(u), shift=-80 * ease(u))
    u = clamp((t - c) / (b - c)); return proj_look(_raw(B, t), t, B, k=0.2 + 0.8 * eout(u), shift=60 * (1 - eout(u)))


def t_lightsup(t, c, A, B, a, b):
    """The projection pushes in until it fills the frame, then the kill-room lights slam on: the next shot, real."""
    if t < c:
        u = clamp((t - a) / (c - a)); f = proj_look(_raw(A, t), t, A); return zoom(f, 1 + 0.25 * ein(u), cy=0.47 * H)
    u = clamp((t - c) / (b - c)); return layer(B, t, 1 + 0.10 * (1 - eout(u))) + 0.6 * math.exp(-(t - c) * 12)


# ---- heart monitor (fx 'ekg' on a shot, 'flatline' out of it)
def _ekg_trace(t, s, flat_from=None):
    """Monitor trace: y offsets for each screen x at time t (beats on the shot's kicks and snares)."""
    import config as _c
    T = 1.1; x_h = ((t - s['start']) / T) % 1.0 * W
    beats = sorted(_hits('KICKS') + _hits('SNARES'))
    xs = np.arange(W, dtype=np.float32); tx = t - ((x_h - xs) % W) / W * T
    y = np.zeros(W, np.float32)
    for tb in beats:
        if flat_from is not None and tb >= flat_from: continue
        d = tx - tb
        y += 18 * np.exp(-((d + 0.09) / 0.02) ** 2) - 170 * np.exp(-(d / 0.009) ** 2) + 60 * np.exp(-((d - 0.02) / 0.008) ** 2) - 30 * np.exp(-((d - 0.16) / 0.035) ** 2)
    gap = ((x_h - xs) % W) > W - 50
    return y, x_h, gap


def ekg_overlay(f, t, s, flat_from=None, k=1.0):
    yb = 0.74 * H; y, x_h, gap = _ekg_trace(t, s, flat_from)
    lay = np.zeros((H, W), np.float32)
    for gy in range(int(0.60 * H), int(0.88 * H), 40): lay[gy:gy + 1] = 0.10
    for gx in range(0, W, 40): lay[int(0.60 * H):int(0.88 * H), gx:gx + 1] = 0.10
    pts = np.stack([np.arange(W), yb + y], 1).astype(np.int32)
    seg = np.split(pts, np.where(np.diff(gap.astype(int)) != 0)[0] + 1)
    tr = np.zeros((H, W), np.float32)
    for sgm in seg:
        if len(sgm) > 1 and not gap[sgm[0][0]]: cv2.polylines(tr, [sgm], False, 1.0, 4, cv2.LINE_AA)
    fade = np.clip(1 - ((x_h - np.arange(W)) % W) / W, 0.25, 1)[None, :]
    tr = tr * fade
    hx = int(x_h); cv2.circle(tr, (hx, int(yb + y[min(hx, W - 1)])), 9, 1.0, -1, cv2.LINE_AA)
    green = np.float32([0.45, 1.0, 0.35])
    out = f * (1 - 0.35 * k * ((YY > 0.58 * H) & (YY < 0.90 * H))[..., None]) + (lay * k)[..., None] * green
    out = out + (cv2.GaussianBlur(tr, (0, 0), 6) * 0.9 + tr)[..., None] * green * k
    hr = 'HR  --' if flat_from is not None and t >= flat_from else 'HR  %d' % (128 + int(4 * math.sin(t * 3)))
    col = (0.25, 0.25, 0.95) if hr.endswith('--') else (0.45, 1.0, 0.35)
    out = _blit(out, _text_mask(hr, MONO, 54), 60, int(0.61 * H), col, k)
    return out


def t_flatline(t, c, A, B, a, b):
    """The trace flatlines (HR --), the picture drains to monitor green and collapses into a single glowing line
    like a CRT switching off; on the hit the next shot snaps open out of the line."""
    if t < c - 0.12:
        f = layer(A, t)
        u = clamp((t - a) / (c - 0.12 - a)); l = luma(f)[..., None]; f = f * (1 - 0.6 * u) + l * np.float32([0.5, 1.0, 0.45]) * 0.6 * u
        return f
    bg = np.full((H, W, 3), 0.055, np.float32) * np.float32([0.6, 1.0, 0.6])
    if t < c:
        u = ein(clamp((t - (c - 0.12)) / 0.12)); sy = lerp(1.0, 0.004, u); sx = lerp(1.0, 0.85, u)
        f = layer(A, t) * (1 + 1.5 * u)
        M = np.float32([[sx, 0, W / 2 * (1 - sx)], [0, sy, H / 2 * (1 - sy)]])
        g = cv2.warpAffine(f, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        m = cv2.warpAffine(np.ones((H, W), np.float32), M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)[..., None]
        line = np.exp(-((YY - H / 2) / (3 + 40 * (1 - u))) ** 2)[..., None] * u * np.float32([0.7, 1.0, 0.7])
        return bg * (1 - m) + g * m + line
    u = clamp((t - c) / (b - c)); e = eout(clamp(u / 0.6)); sy = lerp(0.004, 1.0, e) * (1 + 0.06 * math.sin(u * 14) * (1 - u))
    fb = layer(B, t); M = np.float32([[1, 0, 0], [0, sy, H / 2 * (1 - sy)]])
    g = cv2.warpAffine(fb, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    m = cv2.warpAffine(np.ones((H, W), np.float32), M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)[..., None]
    f = bg * (1 - m) + g * m
    sp = int(14 * (1 - e))
    if sp > 1: f = f.copy(); f[..., 2] = np.roll(f[..., 2], sp, 1); f[..., 0] = np.roll(f[..., 0], -sp, 1)
    return f + (np.exp(-((YY - H / 2) / (6 + 300 * e)) ** 2) * 0.5 * math.exp(-(t - c) * 14))[..., None]


# ---- red heartbeat vignette (fx 'pulse')
def pulse_fx(f, t, s):
    import config as _c
    ks = _hits('BUMP_KICKS') if hasattr(_c, 'BUMP_KICKS') else _hits('KICKS'); a = 0.0
    for tk in ks:
        if s['start'] - 1e-6 <= tk <= t: a = max(a, math.exp(-(t - tk) / 0.16))
    if a < 0.02: return f
    if 'vig' not in _K4: _K4['vig'] = (np.clip(((XX - W / 2) / (0.55 * W)) ** 2 + ((YY - H / 2) / (0.55 * H)) ** 2, 0, 1.6) ** 1.5).astype(np.float32)
    m = (_K4['vig'] * 0.55 * a)[..., None]
    return f * (1 - m) + BLOOD * 0.9 * m


# ---- frost loop (the Ice Truck Killer)
def _frost_assets():
    if 'frost' in _K4: return _K4['frost']
    rng = np.random.default_rng(9)
    grain = cv2.GaussianBlur(rng.random((H, W)).astype(np.float32), (0, 0), 1.1); grain = (grain - grain.min()) / (grain.max() - grain.min())
    veins = np.zeros((H, W), np.float32)
    for k, s in enumerate((1, 2)):
        n = _ring_noise(100 + k, s=s); veins += (1 - np.abs(2 * n - 1)) ** 8 / (k + 1)
    veins /= veins.max()
    dend = np.zeros((H, W), np.float32)
    def branch(x, y, ang, L, depth):
        x1, y1 = x + math.cos(ang) * L, y + math.sin(ang) * L
        cv2.line(dend, (int(x), int(y)), (int(x1), int(y1)), 0.55 + 0.1 * depth, 1, cv2.LINE_AA)
        if depth == 0: return
        for j in range(1, 4):
            px_, py_ = x + (x1 - x) * j / 4, y + (y1 - y) * j / 4
            for sgn in (-1, 1): branch(px_, py_, ang + sgn * math.pi / 3, L * 0.42 * (1 - j / 6), depth - 1)
    for _ in range(80):                                                                       # crystals grow in from the edges
        side = rng.integers(4); u_ = rng.uniform(0, 1)
        x, y = [(u_ * W, 0), (W, u_ * H), (u_ * W, H), (0, u_ * H)][side]; x += rng.normal(0, 60); y += rng.normal(0, 60)
        ang = math.atan2(H * 0.45 - y, W / 2 - x) + rng.normal(0, 0.5)
        branch(x, y, ang, rng.uniform(120, 300), 3)
    dend = np.clip(cv2.GaussianBlur(dend, (0, 0), 0.7) * 1.6, 0, 1)
    pat = np.clip(0.30 * grain + 0.30 * veins + 0.95 * dend, 0, 1.2)
    edge = np.minimum(np.minimum(XX, W - XX) / W, np.minimum(YY, H - YY) / H * (H / W)) / 0.5
    G = np.clip(edge + 0.25 * (_ring_noise(110, s=2) - 0.5) - 0.20 * dend, 0, 1.3)
    spark = (rng.random((H, W)) > 0.9994).astype(np.float32)
    _K4['frost'] = (pat.astype(np.float32), G.astype(np.float32), cv2.GaussianBlur(spark, (0, 0), 1.2) * 6)
    return _K4['frost']


def t_frost(t, A, a, end):
    """Loop: frost creeps in over the lens from the edges (ferns of ice crystals, the picture behind going cold
    and blurred, sparkles); when it has frozen over, a swipe wipes a clear streak through it and the first shot is
    behind the glass."""
    if t >= end: return layer(0, t - DUR)
    pat, G, spark = _frost_assets()
    tw = end - 0.24
    front = 1.12 * ease(clamp((t - a) / (tw - a))) ** 0.8
    cover = ss(front, front - 0.12, G)
    base = layer(A, t)
    cold = blur(base, 14 * cover.mean() + 2) * np.float32([1.15, 1.0, 0.85])
    cold = luma(cold)[..., None] * 0.5 + cold * 0.5
    ice = np.float32([1.0, 0.95, 0.86]) * (0.50 + 0.65 * pat)[..., None]
    tw_ = 0.5 + 0.5 * math.sin(t * 40)
    m = (cover * (0.30 + 0.62 * pat))[..., None]
    f = base * (1 - cover[..., None]) + cold * cover[..., None]
    f = f * (1 - m) + ice * m + (spark * cover * tw_)[..., None]
    if t >= tw:
        u = clamp((t - tw) / (end - tw)); xf = lerp(-0.2 * W, 1.3 * W, eout(clamp(u / 0.45)))
        yl = 0.45 * H + 0.10 * H * np.sin(XX / W * 3.0); hw = lerp(110, 1500, ein(clamp((u - 0.25) / 0.75)))
        d = np.abs(YY - yl)
        dd = np.sqrt(d ** 2 + np.clip(XX - (xf - hw), 0, None) ** 2) + 18 * (_ring_noise(120, s=1) - 0.5)   # rounded palm end
        wipe = ss(hw + 6, hw - 6, dd)
        wet = np.exp(-((dd - hw) / 7) ** 2) * 0.5
        f0 = layer(0, t - DUR)
        f = f * (1 - wipe[..., None]) + f0 * wipe[..., None] + wet[..., None] * np.float32([1.0, 0.98, 0.95])
    return f


# ============================================================ Dexter kit VI: "Miami Night Drive" (Mareux, The Perfect Girl)
# Restrained, cinematic graphics from a night drive through Miami: a palm trunk passing the lens, the outrun
# horizon on the drop, a tinted car window rolling down, sodium streetlights sweeping over, rain and a windshield
# wiper on the crash, and the rear-view mirror for the loop ("objects in mirror are closer than they appear").
NEON_PINK = np.float32([0.75, 0.25, 1.0]); NEON_CYAN = np.float32([1.0, 0.85, 0.20]); SODIUM = np.float32([0.30, 0.62, 1.0])


# ---- a palm trunk passes close to the lens
def _trunk_tex():
    if 'trunk' in _K4: return _K4['trunk']
    w, h = 620, H + 200; yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    n = cv2.resize(fbm(h // 16, w // 16, 4, 61, 4), (w, h), interpolation=cv2.INTER_CUBIC)
    u = (xx - w / 2) / (w / 2); cyl = np.sqrt(np.clip(1 - u * u, 0, 1))
    a1 = np.mod((yy + xx * 0.55) / 46, 1.0); a2 = np.mod((yy - xx * 0.55) / 46, 1.0)                # criss-cross leaf-base scars
    scars = np.exp(-((a1 - 0.5) / 0.08) ** 2) + np.exp(-((a2 - 0.5) / 0.08) ** 2)
    v = (0.06 + 0.07 * cyl + 0.05 * (n - 0.5)) * (1 - 0.45 * np.clip(scars, 0, 1)) * (0.55 + 0.45 * cyl)
    rgb = v[..., None] * np.float32([0.85, 0.85, 1.0])
    rl = np.exp(-((u + 0.93) / 0.06) ** 2); rr = np.exp(-((u - 0.93) / 0.06) ** 2)                  # neon rim lights: pink left, cyan right
    rgb = rgb + rl[..., None] * NEON_PINK * 0.55 + rr[..., None] * NEON_CYAN * 0.45
    al = ss(1.0, 0.97, np.abs(u))
    _K4['trunk'] = (rgb.astype(np.float32), al.astype(np.float32))
    return _K4['trunk']


def t_pass(t, c, A, B, a, b):
    """A palm trunk passes in front of the lens (bark scars, pink and cyan neon rim light, defocused); the next shot
    is revealed behind it as it crosses."""
    rgb, al = _trunk_tex(); tw = rgb.shape[1]
    u = clamp((t - a) / (b - a)); x0 = lerp(W + 30, -tw - 30, u); mid = x0 + tw / 2
    M = np.float32([[1, 0, x0], [0, 1, -100]])
    tr = cv2.warpAffine(rgb, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    ta = cv2.warpAffine(al, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    tr = cv2.blur(cv2.GaussianBlur(tr, (0, 0), 3), (25, 1)); ta = cv2.blur(cv2.GaussianBlur(ta, (0, 0), 3), (25, 1))
    k = ss(mid - 40, mid + 40, XX)[..., None]                                                    # behind the trunk: next shot on its trailing side
    under = layer(A, t) * (1 - k) + layer(B, t) * k
    return under * (1 - ta[..., None]) + tr


# ---- the outrun horizon (the drop)
def _outrun_world(t):
    hz = 0.44 * H
    sky = np.zeros((H, W, 3), np.float32); yn = np.clip(YY / hz, 0, 1)[..., None]
    sky[:] = np.float32([0.16, 0.03, 0.07]) * (1 - yn) + np.float32([0.45, 0.10, 0.55]) * yn
    rng = np.random.default_rng(3); st = np.zeros((H, W), np.float32)
    for _ in range(60): cv2.circle(st, (int(rng.uniform(0, W)), int(rng.uniform(0, hz * 0.8))), 1, float(rng.uniform(0.3, 0.9)), -1)
    sky += st[..., None] * (0.6 + 0.4 * math.sin(t * 7))
    R = 0.36 * W; cx, cy = W / 2, hz - 0.03 * H
    d = np.sqrt((XX - cx) ** 2 + (YY - cy) ** 2)
    gy = np.clip((YY - (cy - R)) / (2 * R), 0, 1)[..., None]
    sun = np.float32([0.35, 0.90, 1.0]) * (1 - gy) + np.float32([0.65, 0.20, 1.0]) * gy
    ph = (YY - (cy - 0.35 * R)) / (hz - (cy - 0.35 * R)); gaps = (ph > 0) & (np.mod(YY - cy + t * 40, 40) < 3 + 16 * np.clip(ph, 0, 1))
    disc = ss(R + 1.5, R - 1.5, d) * (1 - gaps) * (YY < hz)
    halo = np.exp(-np.clip(d - R, 0, None) / 90) * 0.35 * (YY < hz)
    f = sky * (1 - disc[..., None]) + sun * disc[..., None] + halo[..., None] * np.float32([0.6, 0.35, 1.0])
    fl = YY >= hz                                                                                 # the grid floor
    depth = np.maximum(YY - hz, 0) + 1.0
    zf = 9000 / depth; zl = np.mod(zf - t * 14, 6)
    hl = np.exp(-(np.minimum(zl, 6 - zl) / (2.2 * 9000 / depth ** 2)) ** 2)
    xw = (XX - W / 2) * zf / 60; vl = np.exp(-(np.abs(np.mod(xw + 5, 10) - 5) / (1.8 * zf / 60)) ** 2)
    grid = np.clip(hl + vl, 0, 1) * fl * (1 - np.exp(-depth / 50))
    floor = np.float32([0.10, 0.02, 0.08]) * fl[..., None]
    f = f * (1 - fl[..., None]) + floor + (grid * (0.5 + 0.5 * np.exp(-depth / 400)))[..., None] * NEON_PINK
    f = f + (cv2.GaussianBlur(grid.astype(np.float32), (0, 0), 6) * 0.6)[..., None] * NEON_PINK
    f = f + (np.exp(-((YY - hz) / 5) ** 2) * 0.9 + np.exp(-((YY - hz) / 40) ** 2) * 0.25)[..., None] * NEON_PINK
    return f


def t_outrun(t, c, A, B, a, b):
    """On the drop the camera tilts down off the shot onto the outrun horizon (striped sun, neon grid racing at the
    lens), the sun pulses on the crash, then the camera tilts back up into the next shot."""
    t1, t2 = c - 0.05, c + 0.12
    world = _outrun_world(t)
    if t < t1:
        e = ease(clamp((t - a) / (t1 - a))); dy = -e * H
        fa = layer(A, t); M = np.float32([[1, 0, 0], [0, 1, dy]])
        f = cv2.warpAffine(fa, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        Mw = np.float32([[1, 0, 0], [0, 1, dy + H]])
        wv = cv2.warpAffine(world, Mw, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return f + wv * (YY >= H + dy)[..., None]
    if t < t2:
        return world * (1 + 0.35 * math.exp(-max(0.0, t - c) * 14) * (t >= c))
    e = ease(clamp((t - t2) / (b - t2))); dy = e * H
    Mw = np.float32([[1, 0, 0], [0, 1, dy]])
    wv = cv2.warpAffine(world, Mw, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    fb = layer(B, t); Mb = np.float32([[1, 0, 0], [0, 1, dy - H]])
    bv = cv2.warpAffine(fb, Mb, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return wv * (YY >= dy)[..., None] + bv * (YY < dy)[..., None]


# ---- a tinted car window rolls up, then down
def t_window(t, c, A, B, a, b):
    """A tinted car window rolls up over the shot (neon reflections sliding across the glass, rubber seal, chrome
    trim on the door), then rolls down on the beat and the next shot is inside the car."""
    door_h = 0.16 * H
    din = eout(clamp((t - a) / 0.14)) * (1 - ein(clamp((t - (b - 0.12)) / 0.12)))
    top_door = H - door_h * din
    if t < c: g = eout(clamp((t - (a + 0.06)) / (c - 0.06 - (a + 0.06)))); base = layer(A, t)
    else: g = 1 - eout(clamp((t - c) / (b - 0.10 - c))); base = layer(B, t)
    gtop = lerp(top_door, -40, g)                                                                # top edge of the glass
    gm = ss(gtop - 1.5, gtop + 1.5, YY) * (YY < top_door)
    refl = cv2.GaussianBlur(base[::-1], (0, 0), 18) * 0.18
    band = 0.0
    for k, (sp, col) in enumerate(((1.0, NEON_PINK), (1.6, NEON_CYAN), (0.7, SODIUM))):
        pos = ((t * 900 * sp + k * 700) % (W + H + 600)) - 300
        band = band + np.exp(-(((XX + YY * 0.6) - pos) / (60 + 40 * k)) ** 2)[..., None] * col * 0.22
    glass = base * 0.28 * np.float32([1.15, 1.0, 0.9]) + refl + band + 0.03
    f = base * (1 - gm[..., None]) + glass * gm[..., None]
    seal = np.exp(-((YY - gtop - 6) / 5) ** 2) * (gtop < top_door - 4)
    f = f * (1 - seal[..., None] * 0.8) + seal[..., None] * 0.55
    dm = (YY >= top_door)[..., None]
    door = np.float32([0.05, 0.04, 0.05]) + (np.exp(-((YY - top_door - 14) / 5) ** 2) * 0.75)[..., None] * np.float32([1.0, 0.95, 0.95])
    door = door + (np.exp(-((YY - top_door - 26) / 18) ** 2) * 0.10)[..., None]
    return f * (1 - dm) + door * dm


# ---- sodium streetlights sweep over the shot
def t_streetlights(t, c, A, B, a, b):
    """Driving under streetlights: warm sodium light sweeps down over the shot on the beat, three lamps; the last
    one flares and the next shot is underneath it."""
    passes = (c - HALF, c - HALF / 2, c)
    base = layer(A, t) if t < c else layer(B, t)
    light = np.zeros((H, W), np.float32)
    for k, tp in enumerate(passes):
        u = (t - tp + 0.18) / 0.36
        if not 0 <= u <= 1: continue
        yc = lerp(-0.4 * H, 1.4 * H, u)
        light = np.maximum(light, np.exp(-((YY - yc) / (0.22 * H)) ** 2) * (0.8 if k < 2 else 1.0))
    l = luma(base)[..., None]
    f = base * (0.75 + 0.25 * (1 - light[..., None])) + (light[..., None] * (0.25 + 0.75 * l)) * SODIUM * 0.9
    if abs(t - c) < 0.08: f = f + 0.25 * (1 - abs(t - c) / 0.08) * SODIUM
    return f


# ---- rain on the windshield and the wiper (the crash)
def _rain_assets():
    if 'rain' in _K4: return _K4['rain']
    rng = np.random.default_rng(17); h = np.zeros((H, W), np.float32); birth = np.full((H, W), 9.0, np.float32)
    for _ in range(320):
        x, y = rng.uniform(0, W), rng.uniform(0, H); r = rng.uniform(8, 24) * (1 if rng.random() > 0.15 else 1.7)
        bt = rng.uniform(0, 1)
        x0, x1, y0, y1 = int(max(0, x - r - 2)), int(min(W, x + r + 2)), int(max(0, y - r * 1.3 - 2)), int(min(H, y + r * 1.3 + 2))
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32); q = ((xx - x) / r) ** 2 + ((yy - y) / (r * 1.25)) ** 2
        cap = np.sqrt(np.clip(1 - q, 0, 1)) * r * 0.35
        m = cap > h[y0:y1, x0:x1]
        h[y0:y1, x0:x1] = np.where(m, cap, h[y0:y1, x0:x1]); birth[y0:y1, x0:x1] = np.where(m & (cap > 0), bt, birth[y0:y1, x0:x1])
    gy, gx = np.gradient(h)
    _K4['rain'] = (h, gx.astype(np.float32), gy.astype(np.float32), birth)
    return _K4['rain']


def t_wiper(t, c, A, B, a, b):
    """Rain beads up on the windshield over the shot (each drop a little lens), then on the crash the wiper sweeps
    across and the next shot is clear behind it."""
    h, gx, gy, birth = _rain_assets()
    ts, te = c - 0.10, c + 0.16
    P = (0.06 * W, 1.10 * H); Lw = 2500.0
    th0, th1 = math.radians(-182), math.radians(-2)
    thw = lerp(th0, th1, ease(clamp((t - ts) / (te - ts))))
    ang = np.arctan2(YY - P[1], XX - P[0])
    swept = ss(thw + 0.01, thw - 0.01, ang) * (t >= ts)
    grown = clamp((t - a) / (ts - a + 1e-3))
    vis = ((birth <= grown) & (h > 0)).astype(np.float32)
    fa = layer(A, t); fb = layer(B, t)
    k = 3.5
    mx = (XX - gx * k * 9 * vis).astype(np.float32); my = (YY - gy * k * 9 * vis).astype(np.float32)
    fr = cv2.remap(fa, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    nl = np.clip(-gx * 0.6 - gy * 0.8, 0, None)
    drops = fr * (0.80 + 0.35 * vis[..., None]) + (vis * (nl ** 2) * 1.8)[..., None]
    rim = vis * ss(0.0, 1.5, h) * (h < 1.6)
    drops = drops * (1 - 0.35 * rim[..., None])
    wet = fa * (1 - vis[..., None]) + drops * vis[..., None]
    f = wet * (1 - swept[..., None]) + fb * swept[..., None]
    if ts <= t < te + 0.06:                                                                      # the wiper arm and blade
        arm = np.zeros((H, W), np.float32); dx, dy = math.cos(thw), math.sin(thw)
        cv2.line(arm, (int(P[0]), int(P[1])), (int(P[0] + dx * Lw), int(P[1] + dy * Lw)), 1.0, 26, cv2.LINE_AA)
        nx, ny = -dy, dx
        hi = np.zeros((H, W), np.float32)
        cv2.line(hi, (int(P[0] + nx * 8), int(P[1] + ny * 8)), (int(P[0] + dx * Lw + nx * 8), int(P[1] + dy * Lw + ny * 8)), 1.0, 4, cv2.LINE_AA)
        sh = cv2.GaussianBlur(np.roll(arm, 18, 0), (0, 0), 14) * 0.5
        f = f * (1 - sh[..., None]) * (1 - arm[..., None]) + arm[..., None] * 0.035 + hi[..., None] * 0.30
    return f


# ---- rear-view mirror loop
def _mirror_assets():
    if 'mirror' in _K4: return _K4['mirror']
    gw, gh = 900, 300                                                                           # glass size at rest (3:1)
    m = _text_mask('OBJECTS IN MIRROR ARE CLOSER THAN THEY APPEAR', INTER_SEMI, 22)
    _K4['mirror'] = (gw, gh, m)
    return _K4['mirror']


def t_mirror(t, A, a, end):
    """Loop: a rear-view mirror drops into frame over the last shot; in its glass is what's behind you, the first
    shot of the edit (OBJECTS IN MIRROR ARE CLOSER THAN THEY APPEAR); the camera pushes into the glass until it is
    the frame."""
    if t >= end: return layer(0, t - DUR)
    gw, gh, txt = _mirror_assets()
    drop = eout(clamp((t - a) / 0.28)); td = a + 0.40; u = clamp((t - td) / (end - td))
    zmax = H / gh; z = math.exp(math.log(zmax) * ein(u) ** 0.85)
    cx = W / 2; cy0 = lerp(-0.25 * H, 0.30 * H, drop); cy = lerp(cy0, H / 2, ease(u))
    rot = math.radians(6) * (1 - drop) + math.radians(1.5) * math.sin((t - a) * 14) * math.exp(-(t - a) * 6)
    rot *= 1 - ease(u)
    fa = layer(A, t); f = fa * (1 - 0.25 * drop)
    hw, hh = gw / 2 * z, gh / 2 * z; rad = 70 * z
    ca, sa = math.cos(rot), math.sin(rot)
    lx = (XX - cx) * ca + (YY - cy) * sa; ly = -(XX - cx) * sa + (YY - cy) * ca
    def rrect(hx, hy, r):
        qx = np.maximum(np.abs(lx) - (hx - r), 0); qy = np.maximum(np.abs(ly) - (hy - r), 0)
        return np.sqrt(qx * qx + qy * qy) - r + np.minimum(np.maximum(np.abs(lx) - hx, np.abs(ly) - hy), 0) * 0
    dh = rrect(hw + 26 * z, hh + 26 * z, rad + 20 * z)                                          # housing
    dg = rrect(hw, hh, rad)
    housing = ss(1, -1, dh); glassm = ss(1, -1, dg)
    stem = (np.abs(lx) < 30 * z) & (ly < -hh - 20 * z) & (ly > -hh - 20 * z - 0.4 * H)
    hm = np.maximum(housing, stem.astype(np.float32))
    sh = cv2.GaussianBlur(np.roll(np.roll(hm, int(24 * z), 0), int(14 * z), 1), (0, 0), 16) * 0.5 * (1 - u)
    f = f * (1 - sh[..., None])
    hcol = np.float32([0.07, 0.065, 0.07]) + (np.exp(-((ly + hh + 14 * z) / (8 * z)) ** 2) * 0.12)[..., None]
    f = f * (1 - hm[..., None]) + hcol * hm[..., None]
    sc = z * gh / H                                                                             # the first shot, as seen in the glass
    Mg = np.float32([[sc * ca, -sc * sa, 0], [sc * sa, sc * ca, 0]])
    Mg[:, 2] = np.array([cx, cy]) - Mg[:, :2] @ np.array([W / 2, H / 2])
    f0 = layer(0, t - DUR)
    g0 = cv2.warpAffine(f0, Mg, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    gm0 = cv2.warpAffine(np.ones((H, W), np.float32), Mg, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    scb = z * gw / W; Mb = np.float32([[scb * ca, -scb * sa, 0], [scb * sa, scb * ca, 0]]); Mb[:, 2] = np.array([cx, cy]) - Mb[:, :2] @ np.array([W / 2, H / 2])
    gb = cv2.warpAffine(cv2.resize(cv2.GaussianBlur(cv2.resize(f0, (W // 8, H // 8)), (0, 0), 4), (W, H)), Mb, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE) * 0.55
    sg_ = 8 * (1 - u)
    m0 = (cv2.GaussianBlur(gm0, (0, 0), sg_) * gm0 if sg_ > 0.3 else gm0)[..., None]
    g0 = gb * (1 - m0) + g0 * m0
    sheen = (np.exp(-(((lx + ly * 1.4) / max(1.0, z) - lerp(-600, 600, clamp((t - a - 0.15) / 0.35))) / 70) ** 2) * 0.22 * (1 - u))[..., None]
    gl = g0 * (0.92 + 0.08 * (1 - u)) + sheen
    f = f * (1 - glassm[..., None]) + gl * glassm[..., None]
    bev = np.exp(-((dg + 3 * z) / (3 * z)) ** 2) * (1 - u) * 0.5
    f = f + bev[..., None]
    if u < 0.5:
        tm = txt; th_, tw_ = tm.shape; s2 = z
        Mt = np.float32([[s2 * ca, -s2 * sa, 0], [s2 * sa, s2 * ca, 0]]); Mt[:, 2] = np.array([cx, cy]) + Mt[:, :2] @ np.array([-tw_ / 2, gh / 2 - 46])
        tl = cv2.warpAffine(tm, Mt, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0) * glassm * 0.55 * (1 - u / 0.5)
        f = f * (1 - tl[..., None]) + tl[..., None] * 0.95
    return f


# ---- the outrun sun, as a window (the drop)
def _sky_floor(t, hz, speed=1.0):
    sky = np.zeros((H, W, 3), np.float32); yn = np.clip(YY / max(hz, 1.0), 0, 1)[..., None]
    sky[:] = np.float32([0.16, 0.03, 0.07]) * (1 - yn) + np.float32([0.45, 0.10, 0.55]) * yn
    rng = np.random.default_rng(3); st = np.zeros((H, W), np.float32)
    for _ in range(60): cv2.circle(st, (int(rng.uniform(0, W)), int(rng.uniform(0, H))), 1, float(rng.uniform(0.3, 0.9)), -1)
    sky = sky + (st * (YY < hz))[..., None] * 0.7
    fl = (YY >= hz).astype(np.float32)
    depth = np.maximum(YY - hz, 0) + 1.0
    zf = 9000 / depth; zl = np.mod(zf - t * 14 * speed, 6)
    hl = np.exp(-(np.minimum(zl, 6 - zl) / (2.2 * 9000 / depth ** 2)) ** 2)
    xw = (XX - W / 2) * zf / 60; vl = np.exp(-(np.abs(np.mod(xw + 5, 10) - 5) / (1.8 * zf / 60)) ** 2)
    grid = (np.clip(hl + vl, 0, 1) * fl * (1 - np.exp(-depth / 50))).astype(np.float32)
    f = sky * (1 - fl[..., None]) + np.float32([0.10, 0.02, 0.08]) * fl[..., None]
    f = f + (grid * (0.5 + 0.5 * np.exp(-depth / 400)))[..., None] * NEON_PINK + (cv2.GaussianBlur(grid, (0, 0), 6) * 0.6)[..., None] * NEON_PINK
    f = f + (np.exp(-((YY - hz) / 5) ** 2) * 0.9 + np.exp(-((YY - hz) / 40) ** 2) * 0.25)[..., None] * NEON_PINK
    return f


def t_sunwindow(t, c, A, B, a, b):
    """The drop: the camera tilts down off the shot onto the outrun horizon, where the striped sun is made of the
    next shot; the camera flies over the racing grid toward it, the sun rises and its stripes part, and on the
    drop it fills the frame as the next shot."""
    t1 = a + 0.16
    fly = ease(clamp((t - t1) / (b - 0.04 - t1)))
    hz = lerp(0.44 * H, 1.25 * H, fly ** 1.4)                                                    # the camera tilts up as it closes in
    R0 = 0.33 * W; z = math.exp(math.log(RFULL / R0) * ein(clamp((t - t1) / (b - 0.04 - t1))) ** 0.9)
    R = R0 * z; cx = W / 2; cy = lerp(0.44 * H - 0.05 * H, H / 2, fly)
    world = _sky_floor(t, hz, 1 + 3 * fly)
    d = np.sqrt((XX - cx) ** 2 + (YY - cy) ** 2)
    gy = np.clip((YY - (cy - R)) / (2 * R), 0, 1)[..., None]
    grad = np.float32([0.35, 0.90, 1.0]) * (1 - gy) + np.float32([0.65, 0.20, 1.0]) * gy
    f, _ = _live_circle(world, layer(B, t), cx, cy, R, feather=1.5, src=(W / 2, lerp(0.36 * H, H / 2, fly)))   # the sun frames his eyes
    inside = ss(R + 1.5, R - 1.5, d)[..., None]
    tint = 1 - ease(clamp((t - (c - 0.02)) / (b - c)))                                           # the sun's colour drains off the shot
    f = f * (1 - inside * tint * 0.55) + (f * grad * 1.6) * inside * tint * 0.55
    ph = (YY - (cy - 0.35 * R)) / max(1.0, (cy + R) - (cy - 0.35 * R))
    gap_w = (3 + 16 * np.clip(ph, 0, 1)) * z * (1 - fly) ** 1.5
    gaps = ((ph > 0) & (np.mod(YY - cy + t * 40 * z, 40 * z) < gap_w)).astype(np.float32)[..., None] * inside
    f = f * (1 - gaps) + world * gaps
    halo = (np.exp(-np.clip(d - R, 0, None) / (90 * z)) * 0.35 * (d > R) * (1 - fly))[..., None]
    f = f + halo * np.float32([0.6, 0.35, 1.0])
    f = f * (1 - (ss(hz - 2, hz + 2, YY) * (d <= R))[..., None]) + (world * ss(hz - 2, hz + 2, YY)[..., None]) * (d <= R)[..., None]
    if t < t1:                                                                                   # tilt down off the outgoing shot
        e = ease(clamp((t - a) / (t1 - a))); dy = -e * H
        fa = cv2.warpAffine(layer(A, t), np.float32([[1, 0, 0], [0, 1, dy]]), (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        fw = cv2.warpAffine(f, np.float32([[1, 0, 0], [0, 1, dy + H]]), (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return fa * (YY < H + dy)[..., None] + fw * (YY >= H + dy)[..., None]
    if c <= t < c + 0.10: f = f * (1 + 0.25 * (1 - (t - c) / 0.10))
    return f


# ---- rear-view mirror (mid-edit)
def t_mirror2(t, c, A, B, a, b):
    """A rear-view mirror swings down into the shot (glossy housing, silver-rimmed glass); in its glass is the road
    behind -- the next shot -- and the camera pushes into the glass until it is the frame."""
    gw, gh = 900, 300
    drop = eout(clamp((t - a) / 0.22)); td = c - 0.12; u = clamp((t - td) / (b - td))
    zmax = H / gh; z = math.exp(math.log(zmax) * ease(u))
    cx = W / 2; cy0 = lerp(-0.25 * H, 0.30 * H, drop); cy = lerp(cy0, H / 2, ease(u))
    rot = (math.radians(7) * (1 - drop) + math.radians(2.0) * math.sin((t - a) * 16) * math.exp(-(t - a) * 7)) * (1 - ease(u))
    fa = layer(A, t); f = fa * (1 - 0.30 * drop * (1 - u))
    hw, hh = gw / 2 * z, gh / 2 * z; rad = 80 * z
    ca, sa = math.cos(rot), math.sin(rot)
    lx = (XX - cx) * ca + (YY - cy) * sa; ly = -(XX - cx) * sa + (YY - cy) * ca
    def rdist(hx, hy, r):
        qx = np.maximum(np.abs(lx) - (hx - r), 0); qy = np.maximum(np.abs(ly) - (hy - r), 0)
        return np.sqrt(qx * qx + qy * qy) - r
    dh = rdist(hw + 30 * z, hh + 30 * z, rad + 24 * z); dg = rdist(hw, hh, rad)
    housing = ss(1, -1, dh); glassm = ss(1, -1, dg)
    stem = ((np.abs(lx) < 26 * z) & (ly < -hh - 20 * z) & (ly > -hh - 20 * z - 0.4 * H)).astype(np.float32)
    hm = np.maximum(housing, stem)
    sh = cv2.GaussianBlur(np.roll(np.roll(hm, int(26 * z), 0), int(16 * z), 1), (0, 0), 18) * 0.55 * (1 - u)
    f = f * (1 - sh[..., None])
    rimd = np.clip(-dh / (30 * z), 0, 1)                                                         # glossy housing: bevel highlight on the top edge
    hcol = 0.05 + 0.10 * np.exp(-((ly + hh + 20 * z) / (10 * z)) ** 2) + 0.06 * (1 - rimd)
    f = f * (1 - hm[..., None]) + (hcol[..., None] * np.float32([1.0, 0.98, 1.0])) * hm[..., None]
    fb = layer(B, t)
    sc_end = z * gh / H; sc_cov = z * gw / W * 1.02; sc = lerp(sc_cov, sc_end, ease(u))
    src_c = np.array([W / 2, lerp(0.38 * H, H / 2, ease(u))])                                    # the glass frames the face line
    Mg = np.float32([[sc * ca, -sc * sa, 0], [sc * sa, sc * ca, 0]]); Mg[:, 2] = np.array([cx, cy]) - Mg[:, :2] @ src_c
    g = cv2.warpAffine(fb, Mg, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    gmask = cv2.warpAffine(np.ones((H, W), np.float32), Mg, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if u < 0.97: gmask = cv2.GaussianBlur(gmask, (0, 0), 14 * (1 - u) + 0.5) * gmask
    gmask = gmask[..., None]
    scb = z * gw / W * 1.02; Mb = np.float32([[scb * ca, -scb * sa, 0], [scb * sa, scb * ca, 0]]); Mb[:, 2] = np.array([cx, cy]) - Mb[:, :2] @ src_c
    bg = cv2.warpAffine(cv2.GaussianBlur(fb, (0, 0), 18), Mb, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE) * 0.55
    g = bg * (1 - gmask) + g * gmask / np.maximum(cv2.warpAffine(np.ones((H, W), np.float32), Mg, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)[..., None], 1e-3)
    sheen = np.exp(-(((lx + ly * 1.6) / max(1.0, z) - lerp(-700, 700, clamp((t - a - 0.10) / 0.35))) / 60) ** 2) * 0.20 * (1 - u)
    g = g * (0.88 + 0.12 * u) + sheen[..., None]
    f = f * (1 - glassm[..., None]) + g * glassm[..., None]
    rim = np.exp(-((dg + 2 * z) / (2.5 * z)) ** 2) * (1 - u)                                       # silver rim of the glass
    f = f + rim[..., None] * 0.55
    return f


# ---- split-flap board (the crash)
def t_flaps(t, c, A, B, a, b, rows=8, cols=3):
    """On the crash the frame flips over like a split-flap departure board: module by module, top to bottom, each
    top flap falls over its hinge and the next shot clacks into place; the board's gaps show while it runs."""
    fa = layer(A, t); fb = layer(B, t); f = fa.copy(); h = H / rows
    act = clamp((t - a) / 0.05) * (1 - clamp((t - (b - 0.06)) / 0.06))
    shade = np.ones((H, 1, 1), np.float32)
    for r in range(rows):
        t0 = a + 0.03 + r * 0.03; ph = clamp((t - t0) / 0.16)
        y0 = int(r * h); y1 = int((r + 1) * h); yh = int((y0 + y1) / 2)
        if ph <= 0: continue
        if ph >= 1: f[y0:y1] = fb[y0:y1]; continue
        f[y0:yh] = fb[y0:yh]; f[yh:y1] = fa[yh:y1]
        k = math.cos(math.pi * ph)
        if k > 0:                                                                                # A's top flap folding down to the hinge
            hh = max(1, int(round((yh - y0) * k))); src = cv2.resize(fa[y0:yh], (W, hh), interpolation=cv2.INTER_AREA)
            f[yh - hh:yh] = src * (0.45 + 0.55 * k); f[yh - hh:yh - hh + 2] += 0.35 * k              # lit top edge
            sh = int((y1 - yh) * 0.5 * (1 - k)); shade[yh:yh + sh] *= 0.55
        else:                                                                                    # its back, B's bottom half, landing
            hh = max(1, int(round((y1 - yh) * -k))); src = cv2.resize(fb[yh:y1], (W, hh), interpolation=cv2.INTER_AREA)
            f[yh:yh + hh] = src * (0.45 + 0.55 * -k); f[yh + hh - 2:yh + hh] += 0.30 * -k
    f = f * (1 - (1 - shade) * act)
    gap = np.zeros((H, W), np.float32)
    for r in range(rows + 1):
        y = int(r * h); gap[max(0, y - 4):y + 4] = 1.0
        if r < rows: yh = int((r + 0.5) * h); gap[yh - 1:yh + 1] = 0.7
    for j in range(1, cols): x = int(W / cols * j); gap[:, x - 4:x + 4] = 1.0
    return f * (1 - (gap * act * 0.9)[..., None])


# ---- sunglasses reflection loop
def _lens_masks(f):
    """The two dark lenses of the sunglasses in a frame: soft masks plus the bigger lens's centre and size."""
    q = 4; sm = cv2.resize(f, (W // q, H // q), interpolation=cv2.INTER_AREA); l = luma(sm)
    m = ((l < 0.11) & (np.arange(H // q)[:, None] > 0.08 * H / q) & (np.arange(H // q)[:, None] < 0.50 * H / q)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)); m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(m)
    if n <= 1: return None
    order = sorted(range(1, n), key=lambda i: -stats[i, cv2.CC_STAT_AREA])[:2]
    mm = np.isin(lab, order).astype(np.float32)
    big = order[0]; x, y, w, h_ = stats[big, :4]
    mask = cv2.GaussianBlur(cv2.resize(mm, (W, H), interpolation=cv2.INTER_LINEAR), (0, 0), 3)
    return mask, (cent[big][0] * q, cent[big][1] * q), (w * q, h_ * q)


def _lens_track(A, a, end):
    key = ('lens', A, round(a, 3))
    if key in _K4: return _K4[key]
    ts = np.arange(a - 0.05, end + 0.05, 1 / 60.0); cs = []; ss_ = []
    for ti in ts:
        r = _lens_masks(layer(A, ti))
        cs.append(r[1] if r else (np.nan, np.nan)); ss_.append(r[2] if r else (np.nan, np.nan))
    cs = np.array(cs, np.float64); ss_ = np.array(ss_, np.float64)
    for arr in (cs, ss_):
        for j in range(2):
            v = arr[:, j]; ok = ~np.isnan(v)
            if ok.sum() == 0: v[:] = (W / 2 if j == 0 else 0.3 * H); continue
            v[~ok] = np.interp(np.flatnonzero(~ok), np.flatnonzero(ok), v[ok])
            arr[:, j] = np.convolve(np.pad(v, 7, mode='edge'), np.ones(15) / 15, 'valid')
    _K4[key] = (ts, cs, ss_)
    return _K4[key]


def t_shades(t, A, a, end):
    """Loop: the edit ends on Dexter in his sunglasses. The camera eases in, the first shot of the edit surfaces as a
    reflection in the dark lenses (curved, cool, a sheen gliding across), then the camera dives into one lens with a
    radial motion blur until the reflection is the frame, and the edit starts again."""
    if t >= end: return layer(0, t - DUR)
    fa = layer(A, t); f0 = layer(0, t - DUR)
    lm = _lens_masks(fa)
    ts, cs, szs = _lens_track(A, a, end)
    lx0, ly0 = np.interp(t, ts, cs[:, 0]), np.interp(t, ts, cs[:, 1]); lw, lh = np.interp(t, ts, szs[:, 0]), np.interp(t, ts, szs[:, 1])
    mask = lm[0] if lm is not None else np.zeros((H, W), np.float32)
    td = a + 0.38; u = clamp((t - td) / (end - td)); show = ease(clamp((t - a) / 0.32))
    zend = min(16.0, 1.4 * max(W / max(lw, 1), H / max(lh, 1)))
    zp = 1 + 0.07 * ease(clamp((t - a) / 0.38))                                                   # the slow push-in before the dive
    z = zp * math.exp(math.log(zend / zp) * ease(u) ** 1.6)
    cx, cy = lerp(lx0, W / 2, ease(u)), lerp(ly0, H / 2, ease(u))
    M = np.float32([[z, 0, cx - z * lx0], [0, z, cy - z * ly0]])
    fz = cv2.warpAffine(fa, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    mz = cv2.warpAffine(mask, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    sc = lerp(1.15 * lw * z / W, 1.0, ease(u) ** 0.8)
    rr = np.sqrt(((XX - cx) / (lw * z * 0.6)) ** 2 + ((YY - cy) / (lh * z * 0.6)) ** 2)
    bulge = 1 + 0.16 * (1 - u) * np.clip(1 - rr ** 2, 0, 1)
    mx = (W / 2 + (XX - cx) / (sc * bulge)).astype(np.float32); my = (H / 2 + (YY - cy) / (sc * bulge)).astype(np.float32)
    refl = cv2.remap(f0, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    refl = refl * lerp(0.72, 1.0, ease(u)) * np.float32([1.10, 1.0, 0.92]) ** (1 - u)
    sweep = lerp(-0.6, 1.6, clamp((t - a - 0.05) / 0.5))
    refl = refl + (np.exp(-((((XX - cx) / (lw * z) + (YY - cy) / (lh * z) * 0.5) - sweep) / 0.12) ** 2) * 0.22 * (1 - u))[..., None]
    k = (mz * show)[..., None]
    land = ss(0.62, 0.92, u)                                                                      # resolve onto the first shot before the rim can show
    k = k * (1 - land) + land
    vig = 1 - 0.35 * ease(clamp(u / 0.5)) * (1 - land) * np.clip(rr / 3, 0, 1)[..., None]
    f = (fz * (1 - k) + refl * k) * vig
    sp = (ease(u) - ease(max(0.0, u - 0.08))) * (1 - land)                                        # radial motion blur while diving
    if sp > 0.01:
        acc = f.copy()
        for j in (1, 2, 3):
            s_ = 1 + 0.06 * sp * j * 4; Mr = np.float32([[s_, 0, cx * (1 - s_)], [0, s_, cy * (1 - s_)]])
            acc += cv2.warpAffine(f, Mr, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        f = acc / 4
    return f


# ---- neon line-art trace (the drop)
def _neon_edges(f):
    """Clean outline drawing of a frame: Canny on a softened half-res luminance, small specks removed."""
    q = 2; g = cv2.resize(luma(f), (W // q, H // q), interpolation=cv2.INTER_AREA)
    g8 = np.clip(cv2.bilateralFilter(cv2.GaussianBlur(g, (0, 0), 2.4), 9, 0.12, 9) * 255, 0, 255).astype(np.uint8)
    e = cv2.Canny(g8, 34, 96)
    n, lab, st, _ = cv2.connectedComponentsWithStats((e > 0).astype(np.uint8), connectivity=8)
    span = np.maximum(st[:, cv2.CC_STAT_WIDTH], st[:, cv2.CC_STAT_HEIGHT])
    keep = np.zeros(n, bool); keep[1:] = (st[1:, cv2.CC_STAT_AREA] >= 45) & (span[1:] >= 22)
    m = keep[lab].astype(np.float32)
    m = cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR)
    return np.clip(cv2.GaussianBlur(m, (0, 0), 0.8) * 1.6, 0, 1)


def _neon_paint(base, edges, k, t):
    """Edges as neon tubes: a white-hot core, a coloured tube and a soft glow; pink at the top, cyan below."""
    yn = (YY / H)[..., None]; col = NEON_PINK * (1 - yn) + NEON_CYAN * yn
    glow = cv2.GaussianBlur(edges, (0, 0), 7) * 1.6 + cv2.GaussianBlur(edges, (0, 0), 22) * 0.9
    return base + (edges * k)[..., None] * (col * 0.8 + 0.45) + (glow * k)[..., None] * col


def t_neontrace(t, c, A, B, a, b):
    """The drop: the shot goes dark as its own outlines are drawn on in neon (pink to cyan, a white-hot core), they
    flicker like a sign warming up, and on the drop the outlines become the next shot's, which fills in from them."""
    if t < c:
        fa = layer(A, t); e = _neon_edges(fa)
        u = clamp((t - a) / (c - 0.10 - a)); dark = 1 - 0.68 * ease(u)
        l = luma(fa)[..., None]; base = (fa * 0.5 + l * 0.5) * dark * np.float32([1.06, 0.92, 1.04])
        reveal = ss(0, 1, (lerp(-0.15, 1.15, ease(clamp((t - a) / (c - 0.14 - a)))) * H - YY) / 90)
        fl = 1.0
        for tf in (c - 0.16, c - 0.09):                                                          # warm-up flicker
            if tf <= t < tf + 0.03: fl = 0.55
        return _neon_paint(base, e * reveal, fl, t)
    u = clamp((t - c) / (b - c)); fb = layer(B, t); e = _neon_edges(fb)
    fill = ease(clamp((u - 0.10) / 0.80))
    l = luma(fb)[..., None]; base = (fb * 0.5 + l * 0.5) * 0.2 * (1 - fill) + fb * fill
    f = _neon_paint(base, e, (1 - ease(clamp((u - 0.25) / 0.75))) * 1.2, t)
    return f + 0.30 * math.exp(-(t - c) * 16)


# ---- rack focus through bokeh (0:08)
def _disc_blur(f, r):
    """Lens (disc) blur at quarter resolution, highlights boosted so they bloom into bokeh."""
    if r < 0.6: return f
    q = 4; sm = cv2.resize(f, (W // q, H // q), interpolation=cv2.INTER_AREA)
    hl = np.clip(sm - 0.55, 0, None) * 4.0; sm = sm + hl
    rr = max(1, int(round(r / q))); k = np.zeros((2 * rr + 1, 2 * rr + 1), np.float32)
    cv2.circle(k, (rr, rr), rr, 1.0, -1, cv2.LINE_AA); k /= k.sum()
    bl = cv2.filter2D(sm, -1, k, borderType=cv2.BORDER_REFLECT)
    return cv2.resize(bl, (W, H), interpolation=cv2.INTER_LINEAR)


def t_rackfocus(t, c, A, B, a, b):
    """A cinematic rack focus: the shot falls out of focus, its lights blooming into round bokeh with soft neon
    discs drifting in front, and the next shot pulls back into sharp focus on the beat (with a touch of focus
    breathing)."""
    if t < c: u = clamp((t - a) / (c - a)); r = 70 * ease(u) ** 1.2; src = layer(A, t, 1 + 0.05 * ease(u))
    else: u = clamp((t - c) / (b - c)); r = 70 * (1 - ease(u)) ** 1.3; src = layer(B, t, 1.05 - 0.05 * ease(u))
    f = _disc_blur(src, r)
    amt = clamp(r / 70)
    if amt > 0.02:                                                                               # foreground neon bokeh
        rng = np.random.default_rng(21); bk = np.zeros((H // 2, W // 2, 3), np.float32)
        for i in range(26):
            x = rng.uniform(0, W / 2) + (t - a) * rng.uniform(-30, 30); y = rng.uniform(0, H / 2) + (t - a) * rng.uniform(-40, 10)
            rad = rng.uniform(14, 46) * (0.6 + 0.4 * amt); col = (NEON_PINK, NEON_CYAN, SODIUM)[i % 3] * rng.uniform(0.25, 0.6)
            disc = np.zeros((H // 2, W // 2), np.float32); cv2.circle(disc, (int(x), int(y)), int(rad), 1.0, -1, cv2.LINE_AA)
            edge = disc - cv2.erode(disc, np.ones((3, 3), np.uint8)) * 0.6
            bk += (disc * 0.55 + np.clip(edge, 0, 1) * 0.45)[..., None] * col
        f = f + cv2.resize(cv2.GaussianBlur(bk, (0, 0), 1.2), (W, H)) * amt
    return f


# ---- paper tear (the crash)
def _tear_path(seed=4):
    if ('tear', seed) in _K4: return _K4[('tear', seed)]
    rng = np.random.default_rng(seed); ys = np.arange(H, dtype=np.float32)
    x = np.zeros(H, np.float32)
    for k, (amp, per) in enumerate(((70, 900), (28, 260), (10, 70), (4, 18))):
        ph = rng.uniform(0, 2 * math.pi); x += amp * np.sin(ys / per * 2 * math.pi + ph)
    x += W / 2 + 40 * (ys / H - 0.5)
    fib = cv2.resize(np.random.default_rng(seed + 1).random((H // 3, W // 3)).astype(np.float32), (W, H))
    _K4[('tear', seed)] = (x, fib)
    return _K4[('tear', seed)]


def t_tear(t, c, A, B, a, b):
    """The crash: the shot is a print that rips from top to bottom along a jagged line (white fibrous paper core on
    both torn edges); on the hit the halves peel apart toward the lens, throwing a few scraps, onto the next shot."""
    xp, fib = _tear_path(); fa = layer(A, min(t, c - 1e-3)); fb = layer(B, t, 1 + 0.06 * (1 - eout(clamp((t - c) / (b - c)))))
    ytip = H * eout(clamp((t - a) / (c - a)))                                                    # how far the tear has run
    px = xp[np.clip(YY.astype(np.int32), 0, H - 1)]
    dx = XX - px; torn = YY < ytip
    core = (np.abs(dx) < 4 + 10 * fib) & torn                                                     # exposed white paper fibres
    if t < c:
        gapw = 3 * (YY < ytip)
        f = fa.copy(); m = (np.abs(dx) < gapw)
        f = np.where(m[..., None], fb, f)
        f = np.where((core & ~m)[..., None], np.float32([0.86, 0.90, 0.93]) * (0.8 + 0.2 * fib[..., None]), f)
        return f
    u = clamp((t - c) / (b - c)); e = ease(u) ** 0.8
    left = ss(1.5, -1.5, dx); acc = fb.copy(); am = np.zeros((H, W), np.float32)
    paper = np.where(core[..., None], np.float32([0.86, 0.90, 0.93]) * (0.8 + 0.2 * fib[..., None]), fa)
    for side, m in ((-1, left), (1, 1 - left)):
        lift = 1 - 0.35 * np.exp(-np.abs(dx) / 60)[..., None] * e                                 # the torn edge lifts toward the lens
        M = cv2.getRotationMatrix2D((W / 2 + side * W * 0.5, H / 2), side * 12 * e, 1 + 0.12 * e); M[0, 2] += side * W * 0.80 * e; M[1, 2] += 0.04 * H * e
        pc = cv2.warpAffine(paper * lift * m[..., None], M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        pm = cv2.warpAffine(m, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        lit = 1 + 0.12 * e * (0.5 + 0.5 * side)
        acc = acc * (1 - pm[..., None]) + pc * lit; am = np.maximum(am, pm)
    sh = cv2.GaussianBlur(np.roll(np.roll(am, 24, 0), 16, 1), (0, 0), 22) * 0.55
    f = acc * (1 - (sh * (1 - am))[..., None])
    sc = np.zeros((H, W), np.float32); rng = np.random.default_rng(8)                             # a few paper scraps
    for i in range(12):
        y0 = rng.uniform(0.1, 0.9) * H; x0 = xp[int(y0)]; vx = rng.uniform(-500, 500); vy = rng.uniform(-300, 200)
        px_, py_ = x0 + vx * (t - c), y0 + vy * (t - c) + 1400 * (t - c) ** 2; s_ = rng.uniform(5, 13); ang = rng.uniform(0, 6) + (t - c) * 12
        poly = [(px_ + s_ * math.cos(ang + q * 2.1), py_ + s_ * math.sin(ang + q * 2.1)) for q in range(3)]
        cv2.fillConvexPoly(sc, np.int32(poly), 1.0 - u, cv2.LINE_AA)
    f = f * (1 - sc[..., None]) + sc[..., None] * np.float32([0.86, 0.90, 0.93])
    return f + 0.20 * math.exp(-(t - c) * 18)


# ---- the white tee from the opening credits (0:07)
def _knit():
    if 'knit' in _K4: return _K4['knit']
    rng = np.random.default_rng(12)
    col = 0.5 + 0.5 * np.cos(XX * 1.15) ** 2                                                    # jersey knit columns
    row = 0.5 + 0.5 * np.cos(YY * 1.6 + 0.6 * np.sin(XX * 0.21))
    k = 0.7 * col + 0.3 * row
    fz = cv2.resize(fbm(H // 24, W // 24, 4, 13, 4), (W, H), interpolation=cv2.INTER_CUBIC)
    _K4['knit'] = (k.astype(np.float32), fz.astype(np.float32), cv2.GaussianBlur(rng.random((H, W)).astype(np.float32), (0, 0), 0.7))
    return _K4['knit']


def _cloth(under, t, edge_y, press, rise):
    """Thin white cotton over the frame above edge_y: the face beneath pushes into it (relief from the frame's
    luminance), knit texture, folds dragged toward the hem, a ribbed hem band and its shadow."""
    knit, fz, sp = _knit()
    h = cv2.GaussianBlur(luma(under), (0, 0), 9) * (0.6 + 0.8 * press)
    folds = 0.35 * np.sin(XX / 55 + fz * 6 + YY * 0.004 * rise) * np.exp(-np.clip(edge_y - YY, 0, None) / 500)
    hh = h * 22 + folds * 8 + fz * 6
    gy, gx = np.gradient(hh)
    nx, ny = -gx * 0.5, -gy * 0.5; nz = np.ones_like(nx); n = np.sqrt(nx * nx + ny * ny + nz * nz)
    shade = _shade(nx / n, ny / n, nz / n, 0.95, ks=0.12, pw=8, amb=0.55)
    tex = shade * (0.90 + 0.10 * knit) * (0.97 + 0.06 * (sp - 0.5))
    cloth = tex[..., None] * np.float32([0.96, 0.97, 0.98])
    see = cv2.GaussianBlur(under, (0, 0), 6) * 0.14                                             # thin cotton: a little shows through
    cloth = cloth * 0.90 + see * 0.8
    d = edge_y + 40 * np.sin(XX / W * math.pi) - YY
    hem = (d >= 0) & (d < 46)
    rib = 0.88 + 0.12 * np.cos(XX * 0.9)
    cloth = np.where(hem[..., None], cloth * rib[..., None] * (0.92 + 0.08 * (d / 46))[..., None], cloth)
    m = ss(-1.5, 1.5, d)[..., None]
    shadow = (np.exp(-np.clip(-d, 0, None) / 40) * (d < 0) * 0.55)[..., None]
    return under * (1 - shadow) * (1 - m) + cloth * m


def t_tee(t, c, A, B, a, b):
    """The opening-credits tee: a white cotton T-shirt is pulled down over the frame, the face pressing up through
    the knit; on the beat it is pulled off upward and the next shot is underneath."""
    if t < c:
        p = clamp((t - a) / (c - 0.06 - a)); ey = lerp(-0.08 * H, 1.06 * H, ease(p) ** 0.85)
        under = layer(A, t); return _cloth(under, t, ey, ease(clamp((t - (c - 0.14)) / 0.12)), 1.0)
    q = clamp((t - c) / (b - c)); ey = lerp(1.06 * H, -0.10 * H, ease(q) ** 1.2)
    under = layer(B, t, 1 + 0.05 * (1 - ease(q)))
    return _cloth(under, t, ey, 1 - ease(q), 1 + 2 * q)


# ---- sink drain (0:09)
def _sink(t, cx, cy):
    r, th = _polar(cx, cy)
    if 'sinkn' not in _K4: _K4['sinkn'] = (cv2.resize(fbm(H // 32, W // 32, 4, 44, 4), (W, H), interpolation=cv2.INTER_CUBIC) - 0.5)
    nz = _K4['sinkn']
    basin = 0.88 - 0.22 * np.exp(-r / 360) + 0.08 * np.exp(-((r - 560) / 150) ** 2) - 0.10 * ss(700, 900, r)
    f = basin[..., None] * np.float32([0.95, 0.96, 0.97])
    rf = (r + 1.0) / 120; ph = th + 1.9 * np.log(rf) - 7 * t + 2.2 * nz                         # thin, irregular trails spiralling in
    trail = np.zeros_like(r)
    for k, (n_, w_, amp) in enumerate(((2, 0.035, 1.0), (3, 0.020, 0.7), (5, 0.012, 0.5))):
        d = np.abs(np.mod(n_ * ph / (2 * math.pi) + 0.37 * k, 1.0) - 0.5) / n_
        trail = np.maximum(trail, amp * np.exp(-(d / (w_ * (0.6 + 0.8 * ss(800, 100, r)))) ** 2))
    trail = trail * ss(70, 110, r) * ss(900, 250, r) * (0.7 + 0.6 * nz)
    trail = np.clip(trail, 0, 1)
    dilute = ss(500, 120, r)[..., None]                                                         # thick red near the drain, pinker further out
    col = BLOOD * dilute + np.float32([0.38, 0.36, 0.80]) * (1 - dilute)
    f = f * (1 - trail[..., None] * 0.9) + col * trail[..., None] * 0.9
    wet = np.exp(-((np.mod(th * 2 + np.log(rf) * 3 - 10 * t, 2 * math.pi) - 1) / 0.12) ** 2) * 0.06 * ss(80, 200, r)
    f = f + wet[..., None]
    Rd = 78.0                                                                                   # chrome drain with a slotted grate
    ring = ss(Rd + 22, Rd + 18, r) * ss(Rd - 2, Rd + 2, r)
    chrome = 0.55 + 0.35 * np.cos(th - math.radians(-130)) ** 8 + 0.15 * np.cos(r * 0.4)
    f = f * (1 - ring[..., None]) + (chrome * ring)[..., None] * np.float32([1.0, 1.0, 1.02])
    inner = ss(Rd + 1, Rd - 1, r)
    slots = ((np.mod(th + math.pi, math.pi / 4) < 0.22) & (r > 18) & (r < Rd - 8)) | (r < 12)
    grate = np.where(slots, 0.03, 0.62 + 0.25 * np.cos(th - math.radians(-130)) ** 6)
    f = f * (1 - inner[..., None]) + (grate * inner)[..., None]
    return f, r, th


def t_drain(t, c, A, B, a, b):
    """The opening-credits sink: the frame swirls with blood and is sucked down a chrome drain in a white porcelain
    basin, then the next shot spirals back up out of the drain on the beat."""
    cx, cy = W / 2, 0.48 * H
    if t < c: p = ease(clamp((t - a) / (c - 0.03 - a))); fr = layer(A, t); sgn = 1
    else: p = 1 - ease(clamp((t - c) / (b - 0.03 - c))); fr = layer(B, t); sgn = -1
    sink, r, th = _sink(t, cx, cy)
    g = math.exp(math.log(30.0) * p ** 2.4)                                                   # the picture shrinks into the drain
    tw = sgn * 7.0 * p * np.exp(-r / (0.30 * W))
    rs = r * g; ths = th - tw
    mx = (cx + rs * np.cos(ths)).astype(np.float32); my = (cy + rs * np.sin(ths)).astype(np.float32)
    img = cv2.remap(fr, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    vis = cv2.remap(np.ones((H, W), np.float32), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    rim = math.hypot(W, H) / 2 / g
    edge = ss(rim * 1.0, rim * 0.75, r)
    vis = vis * edge
    img = img * (1 - 0.35 * p) + BLOOD * 0.35 * p                                              # the picture reddens as it goes down
    return sink * (1 - vis[..., None]) + img * vis[..., None]


# ---- cinematic letterbox (config LETTERBOX: bars on the build, opening on the drop, back for the loop seam)
def letterbox(f, t):
    import config as _c
    lb = getattr(_c, 'LETTERBOX', None)
    if not lb: return f
    k = 0.0
    for t0, t1 in lb.get('open', []):
        if t < t0: k = 1.0; break
        if t < t1: k = 1 - ease((t - t0) / (t1 - t0)); break
    for t0, t1 in lb.get('close', []):
        if t >= t0: k = max(k, ease(clamp((t - t0) / (t1 - t0))))
    if k <= 0.001: return f
    hb = lb.get('h', 0.09) * H * k
    m = ((YY < hb) | (YY > H - hb)).astype(np.float32)[..., None]
    return f * (1 - m) + np.float32([0.01, 0.01, 0.012]) * m


# ============================================================ blood kit (Perfect Girl v5)
_LB = np.float32([-0.45, -0.60, 0.66]); _LB /= np.linalg.norm(_LB); _HB = _LB + np.float32([0, 0, 1]); _HB /= np.linalg.norm(_HB)


def _blood(f, m, depth=1.0, refr=36.0):
    """Glossy liquid blood over f wherever mask m (0..1): domed surface from the blurred mask, the picture
    refracted and stained through it, a sharp specular glint and a dark rim."""
    if float(m.max()) < 0.003: return f
    if 'bnz' not in _K4: _K4['bnz'] = _ring_noise(131, s=1) - 0.5
    h = cv2.GaussianBlur(m, (0, 0), 6) * depth + _K4['bnz'] * 0.9 * cv2.GaussianBlur(m, (0, 0), 20)   # liquid surface on big pools
    gy, gx = np.gradient(h * 40)
    n = np.sqrt(gx * gx + gy * gy + 1.0); nx, ny, nz = -gx / n, -gy / n, 1 / n
    diff = np.clip(nx * _LB[0] + ny * _LB[1] + nz * _LB[2], 0, 1)
    spec = np.clip(nx * _HB[0] + ny * _HB[1] + nz * _HB[2], 0, 1) ** 70
    mx = (XX + gx * refr / 40).astype(np.float32); my = (YY + gy * refr / 40).astype(np.float32)
    under = cv2.remap(f, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    col = BLOOD * (0.40 + 0.85 * diff)[..., None] * 0.88 + under * np.float32([0.10, 0.08, 0.55]) * 0.16
    rim = np.clip(m - cv2.GaussianBlur(m, (0, 0), 2.5), 0, 1) * 2.5
    col = col * (1 - 0.5 * rim[..., None]) + (spec * 1.1)[..., None]
    mm = np.clip(m, 0, 1)[..., None]
    return f * (1 - mm) + col * mm


def _splat(cx, cy, e, R0, R1, seed, spines=26, sats=60, dirn=None, spread=math.pi):
    """Impact spatter mask: a central pool, tapering spines and satellite droplets, grown by e (0..1).
    dirn/spread limit the spray to a cone (radians)."""
    q = 2; m = np.zeros((H // q, W // q), np.float32); rng = np.random.default_rng(seed)
    cxq, cyq = cx / q, cy / q
    cv2.ellipse(m, (int(cxq), int(cyq)), (max(1, int(R0 * e / q)), max(1, int(R0 * 0.85 * e / q))), rng.uniform(0, 180), 0, 360, 1.0, -1, cv2.LINE_AA)
    for k in range(spines):
        ang = (rng.uniform(-spread, spread) + (dirn or 0.0)) if dirn is not None else rng.uniform(0, 2 * math.pi)
        L = R0 + rng.uniform(0.3, 1.0) * (R1 - R0); L *= e
        w0 = rng.uniform(0.18, 0.35) * R0
        for j in range(8):
            v = j / 7; px = cxq + math.cos(ang) * L * v / q; py = cyq + math.sin(ang) * L * v / q
            cv2.circle(m, (int(px), int(py)), max(1, int(w0 * (1 - 0.85 * v) / q)), 1.0, -1, cv2.LINE_AA)
        tipr = w0 * rng.uniform(0.25, 0.5); cv2.circle(m, (int(cxq + math.cos(ang) * L / q * 1.06), int(cyq + math.sin(ang) * L / q * 1.06)), max(1, int(tipr / q)), 1.0, -1, cv2.LINE_AA)
    for k in range(sats):
        ang = (rng.uniform(-spread, spread) + (dirn or 0.0)) if dirn is not None else rng.uniform(0, 2 * math.pi)
        d = rng.uniform(0.5, 1.45) * R1 * e; rr = rng.uniform(2, 11) * (0.4 + 0.6 * e)
        x, y = cxq + math.cos(ang) * d / q, cyq + math.sin(ang) * d / q
        cv2.ellipse(m, (int(x), int(y)), (max(1, int(rr * 1.6 / q)), max(1, int(rr / q))), math.degrees(ang), 0, 360, 1.0, -1, cv2.LINE_AA)
    m = cv2.GaussianBlur(m, (0, 0), 0.8)
    return cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR)


# ---- a blood drop hits the lens (the drop)
def t_lensdrop(t, c, A, B, a, b):
    """In slow motion a glossy blood drop falls toward the camera and splats on the lens (pool, spines, satellite
    droplets, the picture refracted through the blood); then the blood runs down the glass onto the next shot."""
    ti = c - 0.09; cx, cy = 0.52 * W, 0.40 * H
    if t < ti:
        p = clamp((t - a) / (ti - a)); e = ein(p) * 0.85 + 0.15 * p
        f = layer(A, t) * (1 - 0.25 * p)
        y = lerp(-0.10 * H, cy, e); r = lerp(16, 110, e ** 1.6)
        m = np.zeros((H, W), np.float32)
        cv2.ellipse(m, (int(cx), int(y)), (int(r), int(r * 1.15)), 0, 0, 360, 1.0, -1, cv2.LINE_AA)       # teardrop: round belly
        cv2.ellipse(m, (int(cx), int(y - r * 0.9)), (int(r * 0.45), int(r * 0.9)), 0, 180, 360, 1.0, -1, cv2.LINE_AA)  # and a pointed top
        tr = np.zeros((H, W), np.float32); cv2.line(tr, (int(cx), int(y - r * 2.2)), (int(cx), int(y - r * 6)), 0.25, max(2, int(r * 0.2)), cv2.LINE_AA)
        m = cv2.GaussianBlur(np.maximum(m, cv2.GaussianBlur(tr, (0, 0), 8)), (0, 0), 1.2)
        return _blood(f, m, depth=2.0, refr=60)
    base = layer(A, t) if t < c else layer(B, t, 1 + 0.06 * (1 - ease(clamp((t - c) / (b - c)))))
    e = eout(clamp((t - ti) / 0.07))
    m = _splat(cx, cy, e, 0.22 * W, 0.62 * W, 11, spines=30, sats=80)
    if t >= c:                                                                                  # it runs down the glass
        u = clamp((t - c) / (b - c)); s = 0.9 * H * ein(u) ** 0.9
        m = cv2.warpAffine(m, np.float32([[1, 0, 0], [0, 1 + 0.6 * u, s]]), (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        kb = int(1 + 120 * u) | 1; m = np.maximum(m * 0.0, cv2.blur(m, (1, kb))) * (1 - ease(clamp((u - 0.4) / 0.6)))
    if c - 0.02 <= t < c: m = np.maximum(m, ss(0.0, 0.02, t - (c - 0.02)) * np.ones_like(m))        # fully red on the cut
    f = _blood(base, m, depth=1.4, refr=50)
    if t >= ti: f = f + 0.35 * math.exp(-(t - ti) * 20)
    return f


# ---- cast-off arc
def t_castoff(t, c, A, B, a, b):
    """A blade swing flings a cast-off arc of elongated droplets across the frame; on the beat the arc rips open
    into the next shot, its edges rimmed with glossy blood."""
    acx, acy, Ra = -0.35 * W, 1.10 * H, 1.25 * W
    r = np.sqrt((XX - acx) ** 2 + (YY - acy) ** 2); d = r - Ra
    th0, th1 = math.radians(-82), math.radians(-8)
    sw = clamp((t - a) / (c - 0.04 - a))
    m = np.zeros((H, W), np.float32); rng = np.random.default_rng(23)
    for k in range(70):                                                                         # droplets thrown off the swinging blade
        v = rng.uniform(0, 1)
        if v > sw: continue
        ang = lerp(th0, th1, v); off = rng.normal(0, 26) + 40 * (1 - v)
        x = acx + (Ra + off) * math.cos(ang); y = acy + (Ra + off) * math.sin(ang)
        tang = ang + math.pi / 2; L = rng.uniform(14, 46) * (0.5 + v); w = rng.uniform(5, 14)
        cv2.ellipse(m, (int(x), int(y)), (int(L), int(w)), math.degrees(tang), 0, 360, 1.0, -1, cv2.LINE_AA)
        cv2.circle(m, (int(x + math.cos(tang) * L * 1.5), int(y + math.sin(tang) * L * 1.5)), max(1, int(w * 0.4)), 1.0, -1, cv2.LINE_AA)
    if t < c:
        f = layer(A, t)
        hx, hy = acx + Ra * math.cos(lerp(th0, th1, sw)), acy + Ra * math.sin(lerp(th0, th1, sw))
        trail = np.exp(-(d / 10) ** 2) * ss(lerp(th0, th1, sw) + 0.02, lerp(th0, th1, sw) - 0.3, np.arctan2(YY - acy, XX - acx)) * 0.35
        f = f + (trail * (sw < 1))[..., None] * np.float32([0.6, 0.7, 0.9])
        return _blood(f, cv2.GaussianBlur(m, (0, 0), 1.0), depth=1.6)
    u = clamp((t - c) / (b - c)); w = 18 + 1.5 * H * ease(u) ** 1.3
    nz = _ring_noise(91, s=1) - 0.5
    open_ = ss(w + 2, w - 2, np.abs(d) + 40 * nz)
    fa = layer(A, c - 1e-3); fb = layer(B, t, 1 + 0.08 * (1 - ease(u)))
    f = fa * (1 - open_[..., None]) + fb * open_[..., None]
    rimm = np.exp(-((np.abs(d) + 40 * nz - w) / 14) ** 2) * (1 - 0.6 * u)
    f = _blood(f, np.clip(np.maximum(rimm, m * (1 - open_)), 0, 1), depth=1.4)
    return f + 0.2 * math.exp(-(t - c) * 18)


# ---- bleeding screen
def t_bleed(t, c, A, B, a, b):
    """Thick glossy blood runs down from the top of the frame in rivulets until it covers it, then the whole sheet
    slides away down the screen and the next shot is underneath."""
    rng = np.random.default_rng(31); n = 15
    xs = np.linspace(0, W, n) + rng.uniform(-30, 30, n); ws = rng.uniform(26, 70, n); sp = rng.uniform(0.8, 1.5, n)
    if t < c:
        p = clamp((t - a) / (c - 0.02 - a))
        m = np.zeros((H, W), np.float32)
        sheet = H * 1.08 * ease(p) ** 2.2
        for i in range(n):
            head = min(H * 1.2, sheet + H * 0.75 * sp[i] * p ** 1.1 * (1 - 0.3 * p))
            w = ws[i] * (0.8 + 1.6 * p ** 2.5)
            xp = xs[i] + 14 * np.sin(YY[:, :1] / 90 + i)
            col = (np.abs(XX - xp) < w * (0.55 + 0.45 * ss(head, head - 200, YY))) & (YY < head)
            m = np.maximum(m, col.astype(np.float32))
            cv2.circle(m, (int(xs[i] + 14 * math.sin(head / 90 + i)), int(head)), int(w * 0.8), 1.0, -1, cv2.LINE_AA)
        edge = sheet + 30 * np.sin(XX / 70) + 18 * np.sin(XX / 23 + 2)
        m = np.maximum(m, (YY < edge).astype(np.float32))
        m = cv2.GaussianBlur(m, (0, 0), 2.0)
        return _blood(layer(A, t), m, depth=1.8)
    u = clamp((t - c) / (b - c)); s = H * 1.15 * ease(u) ** 0.9
    trail = np.zeros(W, np.float32)
    for i in range(n): trail += 160 * sp[i] * np.exp(-((np.arange(W) - xs[i]) / ws[i]) ** 2)
    m = (YY > s - trail[None, :] * (0.4 + u)).astype(np.float32)
    m = cv2.GaussianBlur(m, (0, 0), 2.0)
    return _blood(layer(B, t, 1 + 0.05 * (1 - ease(u))), m, depth=1.8)


# ---- impact spatter on a kick
def t_splatburst(t, c, A, B, a, b):
    """On the kick a spatter burst slams onto the frame (pool, spines, satellites) with a flash; the next shot is
    under it, and the blood drips and fades off."""
    e = eout(clamp((t - a) / 0.06)); cx, cy = 0.64 * W, 0.36 * H
    m = _splat(cx, cy, e, 0.10 * W, 0.55 * W, 47, spines=22, sats=70, dirn=math.radians(200), spread=1.4)
    m = np.maximum(m, _splat(0.30 * W, 0.70 * H, e, 0.05 * W, 0.25 * W, 48, spines=10, sats=24))
    base = layer(A, t) if t < c else layer(B, t, 1 + 0.10 * (1 - eout(clamp((t - c) / 0.2))))
    if t >= c:
        u = clamp((t - c) / (b - c))
        m = cv2.warpAffine(m, np.float32([[1, 0, 0], [0, 1, 180 * u * u]]), (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        m = np.maximum(m, cv2.blur(m, (1, int(1 + 90 * u) | 1)) * 0.8) * (1 - ease(clamp((u - 0.35) / 0.65)))
    f = _blood(base, m, depth=1.3)
    return f + 0.22 * math.exp(-max(0.0, t - c) * 26) * (t >= c - 0.01)


# ---- a slow blood drip down the frame edge (fx 'drip')
def drip_fx(f, i, t, s):
    u = clamp((t - s['start']) / (s['end'] - s['start']))
    m = np.zeros((H, W), np.float32)
    for k, (x, sp, w) in enumerate(((0.09 * W, 1.0, 22), (0.93 * W, 0.7, 16))):
        head = -30 + H * 0.85 * sp * ease(u) ** 1.2
        if head < 2: continue
        xp = x + 6 * np.sin(np.arange(int(head)) / 60 + k)
        for y in range(0, int(head), 6): cv2.circle(m, (int(xp[min(y, len(xp) - 1)]), y), int(w * (0.55 + 0.45 * (1 - y / max(head, 1)))), 1.0, -1, cv2.LINE_AA)
        cv2.circle(m, (int(xp[-1]), int(head)), int(w * 1.1), 1.0, -1, cv2.LINE_AA)
    m = np.maximum(m, (YY < 26 + 8 * np.sin(XX / 50)).astype(np.float32) * ss(0.0, 0.15, u))
    return _blood(f, cv2.GaussianBlur(m, (0, 0), 1.5), depth=1.6)


# ---- THE TROPHY: scalpel nick -> blood bead (a lens onto the next shot) -> dive -> glass slide press
def _scalpel(f, tip, ang, z=1.0, alpha=1.0):
    """A drawn steel scalpel: curved #10 blade (bevel, edge glint) on a ribbed handle, tip at `tip`, pointing along ang."""
    L = 230 * z; ca, sa = math.cos(ang), math.sin(ang); nx, ny = -sa, ca
    P = lambda u, v: (tip[0] - ca * u + nx * v, tip[1] - sa * u + ny * v)
    blade = [P(0, 0), P(40 * z, 26 * z), P(120 * z, 34 * z), P(150 * z, 18 * z), P(150 * z, -8 * z), P(70 * z, -6 * z)]
    m = np.zeros((H, W), np.float32); cv2.fillPoly(m, [np.int32(blade)], 1.0, cv2.LINE_AA)
    hl = np.zeros((H, W), np.float32); cv2.polylines(hl, [np.int32([P(4 * z, 3 * z), P(40 * z, 24 * z), P(118 * z, 31 * z)])], False, 1.0, max(1, int(3 * z)), cv2.LINE_AA)
    bev = np.zeros((H, W), np.float32); cv2.fillPoly(bev, [np.int32([P(10 * z, 5 * z), P(40 * z, 22 * z), P(120 * z, 30 * z), P(120 * z, 20 * z), P(40 * z, 12 * z)])], 1.0, cv2.LINE_AA)
    hd = np.zeros((H, W), np.float32); cv2.fillPoly(hd, [np.int32([P(150 * z, 12 * z), P(L + 150 * z, 14 * z), P(L + 150 * z, -16 * z), P(150 * z, -10 * z)])], 1.0, cv2.LINE_AA)
    rib = 0.5 + 0.5 * np.cos(((XX - tip[0]) * ca + (YY - tip[1]) * sa) / (6 * z))
    sh = cv2.GaussianBlur(np.roll(np.roll(np.maximum(m, hd), int(18 * z), 0), int(10 * z), 1), (0, 0), 10 * z) * 0.5 * alpha
    f = f * (1 - sh[..., None])
    steel = (0.62 + 0.18 * bev)[..., None] * np.float32([1.0, 0.99, 0.97])
    f = f * (1 - (m * alpha)[..., None]) + steel * (m * alpha)[..., None] + (hl * 0.9 * alpha)[..., None]
    hcol = (0.55 + 0.18 * rib)[..., None] * np.float32([0.95, 0.95, 0.97])
    return f * (1 - (hd * alpha)[..., None]) + hcol * (hd * alpha)[..., None]


def _slide_label():
    if 'slabel' in _K4: return _K4['slabel']
    from PIL import Image, ImageDraw
    w, h = int(0.84 * W), int(0.075 * H); img = Image.new('RGBA', (w, h), (236, 238, 240, 225)); d = ImageDraw.Draw(img)
    fm = _font_px(MONO, 34); txt = 'MIAMI METRO PD  ·  CASE 0915-07  ·  No. 13'
    tw = d.textlength(txt, font=fm); d.text(((w - tw) / 2, (h - 38) / 2), txt, font=fm, fill=(140, 40, 40, 255))
    a = np.asarray(img).astype(np.float32) / 255
    _K4['slabel'] = (a[..., 2::-1].copy(), a[..., 3].copy())
    return _K4['slabel']


def t_trophy(t, c, A, B, a, b):
    """Dexter's trophy, in one move: a scalpel nicks the cheek, a single glossy bead of blood wells up with the next
    shot inside it like a lens, the camera dives into the bead until it fills the frame, then a glass microscope
    slide presses down, squeezing the blood into a thin film that clears onto the shot."""
    P = np.array([0.55 * W, 0.37 * H]); R = 0.095 * W
    t_cut0, t_cut1, t_bead, tz = a + 0.10, a + 0.20, a + 0.20, c - 0.30
    if t < c:
        ph = ease(clamp((t - tz) / (c - tz)))
        z = math.exp(math.log(RFULL / R) * ph ** 1.8); push = 1 + 0.06 * ease(clamp((t - a) / (tz - a)))
        z *= push
        ctr = P + (np.array([W / 2, H / 2]) - P) * ph
        M = np.float32([[z, 0, ctr[0] - z * P[0]], [0, z, ctr[1] - z * P[1]]])
        f = cv2.warpAffine(layer(A, t), M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        sc = lambda q: (M[:, :2] @ np.asarray(q, np.float64) + M[:, 2])
        k = clamp((t - t_cut0) / (t_cut1 - t_cut0))                                               # the nick
        if t >= t_cut0:
            nk = np.zeros((H, W), np.float32); p0 = sc(P + [30, -4]); p1 = sc(P + [30 - 70 * k, 6 * k])
            cv2.line(nk, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), 1.0, max(3, int(6 * z)), cv2.LINE_AA)
            f = f * (1 - nk[..., None] * 0.9) + BLOOD * 0.9 * nk[..., None]
        if t < t_cut1 + 0.10:                                                                    # the scalpel comes in, cuts, withdraws
            if t < t_cut0: e = eout(clamp((t - a) / (t_cut0 - a))); off = (1 - e) * 600
            elif t < t_cut1: off = 0.0
            else: off = ein(clamp((t - t_cut1) / 0.10)) * 700
            ang = math.radians(-35)
            tipw = P + [30 - 70 * k, 6 * k] + np.array([math.cos(ang + math.pi), math.sin(ang + math.pi)]) * -off * 0 + np.array([off * 0.8, -off * 0.6])
            f = _scalpel(f, sc(tipw), ang + math.pi, z=z * 1.7)
        if t >= t_bead:                                                                          # the bead wells up: a lens onto the next shot
            g = eout(clamp((t - t_bead) / (tz + 0.05 - t_bead)))
            r = R * (0.15 + 0.85 * g)
            cpos = sc(P + [0, r * 0.12]); Rs = r * z
            fb = layer(B, t)
            tint = fb * np.float32([0.42, 0.36, 1.0]) * 0.90 + BLOOD * 0.18
            f, rr = _live_circle(f, tint, cpos[0], cpos[1], Rs, feather=1.5)
            rim = np.exp(-((rr - Rs) / (0.07 * Rs + 2)) ** 2) * (Rs < RFULL * 0.98)
            f = f * (1 - 0.65 * rim[..., None]) + (rim * 0.65)[..., None] * BLOOD * 0.6
            inside = ss(Rs, Rs * 0.85, rr)
            dark = np.clip((rr / max(Rs, 1)) ** 3, 0, 1) * inside * 0.45                         # darker towards the rim: a dome
            f = f * (1 - dark[..., None])
            sx, sy = cpos[0] - 0.36 * Rs, cpos[1] - 0.42 * Rs
            spec = np.exp(-(((XX - sx) / (0.16 * Rs + 1)) ** 2 + ((YY - sy) / (0.10 * Rs + 1)) ** 2)) * 0.9
            spec += np.exp(-(((XX - cpos[0] - 0.30 * Rs) / (0.05 * Rs + 1)) ** 2 + ((YY - cpos[1] + 0.25 * Rs) / (0.05 * Rs + 1)) ** 2)) * 0.5
            f = f + (spec * inside * (Rs < RFULL * 0.9))[..., None]
        return f
    u = clamp((t - c) / (b - c)); fb = layer(B, t)                                               # the slide presses the bead flat
    yl = lerp(-0.05 * H, 1.06 * H, eout(clamp(u / 0.45)))
    above = ss(yl + 2, yl - 2, YY)[..., None]
    film = lerp(0.35, 0.0, ease(clamp((u - 0.25) / 0.70)))
    tint_full = fb * np.float32([0.42, 0.36, 1.0]) * 0.90 + BLOOD * 0.18
    tint_film = fb * (1 - film) + (fb * np.float32([0.55, 0.50, 1.0])) * film
    f = tint_full * (1 - above) + tint_film * above
    squeeze = np.exp(-((YY - yl) / 26) ** 2)[..., None] * (u < 0.5)                              # blood pushed ahead of the glass edge
    f = f * (1 - squeeze * 0.6) + BLOOD * 0.6 * squeeze
    edge = np.exp(-((YY - yl + 8) / 3) ** 2)[..., None] * (u < 0.5)
    f = f + edge * np.float32([0.70, 0.85, 0.75]) * 0.9
    sheen = np.exp(-(((XX * 0.7 + YY * 0.4) - lerp(-200, W + H, u)) / 90) ** 2)[..., None] * 0.10 * (1 - u)
    f = f + sheen * above
    rgb, al = _slide_label(); lh, lw = al.shape; ly0, lx0 = int(0.07 * H), int((W - lw) / 2)
    la = clamp((yl - (ly0 + lh)) / 60) * (1 - ease(clamp((u - 0.55) / 0.35))) * 0              # label removed (user, v7)
    if la > 0:
        sub = f[ly0:ly0 + lh, lx0:lx0 + lw]; aa = al[..., None] * la
        f[ly0:ly0 + lh, lx0:lx0 + lw] = sub * (1 - aa) + rgb * aa
    return f + 0.20 * math.exp(-(t - c) * 18)


# ---- 35mm film strip
def _strip_print():
    if 'strip' in _K4: return _K4['strip']
    m = _text_mask('KODAK VISION3 5219  ·  MIAMI  ·  D.M.  ·  ', INTER_SEMI, 26)
    _K4['strip'] = np.ascontiguousarray(np.rot90(m))
    return _K4['strip']


def t_filmstrip(t, c, A, B, a, b):
    """The shot shrinks into a frame of 35mm film (warm black base, sprocket holes, orange edge code), the strip
    rattles up one frame to the next shot, and the camera zooms back in."""
    s0 = 0.60; t1, t2 = c - 0.10, c + 0.10
    if t < t1: s = lerp(1.0, s0, ease(clamp((t - a) / (t1 - a)))); sh = 0.0
    elif t < t2: s = s0; sh = ease(clamp((t - t1) / (t2 - t1)))
    else: s = lerp(s0, 1.0, ease(clamp((t - t2) / (b - t2)))); sh = 1.0
    pitch = s * H + 70 * s
    base = np.float32([0.035, 0.045, 0.055]); f = np.zeros((H, W, 3), np.float32) + base
    fa, fb = layer(A, min(t, c - 1e-3)), layer(B, t)
    frames = {-1: fa * 0.35, 0: fa, 1: fb, 2: fb * 0.35}
    for k, img in frames.items():
        yc = H / 2 + (k - sh) * pitch
        if abs(yc - H / 2) > H / 2 + pitch: continue
        M = np.float32([[s, 0, W / 2 - s * W / 2], [0, s, yc - s * H / 2]])
        im = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        mk = cv2.warpAffine(np.ones((H, W), np.float32), M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)[..., None]
        f = f * (1 - mk) + im * mk
    if s < 0.999:
        hx = W / 2 + s * W / 2 + (W / 2 - s * W / 2) * 0.5                                       # sprocket holes in the margins
        hw, hh, gap = max(4, int(0.18 * (W / 2 - s * W / 2))), int(26 * s + 6), int(64 * s + 10)
        off = (-sh * pitch) % gap
        holes = np.zeros((H, W), np.float32)
        for xc in (W - hx, hx):
            y = -gap + off
            while y < H + gap:
                cv2.rectangle(holes, (int(xc - hw / 2), int(y)), (int(xc + hw / 2), int(y + hh)), 1.0, -1, cv2.LINE_AA); y += gap
        f = f * (1 - holes[..., None]) + holes[..., None] * np.float32([0.80, 0.84, 0.86]) * 0.85
        txt = _strip_print(); th, tw = txt.shape; x0 = int(W - (W / 2 - s * W / 2) * 0.22 - tw / 2)
        if 0 <= x0 < W - tw:
            col = np.zeros((H, tw), np.float32); y = int((-sh * pitch) % th) - th
            while y < H: ya, yb = max(0, y), min(H, y + th); col[ya:yb] = np.maximum(col[ya:yb], txt[ya - y:yb - y]); y += th
            f[:, x0:x0 + tw] = f[:, x0:x0 + tw] * (1 - col[..., None] * 0.8) + col[..., None] * np.float32([0.15, 0.45, 0.95]) * 0.8
    if t1 <= t < t2:
        v = math.sin(math.pi * sh); kb = int(1 + 70 * v) | 1; f = cv2.blur(f, (1, kb))
    return f


# ---- long-exposure light trails
def t_lighttrails(t, c, A, B, a, b):
    """Long-exposure light trails: red tail-lights, white headlights and amber streetlight streak across the frame
    in curves; they flare into a bright band on the beat and the next shot is behind them as they pass."""
    rng = np.random.default_rng(77); n = 18
    base = layer(A, t) * (1 - 0.35 * ease(clamp((t - a) / (c - a)))) if t < c else layer(B, t) * lerp(0.65, 1.0, ease(clamp((t - c) / (b - c))))
    q = 2; acc = np.zeros((H // q, W // q, 3), np.float32)
    cols = (np.float32([0.15, 0.15, 1.0]), np.float32([0.85, 0.92, 1.0]), SODIUM, np.float32([0.25, 0.25, 0.95]))
    for i in range(n):
        y0 = rng.uniform(0.05, 0.95) * H; bend = rng.uniform(-0.12, 0.12) * H; d = 1 if i % 2 == 0 else -1
        t0 = a + rng.uniform(0, 0.25); sp = rng.uniform(2.5, 4.5) * W
        head = (t - t0) * sp - 0.2 * W
        if head <= 0: continue
        tail = max(-0.2 * W, head - rng.uniform(0.7, 1.4) * W)
        xs = np.linspace(tail, min(head, 1.2 * W), 40)
        xx = xs if d > 0 else W - xs
        ys = y0 + bend * np.sin(xs / W * math.pi)
        pts = np.int32(np.stack([xx / q, ys / q], 1)); col = cols[i % 4] * rng.uniform(0.6, 1.0)
        layer_ = np.zeros((H // q, W // q), np.float32)
        cv2.polylines(layer_, [pts], False, 1.0, max(1, int(rng.uniform(2, 6) / q)), cv2.LINE_AA)
        acc += layer_[..., None] * col
    glow = cv2.GaussianBlur(acc, (0, 0), 6) * 1.5 + cv2.GaussianBlur(acc, (0, 0), 20) * 1.0
    tr = cv2.resize(acc + glow, (W, H), interpolation=cv2.INTER_LINEAR)
    fade = 1 - ease(clamp((t - c - 0.02) / (b - c - 0.12)))
    f = base + tr * fade
    band = math.exp(-((t - c) / 0.045) ** 2)
    f = f + band * (np.exp(-((YY - 0.5 * H) / (0.35 * H)) ** 2)[..., None] * np.float32([0.65, 0.75, 1.0]) * 0.9)
    return f


# ---- sliding panes
def t_panes(t, c, A, B, a, b, n=3):
    """On the kick the next shot slams in as three vertical panes from alternating directions, staggered."""
    fa = layer(A, t) * (1 - 0.3 * clamp((t - a) / (c - a + 0.15))); fb = layer(B, t); f = fa.copy()
    pw = W / n
    for j in range(n):
        x0, x1 = int(j * pw), int((j + 1) * pw); d = -1 if j % 2 == 0 else 1
        u = eout(clamp((t - (a + 0.04 * j)) / 0.18)); off = int(round(d * (1 - u) * H))
        if u <= 0: continue
        src = fb[:, x0:x1]
        pane = np.roll(src, off, 0)
        if off > 0: pane[:off] = fa[:off, x0:x1]
        elif off < 0: pane[off:] = fa[off:, x0:x1]
        spd = (1 - u)
        if spd > 0.05: pane = cv2.blur(pane, (1, int(1 + 80 * spd) | 1))
        f[:, x0:x1] = pane
        if 0 < u < 1:
            ye = (off if off > 0 else H + off)
            f[max(0, ye - 3):ye + 3, x0:x1] = 0.02
        if 0 < u < 1 and j > 0: f[:, x0 - 3:x0 + 3] = 0.02
    return f + 0.12 * math.exp(-max(0.0, t - c) * 20) * (t >= c)


# ---- Harry's Code: the shot turns like a notebook page
FSERIF_I = '/usr/share/fonts/truetype/freefont/FreeSerifItalic.ttf'
FSERIF_BI = '/usr/share/fonts/truetype/freefont/FreeSerifBoldItalic.ttf'


def _code_page():
    if 'codepage' in _K4: return _K4['codepage']
    from PIL import Image, ImageDraw
    pw, ph = 2 * W, 2 * H
    img = Image.new('RGB', (pw, ph), (244, 236, 214)); d = ImageDraw.Draw(img)
    for y in range(0, ph, 58): d.line([(0, y), (pw, y)], fill=(170, 190, 215), width=2)
    for x in (int(pw * 0.30),): d.line([(x, 0), (x, ph)], fill=(205, 120, 120), width=3)
    ox, oy = int(pw * 0.33), int(ph * 0.30); ink = (28, 40, 92)
    d.text((ox, oy), "Harry's Code", font=_font_px(FSERIF_BI, 64), fill=ink)
    d.line([(ox, oy + 82), (ox + 380, oy + 78)], fill=ink, width=3)
    lines = ('1. Never get caught.', '2. Never kill an innocent.', '3. Be sure. Always be sure.', '4. Blend in.')
    for k, ln in enumerate(lines): d.text((ox + 6, oy + 128 + k * 116), ln, font=_font_px(FSERIF_I, 46), fill=ink)
    _K4['codepage_c'] = (ox + 330, oy + 300)
    a = np.asarray(img).astype(np.float32)[..., ::-1] / 255
    n = cv2.resize(fbm(ph // 32, pw // 32, 4, 9, 4), (pw, ph), interpolation=cv2.INTER_CUBIC)
    a *= (0.90 + 0.12 * n)[..., None]
    _K4['codepage'] = a.astype(np.float32)
    return _K4['codepage']


def t_codepage(t, c, A, B, a, b):
    """Harry's Code: the shot is a page in Harry's notebook. It turns from the bottom-right corner (a shaded crease,
    the flap casting a shadow); on its back, handwritten on ruled paper, is the Code. The next shot is underneath."""
    d = np.array([0.62, 0.78]); d = d / np.linalg.norm(d)
    L = 0.5 * (W * d[0] + H * d[1]) + 40
    v = clamp((t - a) / (b - a)); pm = -0.12 * L                                               # peel, a near-pause to read, flick off
    if v < 0.30: pos = lerp(L, pm + 0.03 * L, eout(v / 0.30))
    elif v < 0.72: pos = lerp(pm + 0.03 * L, pm - 0.03 * L, (v - 0.30) / 0.42)
    else: pos = lerp(pm - 0.03 * L, -1.9 * L, ein((v - 0.72) / 0.28))
    cx, cy = W / 2 + d[0] * pos, H / 2 + d[1] * pos
    s = (XX - cx) * d[0] + (YY - cy) * d[1]
    xr = XX - 2 * s * d[0]; yr = YY - 2 * s * d[1]                                               # where each flap point came from
    inpage = (xr >= 0) & (xr < W) & (yr >= 0) & (yr < H) & (s < 0)
    fa = layer(A, min(t, c - 1e-3)) if t < c + 0.05 else layer(A, c - 1e-3); fb = layer(B, t)
    T = _code_page(); tcx, tcy = _K4['codepage_c']
    key = ('cp_F', round(pm, 1))
    if key not in _K4:                                                                            # where the flap sits mid-turn
        sm = (XX - (W / 2 + d[0] * pm)) * d[0] + (YY - (H / 2 + d[1] * pm)) * d[1]
        xm = XX - 2 * sm * d[0]; ym = YY - 2 * sm * d[1]
        mm = (xm >= 0) & (xm < W) & (ym >= 0) & (ym < H) & (sm < 0)
        _K4[key] = (float(XX[mm].mean()), float(YY[mm].mean())) if mm.any() else (W * 0.3, H * 0.3)
    Fx, Fy = _K4[key]
    qx = (XX - 2 * (pos - pm) * d[0] + (tcx - Fx)).astype(np.float32); qy = (YY - 2 * (pos - pm) * d[1] + (tcy - Fy)).astype(np.float32)
    back = cv2.remap(T, qx, qy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    crease = np.exp(s / 34) * (s < 0)                                                            # the rounded crease catches the light
    back = back * (0.80 + 0.14 * np.clip(-s / 400, 0, 1))[..., None] + (crease * 0.22)[..., None]
    fm = inpage.astype(np.float32)
    fm = cv2.GaussianBlur(fm, (0, 0), 0.8)
    under = np.where((s < 0)[..., None], fa, fb)
    shadow_b = (0.55 * np.exp(-np.clip(s, 0, None) / 60) * (s >= 0))[..., None]                  # the lifted page shades what's under it
    under = under * (1 - shadow_b)
    fsh = np.roll(np.roll(fm, 18, 0), 12, 1); fsh = cv2.GaussianBlur(fsh, (0, 0), 16) * 0.45 * (1 - fm)
    under = under * (1 - fsh[..., None])
    return under * (1 - fm[..., None]) + back * fm[..., None]


# ---- EYE DIVE: push into Dexter's eye, a drawn iris takes over, the pupil opens onto the next shot
def _iris_polar():
    if 'irisp' in _K4: return _K4['irisp']
    rng = np.random.default_rng(5)
    hr, wt = 256, 2048
    fib = cv2.GaussianBlur(rng.random((hr, wt)).astype(np.float32), (0, 0), sigmaX=1.2, sigmaY=10)
    fib = (fib - fib.min()) / (fib.max() - fib.min())
    low = cv2.resize(fbm(16, 128, 4, 6, 4), (wt, hr), interpolation=cv2.INTER_CUBIC)
    crypt = (cv2.GaussianBlur(rng.random((hr, wt)).astype(np.float32), (0, 0), 4) > 0.62).astype(np.float32)
    crypt = cv2.GaussianBlur(crypt, (0, 0), 2)
    _K4['irisp'] = (fib.astype(np.float32), low.astype(np.float32), crypt.astype(np.float32))
    return _K4['irisp']


def _draw_iris(cx, cy, R, pupil, t):
    """A procedural iris at screen (cx, cy) radius R: radial stroma fibres, crypts, a jagged collarette, amber
    centre to blue-grey edge, dark limbal ring, black pupil (radius pupil * R). Returns colour, iris alpha, and
    the polar radius map (in units of R)."""
    fib, low, crypt = _iris_polar()
    dx, dy = XX - cx, YY - cy; r = np.sqrt(dx * dx + dy * dy) + 1e-3; th = np.arctan2(dy, dx); rn = r / R
    mx = ((th + math.pi) / (2 * math.pi) * 2047).astype(np.float32); my = (np.clip(rn, 0, 1) * 255).astype(np.float32)
    F = cv2.remap(fib, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    Lw = cv2.remap(low, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    Cr = cv2.remap(crypt, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    k = np.clip((rn - pupil) / max(1e-3, 1 - pupil), 0, 1)[..., None]
    inner = np.float32([0.16, 0.42, 0.62]); outer = np.float32([0.62, 0.56, 0.42])            # amber centre -> blue-grey (BGR)
    col = inner * (1 - ss(0.0, 0.55, k)) + outer * ss(0.0, 0.55, k)
    col = col * (0.45 + 0.85 * F[..., None]) * (0.85 + 0.3 * Lw[..., None])
    col = col * (1 - 0.55 * Cr[..., None] * (k > 0.15))
    coll = np.exp(-((k[..., 0] - (0.32 + 0.05 * (Lw - 0.5))) / 0.035) ** 2)                     # collarette
    col = col + (coll * 0.22)[..., None] * np.float32([0.55, 0.75, 0.85])
    col = col * (1 - 0.85 * ss(0.84, 1.0, rn))[..., None]                                       # limbal ring
    ruff = np.exp(-((rn - pupil) / 0.025) ** 2)[..., None] * np.float32([0.10, 0.22, 0.35])
    col = col * (rn > pupil)[..., None] + ruff
    dome = (1 - 0.35 * np.clip(rn, 0, 1) ** 2)[..., None]                                       # cornea dome
    col = col * dome
    alpha = ss(1.06, 0.98, rn)
    return col.astype(np.float32), alpha.astype(np.float32), rn


def _eye_track(A, ts):
    """Eye position (layer coords) and iris radius for shot A at times ts, fitted linearly for a steady track."""
    s = SHOTS[A]; pts = []; ws = []
    for t in ts:
        k = int(round((PRE + t - s['start']) * FPS)); lm = _landmarks(A, k)
        if lm is None: continue
        e1, e2, fw = lm; e = e1 if abs(e1[0] - W / 2) < abs(e2[0] - W / 2) else e2
        u = (t - s['start']) / (s['end'] - s['start']); z = max(lerp(s['push'][0], s['push'][1], u), 1.0)
        pts.append(((e[0] - W / 2) * z + W / 2, (e[1] - 0.42 * H) * z + 0.42 * H)); ws.append(fw * z)
    if not pts: return (lambda t: (W / 2, 0.40 * H)), 40.0
    pts = np.array(pts); tt = np.array(ts[:len(pts)])
    if len(pts) >= 2: px = np.polyfit(tt, pts[:, 0], 1); py = np.polyfit(tt, pts[:, 1], 1)
    else: px = [0, pts[0, 0]]; py = [0, pts[0, 1]]
    return (lambda t: (np.polyval(px, t), np.polyval(py, t))), 0.105 * float(np.median(ws))


def t_eyedive(t, c, A, B, a, b):
    """The camera pushes slowly in on Dexter's face, then accelerates into one eye (locked on with the face
    tracker). As the footage runs out of detail a drawn iris takes over -- stroma fibres, crypts, collarette,
    limbal ring, a wet catchlight. The pupil dilates; the next shot is inside it, a faint red glint at its edge
    (the Dark Passenger). Then the camera falls through the pupil into the shot."""
    key = ('eyetrk', A, round(c, 3))
    if key not in _K4: _K4[key] = _eye_track(A, list(np.linspace(a, c - 0.02, 6)))
    epos, ri = _K4[key]
    t1, t2 = c - 0.24, c + 0.30
    zi = 0.30 * W / ri                                                                            # zoom where the iris is 30% of the width
    if t < t1: v = clamp((t - a) / (t1 - a)); z = math.exp(math.log(zi) * (0.12 * v + 0.88 * ein(v) ** 0.85)); Rs = ri * z
    else: v = clamp((t - t1) / (t2 - t1)); Rs = 0.30 * W * math.exp(math.log(9.0) * ease(v) ** 1.6); z = Rs / ri
    ex, ey = epos(min(t, c)) if t < c else epos(c - 1e-3)
    tgt = (W / 2, 0.45 * H); ph = ease(clamp((t - a) / (t1 - a)))
    cx, cy = lerp(ex, tgt[0], ph), lerp(ey, tgt[1], ph)
    if t < c:
        fa = layer(A, t)
        M = np.float32([[z, 0, cx - z * ex], [0, z, cy - z * ey]])
        f = cv2.warpAffine(fa, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    else:
        f = np.zeros((H, W, 3), np.float32) + np.float32([0.05, 0.06, 0.08])
    pupil = lerp(0.30, 0.82, ease(clamp((t - (c - 0.22)) / 0.30)))
    take = ss(0.13 * W, 0.27 * W, Rs)                                                            # the drawn iris takes over
    if take > 0.001:
        icol, ial, rn = _draw_iris(cx, cy, Rs, pupil, t)
        soft = cv2.GaussianBlur(f, (0, 0), 3 + 10 * take) if t < c else f
        f = soft * (1 - 0.45 * take * ss(1.0, 2.6, rn))[..., None]                               # the eye's surroundings fall into shadow
        a_ = (ial * take)[..., None]
        f = f * (1 - a_) + icol * a_
        Rp = pupil * Rs
        fb = layer(B, t, 1 + 0.08 * (1 - ease(clamp((t - c) / (b - c)))))
        dim = lerp(0.50, 1.0, ss(0.45 * RFULL, RFULL, Rp))
        g, rr = _live_circle(f, fb * dim, cx, cy, Rp, feather=2.0)
        pm = (ss(Rp + 2, Rp - 2, rr) * take)[..., None]
        f = f * (1 - pm) + g * pm
        glint = np.exp(-((rr - Rp) / (0.05 * Rp + 2)) ** 2) * ss(0.8 * RFULL, 0.3 * RFULL, Rp) * take
        f = f + (glint * 0.14)[..., None] * BLOOD * 1.6
        cl = ss(0.9 * RFULL, 0.3 * RFULL, Rp) * take                                            # wet catchlight on the cornea
        if cl > 0.01:
            ddx, ddy = XX - cx, YY - cy; rr2 = np.sqrt(ddx * ddx + ddy * ddy) / Rs; aa = np.arctan2(ddy, ddx)
            win = ss(0.12, 0.0, np.abs(rr2 - 0.62)) * ss(0.42, 0.0, np.abs(aa + 2.25)) * 0.45        # a curved window, bent by the cornea
            win = cv2.GaussianBlur(win.astype(np.float32), (0, 0), 0.012 * Rs + 1)
            dot = np.exp(-(((XX - (cx + 0.30 * Rs)) ** 2 + (YY - (cy + 0.28 * Rs)) ** 2) / (0.025 * Rs + 1) ** 2)) * 0.35
            f = f + ((win + dot) * cl * (rn < 1.0))[..., None]
    sp = clamp((t - (c - 0.08)) / 0.08) * (1 - clamp((t - (c + 0.03)) / 0.08))                  # radial motion blur through the pupil
    if sp > 0.02:
        acc = f.copy()
        for j in (1, 2, 3):
            s_ = 1 + 0.035 * sp * j; Mr = np.float32([[s_, 0, cx * (1 - s_)], [0, s_, cy * (1 - s_)]])
            acc += cv2.warpAffine(f, Mr, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        f = acc / 4
    if t >= t2: f = layer(B, t, 1 + 0.08 * (1 - ease(clamp((t - c) / (b - c)))))
    return f


# ---- THE GLOVE: a gloved hand pushes through the screen as if it were stretched latex, then tears it open
def _hand_field(p, spread):
    """Height field of a hand pressing into a membrane from behind (palm + four fingers + thumb), grown by p."""
    q = 2; m = np.zeros((H // q, W // q), np.float32)
    pcx, pcy = 0.50 * W, 0.58 * H; s = (0.9 + 0.2 * p)
    cv2.ellipse(m, (int(pcx / q), int(pcy / q)), (int(215 * s / q), int(255 * s / q)), 0, 0, 360, 1.0, -1, cv2.LINE_AA)
    for k, (ang, L) in enumerate(((-21, 300), (-7, 345), (7, 330), (21, 270))):
        a_ = math.radians(-90 + ang * (1 + 0.6 * spread)); bx = pcx + (k - 1.5) * 112 * s; by = pcy - 200 * s
        ex, ey = bx + math.cos(a_) * L * s, by + math.sin(a_) * L * s
        cv2.line(m, (int(bx / q), int(by / q)), (int(ex / q), int(ey / q)), 1.0, int(100 * s / q), cv2.LINE_AA)
        cv2.circle(m, (int(ex / q), int(ey / q)), int(50 * s / q), 1.0, -1, cv2.LINE_AA)
    a_ = math.radians(-150 - 15 * spread); bx, by = pcx - 170 * s, pcy + 20 * s
    ex, ey = bx + math.cos(a_) * 230 * s, by + math.sin(a_) * 230 * s
    cv2.line(m, (int(bx / q), int(by / q)), (int(ex / q), int(ey / q)), 1.0, int(80 * s / q), cv2.LINE_AA)
    cv2.circle(m, (int(ex / q), int(ey / q)), int(40 * s / q), 1.0, -1, cv2.LINE_AA)
    d = cv2.distanceTransform((m > 0.5).astype(np.uint8), cv2.DIST_L2, 5)                     # rounded 3-D bulge, not a flat plate
    m = np.sqrt(np.clip(d / 115.0, 0, 1)) * (m > 0.5)
    m = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 7)
    m = cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR)
    return m * p, (pcx, pcy)


def t_glove(t, c, A, B, a, b):
    """The screen is a sheet of blue nitrile: a gloved hand presses into it from behind -- fingers and palm bulging
    toward the camera, the picture stretched and refracted, a wet sheen on the latex, stress lines around the
    knuckles -- and on the beat the latex tears open from the palm and the torn flaps snap back onto the next shot."""
    pr = ease(clamp((t - a) / (c - 0.04 - a))) ** 1.2
    spread = ease(clamp((t - (c - 0.22)) / 0.18))
    h, (pcx, pcy) = _hand_field(pr, spread)
    hh = cv2.GaussianBlur(h, (0, 0), 8)
    gy, gx = np.gradient(hh * 90)
    fa = layer(A, min(t, c - 1e-3))
    k = 1.0 + 0.35 * pr
    mx = (XX - gx * 0.9 + (XX - pcx) * (-0.10 * hh)).astype(np.float32)                         # stretched over the knuckles
    my = (YY - gy * 0.9 + (YY - pcy) * (-0.10 * hh)).astype(np.float32)
    f = cv2.remap(fa, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    n = np.sqrt(gx * gx + gy * gy + 1.0); nx, ny, nz = -gx / n, -gy / n, 1 / n
    diff = np.clip(nx * _LB[0] + ny * _LB[1] + nz * _LB[2], 0, 1)
    spec = np.clip(nx * _HB[0] + ny * _HB[1] + nz * _HB[2], 0, 1) ** 40
    latex = np.float32([0.98, 0.72, 0.42])                                                       # blue nitrile (BGR)
    tint = np.clip(0.14 * pr + 0.32 * hh, 0, 0.55)[..., None]
    f = f * (1 - tint) + latex * (0.35 + 0.75 * diff)[..., None] * tint
    f = f * (0.80 + 0.20 * diff)[..., None] + (spec * (0.25 + 0.9 * hh))[..., None]
    gm = np.sqrt(gx * gx + gy * gy)                                                              # stress lines: fine streaks radiating off the bulges
    ang = np.arctan2(YY - pcy, XX - pcx)
    streak = (0.5 + 0.5 * np.sin(ang * 70 + 3 * np.sin(ang * 7))) ** 6 * np.clip(gm / 6, 0, 1) * (1 - hh) * pr
    f = f + (streak * 0.12)[..., None]
    if t < c: return f
    u = clamp((t - c) / (b - c)); e = ease(u) ** 0.75                                            # the tear opens from the palm
    nz_ = _ring_noise(151, s=1) - 0.5
    dr = np.sqrt(((XX - pcx) / 1.0) ** 2 + ((YY - pcy) / 1.15) ** 2) * (1 + 0.35 * nz_) + 26 * np.sin(ang * 5 + 1.3) + 14 * np.sin(ang * 13)
    Rt = lerp(30, 1.7 * H, ease(u) ** 1.5)
    hole = ss(Rt + 6, Rt - 6, dr)[..., None]
    fb = layer(B, t, 1 + 0.10 * (1 - e))
    rim = np.exp(-((dr - Rt - 14) / 26) ** 2)[..., None]
    lip = latex * (0.30 + 0.25 * np.clip((dr - Rt) / 40, 0, 1))[..., None]                   # the torn latex lip, matte, curling back
    out = f * (1 - hole) + fb * hole
    out = out * (1 - rim * 0.85) + lip * rim * 0.85
    sh = np.exp(-np.clip(dr - Rt, 0, None) / 40)[..., None] * 0.35 * (1 - hole)                  # the flaps' shadow on the latex
    return out * (1 - sh) + 0.12 * math.exp(-(t - c) * 20)


# ---- THE CARVE: a carving knife glides through in slow motion (the next shot in its mirror blade), slashes, splits the frame
def _carve_sprite():
    """Carving knife sprite (tip at the right): blade mask, steel shading, bevel/edge highlights, handle RGBA."""
    if 'carve' in _K4: return _K4['carve']
    Ls, Hs = 1400, 260; oy = 130
    yy, xx = np.mgrid[0:Hs, 0:Ls].astype(np.float32)
    blade = np.zeros((Hs, Ls), np.float32)
    pts = [(1390, oy + 4), (1100, oy - 52), (420, oy - 70), (420, oy + 92), (700, oy + 98), (1000, oy + 82), (1250, oy + 52)]
    cv2.fillPoly(blade, [np.int32(pts)], 1.0, cv2.LINE_AA)
    across = np.clip((yy - (oy - 70)) / 170.0, 0, 1)                                          # spine (0) -> edge (1)
    edge_d = cv2.distanceTransform((blade < 0.5).astype(np.uint8) * 0 + (blade > 0.5).astype(np.uint8), cv2.DIST_L2, 5)
    steel = 0.55 + 0.25 * np.cos(across * math.pi * 1.2) + 0.06 * np.sin(xx / 37)
    bevel = ((across > 0.72) & (blade > 0.5)).astype(np.float32)
    steel = steel * (1 - 0.18 * bevel) + 0.22 * np.exp(-((across - 0.72) / 0.02) ** 2) * (blade > 0.5)
    edge = np.exp(-(edge_d / 1.6) ** 2) * (yy > oy) * 0.9                                     # bright honed edge
    handle = np.zeros((Hs, Ls), np.float32); cv2.rectangle(handle, (60, oy - 58), (360, oy + 76), 1.0, -1, cv2.LINE_AA)
    cv2.circle(handle, (60, oy + 9), 67, 1.0, -1, cv2.LINE_AA)
    bolster = np.zeros((Hs, Ls), np.float32); cv2.rectangle(bolster, (355, oy - 72), (425, oy + 94), 1.0, -1, cv2.LINE_AA)
    hcol = np.zeros((Hs, Ls, 3), np.float32) + np.float32([0.035, 0.03, 0.03])
    hcol += (np.exp(-((yy - (oy - 30)) / 14) ** 2) * 0.10)[..., None]
    for rx in (130, 220, 310): cv2.circle(hcol, (rx, oy + 9), 13, (0.25, 0.55, 0.75), -1, cv2.LINE_AA)   # brass rivets
    bcol = (0.50 + 0.35 * np.cos(np.clip((yy - (oy - 72)) / 166, 0, 1) * math.pi))[..., None] * np.float32([1.0, 1.0, 1.02])
    rgb = hcol * handle[..., None]; al = handle.copy()
    rgb = rgb * (1 - bolster[..., None]) + bcol * bolster[..., None]; al = np.maximum(al, bolster)
    _K4['carve'] = dict(blade=blade, steel=steel.astype(np.float32), edge=edge.astype(np.float32), across=across, rgb=rgb.astype(np.float32), al=al, size=(Ls, Hs), piv=(900, oy))
    return _K4['carve']


def t_carve(t, c, A, B, a, b):
    """The carve: in slow motion a large carving knife glides across the frame, its mirror blade reflecting the next
    shot, a glint running along it with an anamorphic flare; it swings and slashes diagonally through the frame
    (a white-hot cut line), and on the beat the two halves slide apart onto the next shot with a flash and a jolt."""
    K = _carve_sprite(); Ls, Hs = K['size']; px, py = K['piv']; sc = 0.95
    ts = c - 0.13; u1 = clamp((t - a) / (ts - a)); u2 = clamp((t - ts) / (c - ts))
    cut_c = np.array([W / 2, H * 0.48]); cut_a = math.radians(38)
    cd = np.array([math.cos(cut_a), math.sin(cut_a)]); cn = np.array([-cd[1], cd[0]])
    if t < ts:                                                                                   # the slow glide
        e = 0.15 * u1 + 0.85 * ease(u1)
        pos = np.array([lerp(-0.12 * W, 0.40 * W, e), lerp(0.29 * H, 0.33 * H, e)]); ang = lerp(-14, -6, e)
    else:                                                                                        # the slash
        e = ein(u2) if t < c else 1.0
        pos = np.array([lerp(0.42 * W, 1.05 * W, e), lerp(0.33 * H, 0.86 * H, e)]); ang = lerp(-6, 40, ease(min(1.0, u2 * 1.3)))
        if t >= c: pos = pos + np.array([0.5 * W, 0.5 * H]) * ease(clamp((t - c) / 0.12))
    th = math.radians(ang); ca, sa = math.cos(th) * sc, math.sin(th) * sc
    M = np.float32([[ca, -sa, 0], [sa, ca, 0]]); M[:, 2] = pos - M[:, :2] @ np.array([px, py])
    wp = lambda im: cv2.warpAffine(im, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    fb = layer(B, t, 1 + 0.10 * (1 - ease(clamp((t - c) / (b - c)))))
    if t < c:
        fa = layer(A, t, 1 + 0.05 * u1); f = fa * (1 - 0.22 * ease(u1))
    else:                                                                                        # the split
        u = clamp((t - c) / (b - c)); e3 = eout(clamp(u / 0.8))
        fa = layer(A, c - 1e-3)
        side = ((XX - cut_c[0]) * cn[0] + (YY - cut_c[1]) * cn[1])
        f = fb.copy(); acc_m = np.zeros((H, W), np.float32)
        for sg in (-1, 1):
            m = ss(-1.5 * sg, 1.5 * sg, side) if sg > 0 else ss(1.5, -1.5, side)
            off = cn * sg * e3 * 1.25 * W + cd * sg * e3 * 0.10 * W
            Mh = cv2.getRotationMatrix2D((float(cut_c[0]), float(cut_c[1])), -sg * 6 * e3, 1.0); Mh[:, 2] += off
            pc = cv2.warpAffine(fa * m[..., None], Mh, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            pm = cv2.warpAffine(m, Mh, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            sh = cv2.GaussianBlur(np.roll(pm, int(20 * sg), 1), (0, 0), 22) * 0.55
            f = f * (1 - (sh * (1 - pm))[..., None]); f = f * (1 - pm[..., None]) + pc; acc_m = np.maximum(acc_m, pm)
        rim = np.exp(-(np.abs(side) / 3) ** 2) * (1 - e3)
        f = f + (rim * 0.0)[..., None]
        jolt = int(14 * math.exp(-(t - c) * 18) * math.sin((t - c) * 90)); f = np.roll(f, jolt, 0)
        f = f + 0.22 * math.exp(-(t - c) * 18)
    if t >= ts:                                                                                  # the white-hot cut line
        k = ein(u2) if t < c else 1.0
        along = (XX - cut_c[0]) * cd[0] + (YY - cut_c[1]) * cd[1]; side0 = (XX - cut_c[0]) * cn[0] + (YY - cut_c[1]) * cn[1]
        reach = lerp(-1.2 * H, 1.2 * H, k)
        cut = np.exp(-(side0 / 3.0) ** 2) * (along < reach) * (1 - (ease(clamp((t - c) / 0.10)) if t >= c else 0))
        glow = np.exp(-(side0 / 22.0) ** 2) * (along < reach) * 0.35 * (1 - (ease(clamp((t - c) / 0.10)) if t >= c else 0))
        f = f + (cut * 1.2 + glow)[..., None] * np.float32([0.85, 0.92, 1.0])
    bm = wp(K['blade'])                                                                          # the knife
    if bm.max() > 0.003:
        hal = wp(K['al']); hrgb = wp(K['rgb'] * K['al'][..., None])
        shd = cv2.GaussianBlur(np.roll(np.roll(np.maximum(bm, hal), 26, 0), 16, 1), (0, 0), 18) * 0.5
        f = f * (1 - shd[..., None])
        refl = cv2.GaussianBlur(cv2.flip(fb, 1), (0, 0), 1.5)
        st = wp(K['steel']); ed = wp(K['edge']); acr = wp(K['across'])
        gl = lerp(-0.2, 1.2, clamp((t - a) / (c - a)))                                          # glint along the blade
        al_ = ((XX - pos[0]) * math.cos(th) + (YY - pos[1]) * math.sin(th)) / (Ls * sc) + 0.36
        band = np.exp(-((al_ - gl) / 0.05) ** 2) * bm
        steel = (refl * 0.62 + st[..., None] * 0.48) * np.float32([1.02, 1.0, 0.98]) + (band * 0.9)[..., None] + (ed * 0.8)[..., None]
        f = f * (1 - bm[..., None]) + steel * bm[..., None]
        f = f * (1 - hal[..., None]) + hrgb
        if band.max() > 0.2:                                                                     # anamorphic flare off the glint
            iy, ix = np.unravel_index(int(np.argmax(band)), band.shape)
            fl = np.exp(-((YY - iy) / 5) ** 2) * np.exp(-np.abs(XX - ix) / 320) * 0.7 + np.exp(-(((XX - ix) ** 2 + (YY - iy) ** 2) / 60 ** 2)) * 0.6
            f = f + fl[..., None] * np.float32([1.0, 0.85, 0.70])
    if ts <= t < c + 0.02 and u2 > 0.2:                                                          # motion blur on the slash
        f = 0.5 * f + 0.5 * cv2.blur(f, (9, 9))
    return f


# ============================================================ Flash kit VII: "Are We Still Friends?" (Tyler, the Creator)
ORANGE_GLOW = np.float32([0.20, 0.55, 1.0])


def _sparks(t, t0, cx, cy, n, seed, vmin=900, vmax=2600, life=0.45, col=GOLD_GLOW, g=900):
    """Sparks flung out from (cx, cy) at t0, as short glowing streaks."""
    dt = t - t0
    if not 0 <= dt < life: return None
    rng = np.random.default_rng(seed); q = 2; m = np.zeros((H // q, W // q), np.float32)
    for _ in range(n):
        ang = rng.uniform(0, 2 * math.pi); v = rng.uniform(vmin, vmax)
        x, y = cx + math.cos(ang) * v * dt, cy + math.sin(ang) * v * dt + 0.5 * g * dt * dt
        x2, y2 = x - math.cos(ang) * v * 0.025, y - math.sin(ang) * v * 0.025
        cv2.line(m, (int(x / q), int(y / q)), (int(x2 / q), int(y2 / q)), 1.0 - dt / life, 1, cv2.LINE_AA)
    m = cv2.resize(m, (W, H)); return (m[..., None] * 1.4 + cv2.GaussianBlur(m, (0, 0), 5)[..., None] * 1.2) * col


# ---- the particle accelerator (the drop)
def t_accel(t, c, A, B, a, b):
    """On the near-silence the frame drains to cold blue while a tiny core of light grows at the centre, crackling
    and flaring; on the drop the dark-matter shockwave blasts out -- a white-hot ring with blue and orange fringes and
    lightning along its front -- with the next shot inside it, a camera jolt and sparks."""
    cx, cy = W / 2, 0.44 * H; r, th = _polar(cx, cy)
    if t < c:
        u = clamp((t - a) / (c - a)); k = ease(u)
        fa = layer(A, t); l = luma(fa)[..., None]
        f = fa * (1 - 0.8 * k) + l * np.float32([1.25, 0.85, 0.55]) * 0.55 * 0.8 * k
        f = f * (1 - 0.35 * k * np.clip(r / (0.8 * H), 0, 1) ** 1.5)[..., None]
        rc = 3 + 34 * ein(u) * (1 + 0.15 * math.sin(t * 60))
        f = f + (np.exp(-(r / rc) ** 2) * 1.6 * (0.3 + u))[..., None] + (np.exp(-(r / (rc * 5)) ** 2) * 0.9 * u)[..., None] * BLUE_GLOW
        f = f + (np.exp(-((YY - cy) / 3.5) ** 2) * np.exp(-np.abs(XX - cx) / (120 + 700 * u)) * 0.8 * u)[..., None] * BLUE_GLOW
        rng = np.random.default_rng(int(t * FPS)); segs = []
        for _ in range(int(2 + 7 * u)):
            ang = rng.uniform(0, 2 * math.pi); L = rc * rng.uniform(2.5, 7)
            segs.append((bolt_path((cx, cy), (cx + L * math.cos(ang), cy + L * math.sin(ang)), rng, 5, 0.18), 0.6))
        return add(f, draw_bolts(segs, 2.4, glow=BLUE_GLOW, core=np.float32([1.0, 0.95, 0.9]), amp=0.6 + 0.6 * u))
    u = clamp((t - c) / (b - c)); Rw = 20 + RFULL * 1.2 * eout(clamp(u / 0.75))
    fa = layer(A, c - 1e-3); l = luma(fa)[..., None]; fa = l * np.float32([1.25, 0.85, 0.55]) * 0.45
    disp = 60 * np.exp(-((r - Rw) / 70) ** 2) * (1 - u)
    mx = (XX - (XX - cx) / (r + 1e-3) * disp).astype(np.float32); my = (YY - (YY - cy) / (r + 1e-3) * disp).astype(np.float32)
    fb = cv2.remap(layer(B, t, 1 + 0.12 * (1 - eout(u))), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    inside = ss(Rw + 3, Rw - 3, r)[..., None]
    f = fa * (1 - inside) + fb * inside
    fade = 1 - ease(clamp((u - 0.35) / 0.65))
    ring = (np.exp(-((r - Rw) / 6) ** 2) * 1.5)[..., None] + (np.exp(-((r - Rw) / 34) ** 2) * 0.9)[..., None] * BLUE_GLOW \
        + (np.exp(-((r - Rw - 16) / 12) ** 2) * 0.6)[..., None] * ORANGE_GLOW
    f = f + ring * fade
    rng = np.random.default_rng(int(t * FPS) + 7); segs = []
    for _ in range(12):
        ang = rng.uniform(0, 2 * math.pi); p0 = (cx + Rw * math.cos(ang), cy + Rw * math.sin(ang))
        L = rng.uniform(40, 140); a2 = ang + rng.uniform(-0.6, 0.6)
        segs.append((bolt_path(p0, (p0[0] + L * math.cos(a2), p0[1] + L * math.sin(a2)), rng, 5, 0.2), 0.7))
    f = add(f, draw_bolts(segs, 2.6, glow=BLUE_GLOW, amp=1.2 * fade))
    f = add(f, _sparks(t, c, cx, cy, 70, 3))
    jolt = int(18 * math.exp(-(t - c) * 12) * math.sin((t - c) * 75))
    return np.roll(f, jolt, 0) + 0.55 * math.exp(-(t - c) * 18)


# ---- Barry's speed tornado
def t_tornado(t, c, A, B, a, b):
    """The frame twists into Barry's speed tornado -- the picture swirling inward, wind streaks, gold lightning ribbons
    orbiting the eye -- then unwinds into the next shot."""
    cx, cy = W / 2, 0.45 * H; r, th = _polar(cx, cy)
    if t < c: k = ease(clamp((t - a) / (c - a)))
    else: k = 1 - ease(clamp((t - c) / (b - c)))
    mixb = ease(clamp((t - (c - 0.12)) / 0.24))                                                  # both shots spin together through the eye
    src = layer(A, min(t, c - 1e-3)) * (1 - mixb) + layer(B, t) * mixb
    acc = None
    for j, sp in enumerate((0.92, 1.0, 1.08)):                                                   # rotational motion blur
        tw = 4.2 * k * sp * np.exp(-r / (0.45 * W))
        rs = r * (1 + 0.20 * k); ths = th - tw
        mx = (cx + rs * np.cos(ths)).astype(np.float32); my = (cy + rs * np.sin(ths)).astype(np.float32)
        im = cv2.remap(src, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        acc = im if acc is None else acc + im
    f = acc / 3 * (1 - 0.25 * k)
    streak = (0.5 + 0.5 * np.cos(th * 7 - 4.0 * np.log(r / 90 + 1) + t * 38)) ** 28 * k * ss(60, 320, r)
    f = f + (streak * 0.35)[..., None] * np.float32([0.95, 0.95, 1.0])
    segs = []; rng = np.random.default_rng(int(t * FPS))
    for j in range(3):
        a0 = j * 2.1 + t * 13; ss_ = np.linspace(0, 1, 40)
        ang = a0 + ss_ * 2.6; rad = 110 + 620 * ss_
        pts = np.stack([cx + rad * np.cos(ang), cy + rad * np.sin(ang) * 1.15], 1) + rng.normal(0, 6, (40, 2))
        segs.append((pts, 0.9))
    f = add(f, draw_bolts(segs, 4.2, amp=1.5 * k))
    f = f + (np.exp(-(r / 90) ** 2) * 0.9 * k)[..., None] * np.float32([0.55, 0.85, 1.0])            # a hot eye at the centre
    return f + 0.25 * math.exp(-abs(t - c) * 25)


# ---- Flash Time
def t_flashtime(t, c, A, B, a, b):
    """Flash Time: the outgoing shot slows to a stop, the world goes cold blue and dust hangs frozen in the air; Barry's
    gold lightning trail streaks across the frame, the next shot behind it; then time starts again and the motes fall."""
    u = clamp((t - a) / (c - a)); tt = a + (c - a) * 0.5 * (1 - (1 - u) ** 2) if t < c else c
    k = ease(u) if t < c else 1 - ease(clamp((t - c) / (b - c)))
    fa = layer(A, min(tt, c - 1e-3)); fb = layer(B, t)
    ts, te = c - 0.30, c + 0.06; hx = lerp(-0.25 * W, 1.25 * W, ease(clamp((t - ts) / (te - ts))))
    path_y = lambda x: 0.56 * H + 0.10 * H * np.sin(x / W * math.pi * 1.3 + 0.4)
    behind = ss(hx - 10, hx - 60, XX)[..., None] if t < te else np.ones((H, W, 1), np.float32)
    f = fa * (1 - behind) + fb * behind
    l = luma(f)[..., None]; cool = l * np.float32([1.30, 1.0, 0.70])
    f = f * (1 - 0.85 * k) + (cool * 0.80 + f * 0.20) * 0.85 * k
    rng = np.random.default_rng(12); mm = np.zeros((H // 2, W // 2), np.float32)
    fall = max(0.0, t - c) ** 2 * 1800
    for _ in range(140):
        x, y, s_ = rng.uniform(0, W), rng.uniform(0, H), rng.uniform(1, 2.5)
        cv2.ellipse(mm, (int(x / 2), int((y + fall * s_ / 2) / 2)), (max(1, int(s_ / 2)), max(1, int(s_ * 1.6 / 2))), 0, 0, 360, 1.0, -1, cv2.LINE_AA)
    mm = cv2.resize(mm, (W, H)); motes = (mm * 1.0 + cv2.GaussianBlur(mm, (0, 0), 4) * 1.6)[..., None] * np.float32([1.0, 0.92, 0.85])
    f = f + motes * k
    if ts <= t < c + 0.25:                                                                       # the gold streak
        xs = np.linspace(-0.25 * W, min(hx, 1.25 * W), 60); ys = path_y(xs)
        jit = np.random.default_rng(int(t * FPS)).normal(0, 7, (60, 2))
        pts = np.stack([xs, ys], 1) + jit
        tail = 1 - ease(clamp((t - te) / 0.20))
        f = add(f, draw_bolts([(pts, 1.0)], 9.0, amp=1.9 * tail))
        f = add(f, draw_bolts([(pts + np.array([0, 14]), 0.6)], 4.0, glow=np.float32([0.15, 0.25, 1.0]), amp=1.0 * tail))   # red under-streak
        hy = float(path_y(np.array([hx]))[0])
        f = f + (np.exp(-(((XX - hx) / 70) ** 2 + ((YY - hy) / 40) ** 2)) * 1.4 * tail * (t < te))[..., None] * np.float32([0.55, 0.85, 1.0])
    return f


# ---- the Flash ring
def _emblem_layer(cx, cy, R, rot):
    z0 = draw_emblem(np.zeros((H, W, 3), np.float32), cx, cy, R, rot, 1.0, 1.0, 1.0)
    z1 = draw_emblem(np.ones((H, W, 3), np.float32), cx, cy, R, rot, 1.0, 1.0, 1.0)
    al = np.clip(1 - (z1 - z0).mean(2), 0, 1)
    return z0, al


def t_ringpop(t, c, A, B, a, b):
    """Barry's ring: a gold ring with the emblem on its face pops in over the shot (glint across the metal), the
    emblem lid hinges open with a burst of light, and the camera dives into the ring onto the next shot."""
    cx, cy = W / 2, 0.46 * H; R0 = 0.24 * W
    pop = clamp((t - a) / 0.16); s = (1 + 2.2 * (pop - 1) ** 3 + 1.2 * (pop - 1) ** 2) if pop < 1 else 1.0
    to = c - 0.16; hinge = ease(clamp((t - to) / 0.18))
    dive = ein(clamp((t - c) / (b - c - 0.04))) if t >= c else 0.0
    R = R0 * max(0.01, s) * math.exp(math.log(RFULL / (R0 * 0.82)) * dive)
    fa = layer(A, min(t, c - 1e-3)) * (1 - 0.35 * pop) if t < c + 0.30 else None
    f = fa if fa is not None else layer(B, t)
    r, th = _polar(cx, cy)
    band = ss(R * 1.24, R * 1.20, r) * ss(R * 0.80, R * 0.84, r)                                 # the gold band
    tilt = np.clip((r - R * 1.02) / (R * 0.22), -1, 1) * 0.85
    gold = _shade(np.cos(th) * tilt, np.sin(th) * tilt, _nz(tilt), 0.55, ks=1.2, pw=45)[..., None] * np.float32([0.25, 0.62, 0.95])
    sh = cv2.GaussianBlur(np.roll(np.roll(band, 24, 0), 14, 1), (0, 0), 18) * 0.5 * (1 - dive)
    f = f * (1 - sh[..., None])
    if hinge > 0 or t >= c:                                                                      # the opened ring: light and the next shot inside
        fb = layer(B, t, 1 + 0.1 * (1 - dive)); g, rr = _live_circle(f, fb, cx, cy, R * 0.82, feather=2)
        inner = ss(R * 0.84, R * 0.80, r)[..., None] * max(hinge, 1.0 if t >= c else 0)
        light = np.exp(-(r / (R * 0.5)) ** 2)[..., None] * np.float32([0.55, 0.85, 1.0]) * 1.4 * hinge * (1 - dive)
        f = f * (1 - inner) + (g + light) * inner
    f = f * (1 - band[..., None]) + gold * band[..., None]
    if t < c + 0.02 and hinge < 1:                                                               # the emblem lid, hinging up from its top
        e0, al = _emblem_layer(cx, cy, R * 0.82, 0.0)
        sy = math.cos(hinge * math.radians(88)); y0 = cy - R * 0.82
        M = np.float32([[1, 0, 0], [0, sy, y0 * (1 - sy)]])
        e0 = cv2.warpAffine(e0, M, (W, H)); al = cv2.warpAffine(al, M, (W, H))
        f = f * (1 - al[..., None]) + e0
    gl = clamp((t - (a + 0.12)) / 0.25)
    if 0 < gl < 1: f = f + (np.exp(-((((XX - cx) + (YY - cy) * 0.6) / R - lerp(-1.6, 1.6, gl)) / 0.12) ** 2) * band * 0.6)[..., None]
    if t >= to:                                                                                  # rays and sparks as it opens
        ray = (0.5 + 0.5 * np.cos(th * 18 + t * 4)) ** 12 * np.exp(-r / (R * 2.2)) * hinge * (1 - dive)
        f = f + (ray * 0.5)[..., None] * np.float32([0.45, 0.8, 1.0])
        f = add(f, _sparks(t, to + 0.08, cx, cy - R * 0.4, 40, 9, vmin=500, vmax=1600))
    return f + 0.35 * math.exp(-max(0.0, t - c) * 20) * (t >= c)


# ---- breach (loop)
def t_breach(t, A, a, end):
    """Loop: a swirling blue multiverse breach tears open over the last shot -- energy spiralling round its rim,
    lightning crackling off it, the picture pulled toward it -- and through it is the start of the edit; the camera
    goes through."""
    if t >= end: return layer(0, t - DUR)
    cx, cy = W / 2, 0.45 * H; r, th = _polar(cx, cy)
    op = eout(clamp((t - a) / 0.36)); dv = ein(clamp((t - (a + 0.42)) / (end - a - 0.42)))
    Rp = 6 + 0.30 * W * op; Rp = Rp * math.exp(math.log(RFULL * 1.05 / (0.30 * W + 6)) * dv)
    fa = layer(A, t)
    pull = 90 * np.exp(-np.clip(r - Rp, 0, None) / 260) * op * (1 - dv)
    tw = 0.5 * np.exp(-np.clip(r - Rp, 0, None) / 300) * op
    rs = r + pull; ths = th + tw
    fa = cv2.remap(fa, (cx + rs * np.cos(ths)).astype(np.float32), (cy + rs * np.sin(ths)).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    f, rr = _live_circle(fa, layer(0, t - DUR), cx, cy, Rp * 0.9, feather=3)
    if 'brn' not in _K4: _K4['brn'] = cv2.resize(fbm(64, 512, 4, 33, 4), (2048, 256), interpolation=cv2.INTER_CUBIC)
    rn = (r - Rp * 0.8) / (Rp * 0.5 + 1)
    mx = ((np.mod(th + 2.5 * np.log(r / (Rp + 1) + 1) - t * 9, 2 * math.pi)) / (2 * math.pi) * 2047).astype(np.float32)
    my = (np.clip(rn, 0, 1) * 255).astype(np.float32)
    sw = cv2.remap(_K4['brn'], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    rim = np.exp(-((r - Rp) / (0.12 * Rp + 4)) ** 2) * (0.5 + 0.9 * sw) * (1 - dv * 0.8)
    f = f + (rim * 1.1)[..., None] * BLUE_GLOW + (np.exp(-((r - Rp) / (0.03 * Rp + 2)) ** 2) * 0.8 * (1 - dv))[..., None]
    rng = np.random.default_rng(int(t * FPS)); segs = []
    for _ in range(9):
        ang = rng.uniform(0, 2 * math.pi); p0 = (cx + Rp * math.cos(ang), cy + Rp * math.sin(ang)); L = rng.uniform(40, 160)
        segs.append((bolt_path(p0, (p0[0] + L * math.cos(ang + 0.5), p0[1] + L * math.sin(ang + 0.5)), rng, 5, 0.2), 0.6))
    f = add(f, draw_bolts(segs, 2.2, glow=BLUE_GLOW, amp=0.9 * op * (1 - dv)))
    return f


# ============================================================ Flash kit VIII (Still Friends v3): mirage, trailbolt, disperse, remnants, struck
RED_T = np.float32([0.25, 0.30, 1.0]); GOLD_T = np.float32([0.35, 0.80, 1.0]); BLUE_T = np.float32([1.0, 0.70, 0.35])


def _hspeed(f, k):
    """Horizontal motion blur of strength k (0..1)."""
    kb = int(1 + 120 * k) | 1
    return f if kb < 3 else cv2.blur(f, (kb, 1))


# ---- speed mirage (the drop)
def t_mirage(t, c, A, B, a, b):
    """The drop: through the hush Barry starts to vibrate -- faint red and gold afterimages jitter off him, wider and
    wider; on the drop the afterimages burst out sideways in a fan of speed-blurred copies, and the next shot is left
    standing in the middle as they streak off the frame."""
    if t < c:
        u = clamp((t - a) / (c - a)); fa = layer(A, t)
        f = fa.copy()
        for k, col in enumerate((RED_T, GOLD_T, RED_T)):
            amp = (6 + 70 * u ** 2.2) * (k + 1) / 3
            dx = int(amp * math.sin(t * 95 + k * 2.1)); dy = int(0.25 * amp * math.cos(t * 71 + k))
            g = np.roll(np.roll(fa, dx, 1), dy, 0) * col
            f = np.maximum(f, g * (0.25 + 0.55 * u))
        f = f + (np.exp(-((XX - W / 2) / (W * 0.6)) ** 2) * 0.08 * u)[..., None] * GOLD_T
        return f + 0.35 * ein(clamp((t - (c - 0.05)) / 0.05))
    v = clamp((t - c) / (b - c)); e = eout(v)
    fb = layer(B, t, 1.12 - 0.12 * e)
    fa = layer(A, c - 1e-3)
    wb = ease(clamp((t - c - 0.04) / 0.26))
    f = fb * (0.30 + 0.70 * wb) + fa * 0.25 * (1 - wb) * np.float32([0.6, 0.6, 0.9])
    for k in range(1, 7):                                                                        # the fan of afterimages, screened on
        sg = -1 if k % 2 else 1; dist = sg * (0.10 + 0.16 * ((k + 1) // 2)) * W * (0.3 + 1.7 * e)
        col = (RED_T, GOLD_T)[k % 2]
        g = np.clip(_hspeed(np.roll(fa, int(dist), 1), 0.20 + 0.35 * (1 - v)) * col, 0, 1)
        al = (0.60 - 0.07 * k) * (1 - v) ** 1.1
        f = 1 - (1 - np.clip(f, 0, 1)) * (1 - g * al)
    lines = np.zeros((H // 4, W // 4), np.float32); rng = np.random.default_rng(int(t * FPS))   # speed lines
    for _ in range(40):
        y = rng.integers(0, H // 4); x0 = rng.integers(0, W // 4); L = rng.integers(20, 120)
        cv2.line(lines, (int(x0), int(y)), (int(x0 + L), int(y)), float(rng.uniform(0.3, 1.0)), 1, cv2.LINE_AA)
    lines = cv2.resize(lines, (W, H)) * (1 - v) ** 1.5
    f = f + lines[..., None] * GOLD_T * 0.5
    return f + 0.25 * math.exp(-(t - c) * 22)


# ---- the trail draws the bolt
_BOLT_N = np.array([[0.66, 0.10], [0.30, 0.54], [0.49, 0.54], [0.34, 0.90], [0.72, 0.42], [0.53, 0.42], [0.70, 0.10]])


def t_trailbolt(t, c, A, B, a, b):
    """Barry's gold lightning trail races round the frame and draws the Flash's bolt; when the shape closes it flares
    and fills with the next shot, then the bolt swells past the edges of the frame onto it."""
    pts = _BOLT_N * np.array([W, H]); poly = np.vstack([pts, pts[:1]])
    seg = np.hypot(*np.diff(poly, axis=0).T); cum = np.concatenate([[0], np.cumsum(seg)]); Ltot = cum[-1]
    tc = c - 0.06; p = ease(clamp((t - a) / (tc - a)))
    fa = layer(A, min(t, c - 1e-3)) * (1 - 0.30 * p)
    if t < tc:
        L = p * Ltot; n = int(np.searchsorted(cum, L)); path = list(poly[:n])
        if n < len(poly):
            k = (L - cum[n - 1]) / max(seg[n - 1], 1e-3); path.append(poly[n - 1] + (poly[n] - poly[n - 1]) * k)
        path = np.array(path, np.float64)
        dense = np.vstack([np.linspace(path[i], path[i + 1], 12, endpoint=False) for i in range(len(path) - 1)] + [path[-1:]]) if len(path) > 1 else path
        jit = np.random.default_rng(int(t * FPS)).normal(0, 3.5, dense.shape)
        f = add(fa, draw_bolts([(dense + jit, 1.0)], 7.0, amp=1.5))
        hx, hy = dense[-1]
        f = f + (np.exp(-(((XX - hx) ** 2 + (YY - hy) ** 2) / 60 ** 2)) * 1.6)[..., None] * np.float32([0.6, 0.9, 1.0])
        return f
    u = clamp((t - tc) / (b - tc)); cen = pts.mean(0); s = 1.0 + 9.0 * ein(u) ** 1.2
    P = (cen + (pts - cen) * s).astype(np.int32)
    m = np.zeros((H, W), np.float32); cv2.fillPoly(m, [P], 1.0, cv2.LINE_AA)
    fb = layer(B, t, 1 + 0.12 * (1 - eout(u)))
    f = fa * (1 - m[..., None]) + fb * m[..., None]
    rim = np.zeros((H, W), np.float32); cv2.polylines(rim, [P], True, 1.0, 6, cv2.LINE_AA)
    f = f + (cv2.GaussianBlur(rim, (0, 0), 6) * 1.8 + rim)[..., None] * GOLD_GLOW * (1 - u)
    return f + 0.45 * math.exp(-(t - tc) * 18)


# ---- speed disintegration
def _particles(img, q, keep):
    sm = cv2.resize(img, (W // q, H // q), interpolation=cv2.INTER_AREA)
    gy, gx = np.mgrid[0:H // q, 0:W // q]; return gx.ravel() * q + q / 2, gy.ravel() * q + q / 2, sm.reshape(-1, 3)


def t_disperse(t, c, A, B, a, b):
    """At super speed Barry takes the picture with him: the shot breaks into thousands of gold-edged particles that
    stream off to the right in waves; the next shot streams in from the left and assembles particle by particle."""
    q = 6; rng = np.random.default_rng(3); h_, w_ = H // q, W // q
    if 'dsp_noise' not in _K4: _K4['dsp_noise'] = cv2.resize(fbm(h_ // 4 + 2, w_ // 4 + 2, 3, 41, 4), (w_, h_)).astype(np.float32)
    nz = _K4['dsp_noise']; gx = (np.arange(w_)[None, :] / w_).repeat(h_, 0)
    delay = (0.55 * gx + 0.30 * nz).ravel(); jitter = (rng.random(h_ * w_) - 0.5)
    half = np.zeros((H // 2, W // 2, 3), np.float32) + np.float32([0.05, 0.03, 0.02])
    def splat(buf, X, Y, C):
        xi = (X / 2).astype(np.int32); yi = (Y / 2).astype(np.int32); ok = (xi >= 0) & (xi < W // 2 - 2) & (yi >= 0) & (yi < H // 2 - 2)
        xi, yi, C = xi[ok], yi[ok], C[ok]
        for dy in (0, 1, 2):
            for dx in (0, 1, 2): buf[yi + dy, xi + dx] = C
    pb = clamp((t - (c - 0.30)) / (b - (c - 0.30)))                                                # B assembles from the left
    fb = layer(B, t); Xb, Yb, Cb = _particles(fb, q, None)
    lb = np.clip((pb * 1.25 - (0.85 - delay)) / 0.40, 0, 1)
    done_b = cv2.resize(lb.reshape(h_, w_), (W, H), interpolation=cv2.INTER_NEAREST) >= 1.0
    mv = (lb > 0) & (lb < 1)
    off = (1 - lb[mv]) ** 2 * W * 0.9
    splat(half, Xb[mv] - off, Yb[mv] + jitter[mv] * 80 * (1 - lb[mv]), Cb[mv] * (1 - 0.5 * (1 - lb[mv]))[:, None] + GOLD_T * 0.5 * (1 - lb[mv])[:, None])
    pa = clamp((t - a) / (c + 0.16 - a))                                                           # A streams off to the right
    fa = layer(A, min(t, c - 1e-3)); Xa, Ya, Ca = _particles(fa, q, None)
    la = np.clip((pa * 1.25 - delay) / 0.40, 0, 1)
    intact_a = cv2.resize(la.reshape(h_, w_), (W, H), interpolation=cv2.INTER_NEAREST) <= 0.0
    mv = (la > 0) & (la < 1)
    off = la[mv] ** 2 * W * 0.9
    splat(half, Xa[mv] + off, Ya[mv] + jitter[mv] * 80 * la[mv], Ca[mv] * (1 - 0.5 * la[mv])[:, None] + GOLD_T * 0.6 * la[mv][:, None])
    f = cv2.resize(half, (W, H), interpolation=cv2.INTER_LINEAR)
    f = np.where(done_b[..., None], fb, f)
    f = np.where(intact_a[..., None], fa, f)
    edge = cv2.GaussianBlur((~intact_a).astype(np.float32) - cv2.erode((~intact_a).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(np.float32), (0, 0), 4)
    return f + (np.clip(edge, 0, 1) * 1.4)[..., None] * GOLD_GLOW * (0 < pa < 1)


# ---- time remnants
def t_remnants(t, c, A, B, a, b):
    """Time remnants: just before the hit, ghostly coloured copies of the next shot race in from five directions,
    shrinking and converging, and lock together into one on the beat with a gold flash."""
    v = clamp((t - a) / (b - 0.06 - a)); e = eout(v)
    fa = layer(A, min(t, c - 1e-3)); fb = layer(B, t)
    base = fa * (1 - ease(clamp((t - (c - 0.10)) / 0.14))) + fb * ease(clamp((t - (c - 0.10)) / 0.14))
    f = base.copy()
    if 'remfeather' not in _K4:
        fm = np.zeros((H, W), np.float32); cv2.rectangle(fm, (60, 60), (W - 60, H - 60), 1.0, -1)
        _K4['remfeather'] = cv2.GaussianBlur(fm, (0, 0), 50)
    fm = _K4['remfeather']
    dirs = ((-1.0, -0.4, RED_T), (1.0, -0.2, GOLD_T), (0.0, 1.0, BLUE_T))
    for k, (dx, dy, col) in enumerate(dirs):
        lag = 0.05 * k; vv = clamp((t - a - lag) / (c + 0.02 - a - lag)); ee = eout(vv)
        if vv <= 0 or vv >= 1: continue
        s = 1.20 - 0.20 * ee; ox, oy = dx * 0.55 * W * (1 - ee), dy * 0.35 * H * (1 - ee)
        M = np.float32([[s, 0, W / 2 * (1 - s) + ox], [0, s, H / 2 * (1 - s) + oy]])
        g = cv2.warpAffine(fb * fm[..., None], M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        g = np.clip(_hspeed(g, 0.25 * (1 - ee)) * col * 1.25, 0, 1)
        f = 1 - (1 - np.clip(f, 0, 1)) * (1 - g * 0.75 * (1 - vv) ** 0.5)
    return f + 0.50 * math.exp(-((t - c) / 0.05) ** 2)


# ---- struck by lightning (loop)
def _struck_bolt(seed, tgt):
    rng = np.random.default_rng(seed)
    return bolt_tree((tgt[0] + rng.normal(0, 120), -40), tgt, rng, branches=6, disp=0.16, depth=7)


def t_struck(t, A, a, end):
    """Loop: the night Barry was struck. A stepped lightning leader feels its way down the sky toward him, the return
    stroke hits in a white-out, the bolt strobes twice, and out of the glare is the first shot (Barry, blood on his
    face) with the strike's afterglow still crackling."""
    if t >= end: return layer(0, t - DUR)
    tgt = (W / 2, 0.33 * H); ts = a + 0.38
    if t < ts:                                                                                   # the leader descends
        fa = layer(A, t) * (1 - 0.35 * ease(clamp((t - a) / 0.3)))
        u = clamp((t - a) / (ts - a)); segs = []
        for pts, w in _struck_bolt(5, tgt):
            n = max(2, int(len(pts) * min(1.0, u * (1.4 if w == 1.0 else 1.0))))
            segs.append((pts[:n], w))
        f = add(fa, draw_bolts(segs, 2.2, glow=BLUE_GLOW, core=np.float32([1, 1, 1]), amp=0.6 + 0.4 * u))
        return f + 0.05 * u
    dt = t - ts; fl = math.exp(-dt * 7)
    for t0 in (0.09, 0.19):                                                                        # two re-strikes
        if dt >= t0: fl = max(fl, 0.75 * math.exp(-(dt - t0) * 22))
    f0 = layer(0, t - DUR); k = ease(clamp((dt - 0.10) / (end - ts - 0.10)))
    last = layer(A, ts) if dt < 0.12 else f0
    base = last * (1 - k) + f0 * k
    white = np.float32([1.0, 0.97, 0.94])
    f = base * (1 - 0.85 * fl) + white * 0.85 * fl * (1 - 0.3 * k)
    seed = 5 if dt < 0.09 else (6 if dt < 0.19 else 7)
    f = add(f, draw_bolts(_struck_bolt(seed, tgt), 5.0, glow=BLUE_GLOW, core=np.float32([1, 1, 1]), amp=1.6 * max(fl, 0.25 * (1 - k))))
    jolt = int(16 * math.exp(-dt * 14) * math.sin(dt * 80))
    return np.roll(f, jolt, 0)


# ---- molecular phase: the frame vibrates apart in bands and phases into the next shot
def t_phaseslice(t, c, A, B, a, b, n=26):
    """Barry phases through: the frame splits into horizontal bands that vibrate sideways at speed (blue-white glow
    on their edges, a slight colour split), each band snapping over to the next shot at its own instant around the
    beat, then the vibration dies down."""
    env = ease(clamp((t - a) / (c - a))) if t < c else 1 - ease(clamp((t - c) / (b - c)))
    fa = layer(A, min(t, c - 1e-3)); fb = layer(B, t)
    rng = np.random.default_rng(14); sw = c + (rng.random(n) - 0.5) * 0.22
    hb = H / n; f = np.empty_like(fa); edges = np.zeros((H, W), np.float32)
    for k in range(n):
        y0, y1 = int(k * hb), int((k + 1) * hb)
        src = fb if t >= sw[k] else fa
        off = int(env * W * 0.07 * math.sin(t * 140 + k * 1.9) * (0.5 + rng.random()))
        band = np.roll(src[y0:y1], off, 1)
        if env > 0.15:
            band = band.copy(); sp = int(10 * env); band[..., 2] = np.roll(band[..., 2], sp, 1); band[..., 0] = np.roll(band[..., 0], -sp, 1)
        f[y0:y1] = band
        if abs(t - sw[k]) < 0.05: f[y0:y1] = f[y0:y1] + 0.35 * (1 - abs(t - sw[k]) / 0.05)
        edges[max(0, y0 - 1):y0 + 2] = 1.0
    glow = cv2.GaussianBlur(edges, (0, 0), 3) * 1.5 + edges
    return f + (glow * env * 0.45)[..., None] * BLUE_GLOW


# ---- speed-out whip loop
def t_whiploop(t, A, a, end):
    """Loop: gold lightning gathers round Barry on the last shot, he tears off to the right as a red-gold blur with
    a lightning trail, and the camera whip-pans after him across the gap -- the whip eases into the first shot of the
    edit, so the end runs straight into the beginning without a cut."""
    if t >= end: return layer(0, t - DUR)
    t1 = a + 0.28; gap = 0.18 * W; span = W + gap
    p = ease(clamp((t - t1) / (end - t1))); vel = 6 * clamp((t - t1) / (end - t1)) * (1 - clamp((t - t1) / (end - t1)))
    X = p * span                                                                                # camera position
    fa = layer(A, t); f0 = layer(0, t - DUR)
    charge = ease(clamp((t - a) / (t1 - a + 0.10)))
    if t < t1 + 0.12:                                                                            # lightning gathers round him
        rng = np.random.default_rng(int(t * FPS)); segs = []
        cx, cy = W * 0.50, H * 0.40
        for _ in range(int(3 + 6 * charge)):
            ang = rng.uniform(0, 2 * math.pi); r0 = rng.uniform(0.18, 0.32) * W
            p0 = (cx + r0 * math.cos(ang), cy + 1.3 * r0 * math.sin(ang)); L = rng.uniform(60, 180)
            segs.append((bolt_path(p0, (p0[0] + L * math.cos(ang + 1.2), p0[1] + L * math.sin(ang + 1.2)), rng, 5, 0.16), 0.7))
        fa = add(fa, draw_bolts(segs, 2.8, amp=1.1 * charge))
    if t >= t1 - 0.04:                                                                           # Barry blurs out to the right
        k = clamp((t - (t1 - 0.04)) / 0.20)
        trail = fa.copy()
        for j in range(1, 6):
            g = np.roll(fa, int(j * 60 * k), 1) * (RED_T if j % 2 else GOLD_T)
            trail = np.maximum(trail, g * (0.7 - 0.1 * j) * k)
        fa = trail
    canvas = np.zeros((H, W, 3), np.float32) + np.float32([0.03, 0.03, 0.05])
    def place(img, x0):
        x0 = int(round(x0)); xs, xe = max(0, x0), min(W, x0 + W)
        if xe > xs: canvas[:, xs:xe] = img[:, xs - x0:xe - x0]
    place(fa, -X); place(f0, span - X)
    kb = int(1 + 220 * vel) | 1                                                                   # motion blur with the camera's speed
    if kb >= 3: canvas = cv2.blur(canvas, (kb, 1))
    if t >= t1:                                                                                  # the lightning trail he leaves, across the gap
        ys = H * 0.46; sx0, sx1 = W * 0.55 - X, span + W * 0.40 - X
        head = lerp(sx0, sx1, clamp((t - t1) / (end - t1) * 1.15))
        xs = np.linspace(sx0, head, 50); ys_ = ys + 40 * np.sin(xs / 210) + np.random.default_rng(int(t * FPS)).normal(0, 6, 50)
        fade = 1 - ease(clamp((t - (end - 0.18)) / 0.18))
        canvas = add(canvas, draw_bolts([(np.stack([xs, ys_], 1), 1.0)], 6.0, amp=1.4 * fade))
        canvas = add(canvas, draw_bolts([(np.stack([xs, ys_ + 16], 1), 0.6)], 3.0, glow=np.float32([0.15, 0.25, 1.0]), amp=0.9 * fade))
    return canvas


# ============================================================ Flash kit IX (Still Friends v5)
# ---- emblem match cut (the drop)
def _bright_blob(img):
    """Centre and radius of the brightest compact blob in a frame (the glowing chest emblem)."""
    l = cv2.GaussianBlur(luma(img), (0, 0), 6)
    roi = np.zeros_like(l); roi[int(0.45 * H):int(0.85 * H), int(0.2 * W):int(0.8 * W)] = 1   # the chest, not the sky
    l = l * roi; thr = np.percentile(l[roi > 0], 98.5)
    m = (l >= thr).astype(np.uint8); n, lab, st, cen = cv2.connectedComponentsWithStats(m)
    if n <= 1: return W / 2, 0.62 * H, 0.12 * W
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    return float(cen[k][0]), float(cen[k][1]), float(max(st[k, cv2.CC_STAT_WIDTH], st[k, cv2.CC_STAT_HEIGHT])) * 0.55


def t_emblemmatch(t, c, A, B, a, b):
    """The drop: through the hush gold lightning traces the Flash's ring over Barry, the pearl disc materialises and
    the bevelled bolt strikes into it with a flash; then the emblem flies down and locks onto the real glowing emblem
    on his suit in the next shot (a match cut), with a ring of light as it seats."""
    cx0, cy0, R0 = W / 2, 0.40 * H, 0.30 * W
    key = ('ematch', B, round(c, 3))
    if key not in _K4: _K4[key] = _bright_blob(layer(B, c + 0.30))
    ex, ey, Re = _K4[key]
    if t < c:
        p = clamp((t - a) / (c - a))
        fa = layer(A, t) * (1 - 0.45 * ease(p))
        ring_u = eout(clamp(p / 0.55)); disc_u = clamp((p - 0.48) / 0.22); bolt_u = ein(clamp((p - 0.74) / 0.16))
        glint = lerp(-1.8, 1.8, ease(clamp((p - 0.88) / 0.12))) if p > 0.88 else None
        f = draw_emblem(fa, cx0, cy0, R0, math.radians(lerp(-10, 0, eout(p))), ring_u, disc_u, bolt_u, glint)
        if ring_u < 1:                                                                          # lightning crawling along the drawn ring
            ang = -math.pi / 2 + ring_u * 2 * math.pi; rng = np.random.default_rng(int(t * FPS)); segs = []
            hx, hy = cx0 + R0 * math.cos(ang), cy0 + R0 * math.sin(ang)
            for _ in range(3):
                a2 = ang + rng.uniform(-1.2, 1.2); L = rng.uniform(40, 110)
                segs.append((bolt_path((hx, hy), (hx + L * math.cos(a2), hy + L * math.sin(a2)), rng, 5, 0.18), 0.7))
            f = add(f, draw_bolts(segs, 2.6, amp=1.2))
        land = clamp((p - 0.88) / 0.12)
        if 0 < land: f = f + 0.35 * land
        return f
    v = clamp((t - c) / (b - c)); m = eout(clamp(v / 0.72))
    fb = layer(B, t, 1.0 + 0.06 * (1 - m))
    cx, cy, R = lerp(cx0, ex, m), lerp(cy0, ey, m), lerp(R0, Re, m)
    al = 1 - ease(clamp((v - 0.62) / 0.30))
    f = draw_emblem(fb * (0.75 + 0.25 * m), cx, cy, R, 0.0, 1.0, 1.0, 1.0, None, alpha=al)
    r = np.sqrt((XX - cx) ** 2 + (YY - cy) ** 2)
    if v > 0.55:                                                                                 # seats with a ring of light
        k = clamp((v - 0.55) / 0.45); Rr = R * (1 + 2.2 * eout(k))
        f = f + (np.exp(-((r - Rr) / (6 + 10 * k)) ** 2) * 1.2 * (1 - k))[..., None] * GOLD_GLOW
    return f + 0.45 * math.exp(-(t - c) * 18)


# ---- light-speed warp
def t_warp(t, c, A, B, a, b):
    """Faster than light: the shot stretches toward the vanishing point in a radial streak of gold and white star
    lines, flares white on the beat, and the next shot decelerates out of the streaks."""
    cx, cy = W / 2, 0.44 * H
    if t < c: u = ease(clamp((t - a) / (c - a))); src = layer(A, t)
    else: u = 1 - ease(clamp((t - c) / (b - c))); src = layer(B, t)
    acc = None
    for j in range(7):                                                                           # radial (zoom) motion blur
        s = 1 + 0.55 * u * u * j / 6
        M = np.float32([[s, 0, cx * (1 - s)], [0, s, cy * (1 - s)]])
        im = cv2.warpAffine(src, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        acc = im if acc is None else acc + im
    f = acc / 7
    q = 2; st = np.zeros((H // q, W // q), np.float32); rng = np.random.default_rng(31)
    phase = t * 3.0
    for i in range(160):                                                                         # star streaks racing outward
        ang = rng.uniform(0, 2 * math.pi); sp = rng.uniform(0.6, 1.4); d0 = (rng.uniform(0, 1) + phase * sp) % 1.0
        r0 = 30 + d0 ** 2 * 1100; L = (40 + 600 * d0 ** 2) * u
        x0, y0 = cx + r0 * math.cos(ang), cy + r0 * math.sin(ang); x1, y1 = cx + (r0 + L) * math.cos(ang), cy + (r0 + L) * math.sin(ang)
        cv2.line(st, (int(x0 / q), int(y0 / q)), (int(x1 / q), int(y1 / q)), float(0.4 + 0.6 * d0), 1, cv2.LINE_AA)
    st = cv2.resize(st, (W, H)); glow = cv2.GaussianBlur(st, (0, 0), 4)
    f = f + (st * 0.9 + glow * 1.2)[..., None] * (GOLD_T * 0.6 + 0.4) * u
    r = np.sqrt((XX - cx) ** 2 + (YY - cy) ** 2)
    f = f + (np.exp(-(r / (60 + 400 * u)) ** 2) * 0.35 * u)[..., None] * np.float32([0.8, 0.92, 1.0])
    return f + 0.30 * math.exp(-((t - c) / 0.03) ** 2)


# ---- the future newspaper
def _flash_paper_sprite():
    if 'fpaper' in _DK: return _DK['fpaper']
    from PIL import Image, ImageDraw
    S = 2; pw, ph = 930 * S, 1330 * S
    img = Image.new('RGB', (pw, ph), (234, 228, 212)); d = ImageDraw.Draw(img); ink = (24, 22, 22)
    mast = _font_px(SERIF_B, 70 * S); t1 = 'CENTRAL CITY'; t2 = 'PICTURE NEWS'
    d.text(((pw - d.textlength(t1, font=mast)) / 2, 18 * S), t1, font=mast, fill=ink)
    d.text(((pw - d.textlength(t2, font=mast)) / 2, 92 * S), t2, font=mast, fill=ink)
    d.line([(40 * S, 182 * S), (pw - 40 * S, 182 * S)], fill=ink, width=5 * S); d.line([(40 * S, 222 * S), (pw - 40 * S, 222 * S)], fill=ink, width=2 * S)
    sm = _font_px(SERIF_B, 26 * S)
    d.text((44 * S, 189 * S), 'WEDNESDAY, APRIL 25, 2024', font=sm, fill=ink); t3 = 'FINAL EDITION'; d.text((pw - 44 * S - d.textlength(t3, font=sm), 189 * S), t3, font=sm, fill=ink)
    hs = 120
    while d.textlength('FLASH MISSING', font=_font_px(SERIF_B, hs * S)) > pw - 90 * S: hs -= 2
    hl = _font_px(SERIF_B, hs * S); d.text(((pw - d.textlength('FLASH MISSING', font=hl)) / 2, 236 * S), 'FLASH MISSING', font=hl, fill=ink)
    h2 = _font_px(SERIF_B, int(hs * 0.62) * S); d.text(((pw - d.textlength('VANISHES IN CRISIS', font=h2)) / 2, (250 + hs) * S), 'VANISHES IN CRISIS', font=h2, fill=ink)
    by = _font_px(SERIF_B, 28 * S); t4 = 'By Iris West-Allen'; d.text(((pw - d.textlength(t4, font=by)) / 2, (262 + hs + int(hs * 0.62)) * S), t4, font=by, fill=(70, 66, 60))
    px0, py0, pww, phh = 44 * S, 510 * S, 448 * S, 796 * S
    d.rectangle([px0 - 3 * S, py0 - 3 * S, px0 + pww + 3 * S, py0 + phh + 3 * S], outline=ink, width=3 * S)
    r = np.random.default_rng(12); x0 = px0 + pww + 30 * S
    for col in range(2):
        cx = x0 + col * 196 * S; y = py0
        while y < ph - 50 * S:
            if r.random() < 0.08: y += 22 * S; continue
            w_ = int((176 if r.random() > 0.15 else r.uniform(60, 160)) * S); d.rectangle([cx, y, cx + w_, y + 9 * S], fill=(95, 90, 84)); y += 22 * S
    a = np.asarray(img).astype(np.float32)[..., ::-1] / 255
    n = cv2.resize(fbm(ph // 16, pw // 16, 4, 77, 6), (pw, ph)); a *= (0.90 + 0.14 * n)[..., None]
    yy, xx = np.mgrid[0:ph, 0:pw].astype(np.float32)
    a *= (1 - 0.18 * ss(0.25, 0.0, np.minimum(np.minimum(xx, pw - xx), np.minimum(yy, ph - yy)) / pw))[..., None]
    a *= (1 - 0.10 * np.exp(-((xx - pw / 2) / (6 * S)) ** 2))[..., None]
    _DK['fpaper'] = (np.dstack([a, np.ones((ph, pw), np.float32)]), (px0, py0, pww, phh), S)
    return _DK['fpaper']


# ---- Speed Force memories (loop)
_MEM = {}


def _mem_assets():
    if 'thumbs' in _MEM: return _MEM
    n = len(SHOTS); _MEM['thumbs'] = []
    for k in range(n):
        s = SHOTS[k]; fr = layer(k, (s['start'] + s['end']) / 2)
        _MEM['thumbs'].append(cv2.resize(fr, (W // 2, H // 2), interpolation=cv2.INTER_AREA))
    hh = H * 2; ww = int(W * (n + 3) * 0.55)
    fl = cv2.resize(fbm(hh // 16, ww // 16, 5, 51, 4), (ww, hh), interpolation=cv2.INTER_CUBIC)
    rib = (0.5 + 0.5 * np.sin(np.arange(hh, dtype=np.float32)[:, None] / 37 + fl * 9)) ** 30
    base = np.zeros((hh, ww, 3), np.float32) + np.float32([0.10, 0.04, 0.02])
    base += (fl ** 3)[..., None] * np.float32([0.45, 0.20, 0.06])
    base += (rib * fl)[..., None] * np.float32([0.25, 0.65, 1.0]) * 0.8
    _MEM['void'] = base.astype(np.float32)
    return _MEM


def t_memories(t, A, a, end):
    """Loop: the last shot shrinks into a glowing panel and the camera pulls back into the Speed Force, where every
    shot of the edit hangs as a memory panel in a line through the void; it races back along them to the first and
    dives into it, which is the start of the edit."""
    if t >= end: return layer(0, t - DUR)
    M = _mem_assets(); n = len(SHOTS); ps = 0.36; gap = 0.50 * W
    u = clamp((t - a) / (end - a)); zmax = 1 / ps
    xk = lambda k: k * gap; yk = lambda k: 0.05 * H * math.sin(k * 1.3)
    if u < 0.25: e = ease(u / 0.25); z = math.exp(math.log(zmax) * (1 - e)); cam = (xk(n - 1), yk(n - 1) * (1 - e) + 0)
    elif u < 0.75: e = ease((u - 0.25) / 0.50); z = 1.0 - 0.12 * math.sin(math.pi * e); cam = (lerp(xk(n - 1), xk(0), e), 0.0)
    else: e = ein((u - 0.75) / 0.25); z = math.exp(math.log(zmax) * e); cam = (xk(0), yk(0) * e)
    speed = abs(xk(n - 1) - xk(0)) / (0.50 * (end - a)) * (0.25 <= u < 0.75) * math.sin(math.pi * clamp((u - 0.25) / 0.50))
    void = M['void']; vh, vw = void.shape[:2]
    Mv = np.float32([[z * 0.7, 0, W / 2 - (cam[0] * 0.35 + vw * 0.3) * z * 0.7], [0, z * 0.7, H / 2 - vh / 2 * z * 0.7]])
    f = cv2.warpAffine(void, Mv, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    for k in range(n):
        sx = (xk(k) - cam[0]) * z + W / 2; sy = (yk(k) - cam[1]) * z + H / 2; s = ps * z
        if abs(sx - W / 2) > W / 2 + s * W: continue
        live = (k == n - 1 and u < 0.3) or (k == 0 and u > 0.7)
        img = (layer(k, t) if k == n - 1 else layer(0, t - DUR)) if live else M['thumbs'][k]
        sc = s if live else s * 2
        A_ = np.float32([[sc, 0, sx - sc * img.shape[1] / 2], [0, sc, sy - sc * img.shape[0] / 2]])
        pim = cv2.warpAffine(img, A_, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        pm = cv2.warpAffine(np.ones(img.shape[:2], np.float32), A_, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        rim = cv2.GaussianBlur(pm, (0, 0), 3 + 10 * min(z, 2.0)) - pm * 0.0
        glow = np.clip(cv2.GaussianBlur(pm, (0, 0), 18) - pm, 0, 1)
        f = f * (1 - pm[..., None]) + pim * pm[..., None] + (glow * 1.6)[..., None] * GOLD_GLOW * (1 if z < 2.4 else max(0.0, (2.8 - z) / 0.4))
    race = (0.18 <= u < 0.82) * math.sin(math.pi * clamp((u - 0.18) / 0.64))
    if race > 0.02:                                                                             # Speed Force lightning streams racing past
        rng = np.random.default_rng(int(t * FPS)); segs = []
        for j in range(5):
            y = H * (0.12 + 0.19 * j) + rng.normal(0, 20); xs = np.linspace(-40, W + 40, 30)
            segs.append((np.stack([xs, y + 50 * np.sin(xs / 170 + j + t * 9) + rng.normal(0, 7, 30)], 1), 0.8 if j % 2 else 1.0))
        f = add(f, draw_bolts(segs, 3.0, glow=(GOLD_GLOW, BLUE_GLOW)[int(t * 20) % 2], amp=0.9 * race))
        sp_ = np.zeros((H // 2, W // 2), np.float32)
        for _ in range(90): cv2.circle(sp_, (int(rng.uniform(0, W / 2)), int(rng.uniform(0, H / 2))), 1, float(rng.uniform(0.4, 1.0)), -1)
        f = f + cv2.resize(cv2.GaussianBlur(sp_, (0, 0), 1.2), (W, H))[..., None] * GOLD_T * 1.2 * race
    f = f + (np.exp(-(((XX - W / 2) / (0.7 * W)) ** 2 + ((YY - H / 2) / (0.5 * H)) ** 2)) * 0.18 * (z < 1.5))[..., None] * np.float32([0.9, 0.7, 0.4])
    if speed > 50:
        kb = int(min(91, speed / 120)) | 1; f = cv2.blur(f, (kb, 1))
    return f


# ============================================================ Flash kit X (Still Friends v6)
# ---- the real emblem: light bleeds through, then pull back off Barry's glowing chest emblem (the drop)
def _emblem_contour(img, ex, ey, Re):
    l = cv2.GaussianBlur(luma(img), (0, 0), 2)
    roi = np.zeros_like(l); cv2.circle(roi, (int(ex), int(ey)), int(Re * 2.2), 1.0, -1)
    v = l[roi > 0]; thr = np.percentile(v, 80)
    m = ((l > thr) & (roi > 0)).astype(np.uint8)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cs = sorted(cs, key=cv2.contourArea, reverse=True)[:4]
    return [c_[:, 0, :].astype(np.float64) for c_ in cs if len(c_) > 30]


def t_emblempull(t, c, A, B, a, b):
    """The drop: the glow of Barry's suit emblem bleeds up through the bloodied close-up, brightest light first; on
    the drop the frame is tight on the real emblem, lightning crawling along its outline, and the camera pulls back
    in one eased move until Barry in the suit is revealed."""
    key = ('epull', B, round(c, 3))
    if key not in _K4:
        ref = layer(B, c + 0.25); ex, ey, Re = _bright_blob(ref)
        _K4[key] = (ex, ey, Re, _emblem_contour(ref, ex, ey, Re))
    ex, ey, Re, conts = _K4[key]
    zmax = min(3.0, 0.62 * W / max(Re, 1.0))
    def zoomed(v, tt):
        z = 1 + (zmax - 1) * (1 - ease(v)) ** 1.6
        sx, sy = lerp(W / 2, ex, ease(v)), lerp(H / 2, ey, ease(v))
        M = np.float32([[z, 0, sx - z * ex], [0, z, sy - z * ey]])
        im = cv2.warpAffine(layer(B, tt), M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        k_ = (z - 1) / max(zmax - 1, 1e-3); im = im * (1 - 0.28 * k_)                         # tame the glow while tight on it
        return im, M, z
    if t < c:
        u = clamp((t - a) / (c - a))
        fa = layer(A, t, 1 + 0.10 * ease(u))
        if u < 0.62: return fa * (1 - 0.25 * ease(u / 0.62))
        k = ease((u - 0.62) / 0.38)
        fb, M, z = zoomed(0.0, c)
        key_ = np.clip((luma(fb) - (1.0 - 1.1 * k)) / 0.25, 0, 1)[..., None]                       # brightest light first
        f = fa * (1 - 0.25) * (1 - key_) + fb * key_
        return f + (k ** 3 * 0.25)
    v = clamp((t - c) / (b - c)); fb, M, z = zoomed(v, t)
    segs = []; rng = np.random.default_rng(int(t * FPS))
    amp = 1 - ease(clamp((v - 0.35) / 0.65))
    for cc in conts:                                                                              # arcs crawling the real outline
        P = cc @ M[:, :2].T + M[:, 2]
        for _ in range(3):
            i0 = rng.integers(0, len(P)); L = rng.integers(20, 60); i1 = min(len(P) - 1, i0 + L)
            seg = P[i0:i1:3]
            if len(seg) > 2: segs.append((seg + rng.normal(0, 4 + 6 * z / zmax, seg.shape), 0.8))
    f = add(fb, draw_bolts(segs, 3.6, glow=BLUE_GLOW, core=np.float32([1, 1, 1]), amp=1.4 * amp))
    sxy = M[:, :2] @ np.array([ex, ey]) + M[:, 2]; r = np.sqrt((XX - sxy[0]) ** 2 + (YY - sxy[1]) ** 2)
    Rr = Re * z * (1.0 + 3.5 * eout(clamp(v / 0.5)))
    f = f + (np.exp(-((r - Rr) / 14) ** 2) * 0.9 * (1 - clamp(v / 0.5)))[..., None] * BLUE_GLOW
    return f + 0.30 * math.exp(-(t - c) * 14)


# ---- red match cut (0:10)
def t_redmatch(t, c, A, B, a, b):
    """The camera rushes into Barry's red mask in the fireball (heat haze, radial blur, sparks) until the red fills
    the frame; on the beat the red is the next shot's suit, and the camera pulls straight back out of it."""
    def rush(src, z, cx, cy, blur_k):
        acc = None
        for j in range(6):
            zz = z * (1 + blur_k * j / 5)
            M = np.float32([[zz, 0, W / 2 - zz * cx], [0, zz, H * 0.45 - zz * cy]])
            im = cv2.warpAffine(src, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            acc = im if acc is None else acc + im
        return acc / 6
    if t < c:
        u = clamp((t - a) / (c - a)); e = ein(u)
        src = layer(A, t)
        nz = _ring_noise(161, s=1) - 0.5                                                           # heat haze
        hz = (8 * e) * np.sin(YY / 23 + t * 40 + 6 * nz)
        src = cv2.remap(src, (XX + hz).astype(np.float32), YY, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        f = rush(src, 1 + 5.0 * e, 0.62 * W, 0.20 * H, 0.35 * e)
        f = add(f, _sparks(t, a + 0.05, W / 2, H * 0.45, 50, 4, vmin=600, vmax=2200, life=0.5, col=ORANGE_GLOW))
        rd = np.float32([0.10, 0.08, 0.62]); kk = ease(clamp((u - 0.55) / 0.45))                # the frame saturates to the suit's red
        return f * (1 - 0.75 * kk) + rd * 0.75 * kk
    u = clamp((t - c) / (b - c)); e = 1 - eout(u)
    key = ('rm_red', B)
    if key not in _K4:                                                                            # the reddest part of the next shot
        fb0 = layer(B, c + 0.3); red = fb0[..., 2] - 0.5 * (fb0[..., 0] + fb0[..., 1])
        red = cv2.GaussianBlur(red, (0, 0), 40); yy, xx = np.unravel_index(int(np.argmax(red)), red.shape); _K4[key] = (float(xx), float(yy))
    rx, ry = _K4[key]
    f = rush(layer(B, t), 1 + 5.0 * e, lerp(W / 2, rx, e), lerp(H * 0.45, ry, e), 0.35 * e)
    rd = np.float32([0.10, 0.08, 0.62]); kk = 1 - ease(clamp(u / 0.45))
    return f * (1 - 0.75 * kk) + rd * 0.75 * kk


# ---- the future newspaper (2024), with a camera move to the byline
def _future_paper():
    if 'futpaper' in _DK: return _DK['futpaper']
    from PIL import Image, ImageDraw
    S = 2; pw, ph = 930 * S, 1330 * S
    img = Image.new('RGB', (pw, ph), (232, 226, 208)); d = ImageDraw.Draw(img); ink = (22, 20, 20)
    sm = _font_px(SERIF_B, 24 * S); d.text((44 * S, 20 * S), 'LATE CITY EDITION', font=sm, fill=ink)
    t0 = '$1.50'; d.text((pw - 44 * S - d.textlength(t0, font=sm), 20 * S), t0, font=sm, fill=ink)
    mast = _font_px(SERIF_B, 64 * S)
    for k_, txt in enumerate(('Central City', 'Picture News')):
        d.text(((pw - d.textlength(txt, font=mast)) / 2, (50 + 66 * k_) * S), txt, font=mast, fill=ink)
    d.line([(40 * S, 196 * S), (pw - 40 * S, 196 * S)], fill=ink, width=5 * S); d.line([(40 * S, 236 * S), (pw - 40 * S, 236 * S)], fill=ink, width=2 * S)
    d.text((44 * S, 203 * S), 'WEDNESDAY, APRIL 25, 2024', font=sm, fill=ink)
    t1 = 'CENTRAL CITY, MO'; d.text((pw - 44 * S - d.textlength(t1, font=sm), 203 * S), t1, font=sm, fill=ink)
    hs = 128
    while d.textlength('FLASH MISSING', font=_font_px(SERIF_B, hs * S)) > pw - 90 * S: hs -= 2
    hl = _font_px(SERIF_B, hs * S); d.text(((pw - d.textlength('FLASH MISSING', font=hl)) / 2, 250 * S), 'FLASH MISSING', font=hl, fill=ink)
    h2 = _font_px(SERIF_B, int(hs * 0.60) * S); y2 = 262 + hs
    d.text(((pw - d.textlength('VANISHES IN CRISIS', font=h2)) / 2, y2 * S), 'VANISHES IN CRISIS', font=h2, fill=ink)
    by = _font_px(SERIF_B, 34 * S); yb = y2 + int(hs * 0.60) + 18; t4 = 'By Iris West-Allen'
    bw_ = d.textlength(t4, font=by); bx = (pw - bw_) / 2; d.text((bx, yb * S), t4, font=by, fill=(40, 36, 32))
    d.line([(40 * S, (yb + 52) * S), (pw - 40 * S, (yb + 52) * S)], fill=ink, width=2 * S)
    px0, py0, pww, phh = 44 * S, (yb + 70) * S, 448 * S, 796 * S
    d.rectangle([px0 - 3 * S, py0 - 3 * S, px0 + pww + 3 * S, py0 + phh + 3 * S], outline=ink, width=3 * S)
    cap = _font_px(SERIF_B, 20 * S); d.text((px0, py0 + phh + 8 * S), 'The Scarlet Speedster, last seen during the Crisis.', font=cap, fill=(70, 66, 60))
    r = np.random.default_rng(12); x0 = px0 + pww + 30 * S
    sub = _font_px(SERIF_B, 30 * S); d.text((x0, py0), 'RED SKIES', font=sub, fill=ink); d.text((x0, py0 + 36 * S), 'VANISH', font=sub, fill=ink)
    for col in range(2):
        cx = x0 + col * 196 * S; y = py0 + 90 * S
        while y < ph - 50 * S:
            if r.random() < 0.08: y += 22 * S; continue
            w_ = int((176 if r.random() > 0.15 else r.uniform(60, 160)) * S); d.rectangle([cx, y, cx + w_, y + 9 * S], fill=(95, 90, 84)); y += 22 * S
    a = np.asarray(img).astype(np.float32)[..., ::-1] / 255
    n = cv2.resize(fbm(ph // 16, pw // 16, 4, 78, 6), (pw, ph)); a *= (0.90 + 0.14 * n)[..., None]
    yy, xx = np.mgrid[0:ph, 0:pw].astype(np.float32)
    a *= (1 - 0.20 * ss(0.25, 0.0, np.minimum(np.minimum(xx, pw - xx), np.minimum(yy, ph - yy)) / pw))[..., None]
    a *= (1 - 0.10 * np.exp(-((yy - ph / 2) / (5 * S)) ** 2))[..., None]                          # horizontal fold
    hl_c = (pw / 2, (250 + hs * 0.6) * S); by_c = (pw / 2, (yb + 20) * S)
    _DK['futpaper'] = (a.astype(np.float32), (px0, py0, pww, phh), hl_c, by_c, bw_)
    return _DK['futpaper']


def t_futurepaper(t, c, A, B, a, b):
    """The 2024 newspaper from The Flash: it drops onto the frame and lands; the camera pans down the FLASH MISSING
    headline to the byline -- By Iris West-Allen -- and settles there with a focus pull; then it sweeps down to the
    front-page photo and dives in, the halftone resolving into the next shot."""
    pap, (px0, py0, pww, phh), hl_c, by_c, bw_ = _future_paper(); ph, pw = pap.shape[:2]
    base = 0.5; pc = np.array([px0 + pww / 2, py0 + phh / 2])
    k0 = (np.array([pw / 2, ph / 2]), 1.0); k1 = (np.array(hl_c), 1.7); k2 = (np.array(by_c), W / (bw_ * base) * 0.80)
    zf = W / (pww * base); k3 = (pc, zf)
    T0, T1, T2, T3 = a + 0.24, a + 0.46, a + 0.72, c - 0.22
    def mix(p, q, u): return (p[0] + (q[0] - p[0]) * u, math.exp(math.log(p[1]) + (math.log(q[1]) - math.log(p[1])) * u))
    rot = 0.0; drop = 0.0
    if t < T0:
        u = clamp((t - a) / (T0 - a)); cam = k0; drop = (1 - ease(u)) * H * 1.1; rot = math.radians(-6) * (1 - ease(u))
    elif t < T1: cam = mix(k0, k1, ease((t - T0) / (T1 - T0)))
    elif t < T2: cam = mix(k1, k2, ease((t - T1) / (T2 - T1)))
    elif t < T3: cam = (k2[0], k2[1] * (1 + 0.06 * (t - T2) / (T3 - T2)))
    else: cam = mix((k2[0], k2[1] * 1.06), k3, ease(clamp((t - T3) / (b - 0.05 - T3))))
    ctr, z = cam; s = base * z; ca, sa = math.cos(rot), math.sin(rot)
    M = np.float32([[s * ca, -s * sa, 0], [s * sa, s * ca, 0]]); M[:, 2] = np.array([W / 2, H / 2 + drop]) - M[:, :2] @ ctr
    fa = layer(A, min(t, c - 1e-3)); bg = blur(fa, 10 * clamp((t - a) / 0.15)) * lerp(1.0, 0.6, clamp((t - a) / 0.25))
    lay = cv2.warpAffine(np.dstack([pap, np.ones(pap.shape[:2], np.float32)]), M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if t < T0:                                                                                   # its own motion blur while it drops
        sp = 2 * (1 - clamp((t - a) / (T0 - a))); kb = int(1 + 70 * sp) | 1
        if kb > 2: lay = cv2.blur(lay, (1, kb))
    sh = cv2.GaussianBlur(np.roll(np.roll(lay[..., 3], 26, 0), 16, 1), (0, 0), 20) * 0.6
    f = bg * (1 - sh[..., None]) * (1 - lay[..., 3:4]) + lay[..., :3] * lay[..., 3:4]
    corners = np.array([[px0, py0], [px0 + pww, py0], [px0 + pww, py0 + phh], [px0, py0 + phh]], np.float64)
    scr = corners @ M[:, :2].T + M[:, 2]
    P = cv2.getPerspectiveTransform(np.float32([[0, 0], [W, 0], [W, H], [0, H]]), np.float32(scr))
    dv = ease(clamp((t - T3) / (b - 0.05 - T3))); fb = layer(B, t); photo = _halftone(fb) * (1 - dv) + fb * dv
    win = cv2.warpPerspective(photo, P, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    wm = cv2.warpPerspective(np.ones((H, W), np.float32), P, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)[..., None]
    f = f * (1 - wm) + win * wm
    foc = clamp((t - (T2 - 0.08)) / 0.12) * (1 - clamp((t - T3) / 0.10))                         # focus pull onto the byline
    if foc > 0.01:
        byy = (M[:, :2] @ np.array(by_c) + M[:, 2])[1]
        dm = np.clip(np.abs(YY - byy) / (0.12 * H), 0, 1)[..., None] * foc
        f = f * (1 - dm) + blur(f, 9) * dm
        f = f * (1 - 0.25 * dm)
    if t < T0 + 0.10: f = f + 0.18 * math.exp(-max(0.0, t - T0) * 30) * (t >= T0)
    if t >= T3 and t < c + 0.15:
        sp = math.sin(math.pi * clamp((t - T3) / (c + 0.15 - T3))); kb = int(1 + 60 * sp) | 1
        if kb > 2: f = cv2.blur(f, (1, kb))
    return f


# ---- lightning iris (0:17)
def t_boltiris(t, c, A, B, a, b):
    """A ring of crackling gold lightning closes in on Barry's face like an iris, the world outside going dark; on
    the beat it snaps open on the next shot with a shock of light."""
    def face_c(i, tt):
        s = SHOTS[i]; k = int(round((PRE + tt - s['start']) * FPS)); lm = _landmarks(i, k)
        if lm is None: return (W / 2, 0.40 * H)
        e1, e2, fw = lm; return ((e1[0] + e2[0]) / 2, (e1[1] + e2[1]) / 2 + 0.25 * fw)
    if t < c: u = ease(clamp((t - a) / (c - a))); src = layer(A, t); cx, cy = face_c(A, min(t, c - 1e-3)); Rr = lerp(RFULL * 1.05, 0.16 * W, u)
    else: u = 1 - eout(clamp((t - c) / (b - c))); src = layer(B, t); cx, cy = face_c(B, t); Rr = lerp(RFULL * 1.45, 0.16 * W, u)
    r = np.sqrt((XX - cx) ** 2 + (YY - cy) ** 2)
    inside = ss(Rr + 3, Rr - 3, r)[..., None]
    f = src * inside + src * (1 - inside) * 0.08
    rng = np.random.default_rng(int(t * FPS)); segs = []
    for j in range(3):
        n = 64; th = np.linspace(0, 2 * math.pi, n) + rng.uniform(0, 1)
        rr = Rr * (1 + rng.normal(0, 0.025, n)); pts = np.stack([cx + rr * np.cos(th), cy + rr * np.sin(th)], 1)
        segs.append((pts, 1.0 if j == 0 else 0.6))
    for _ in range(6):
        ang = rng.uniform(0, 2 * math.pi); p0 = (cx + Rr * math.cos(ang), cy + Rr * math.sin(ang)); L = rng.uniform(40, 120)
        segs.append((bolt_path(p0, (p0[0] + L * math.cos(ang), p0[1] + L * math.sin(ang)), rng, 5, 0.2), 0.5))
    fade = 1.0 if t < c else ease(clamp(u / 0.5))
    f = add(f, draw_bolts(segs, 4.0, amp=1.3 * (0.4 + 0.6 * u) * fade))
    return f + 0.30 * math.exp(-abs(t - c) * 30)


# ---- Speed Force memories, 3-D (loop)
def _mem3d_assets():
    M = _mem_assets()
    if 'thumbs_b' not in M: M['thumbs_b'] = [cv2.GaussianBlur(th, (0, 0), 5) for th in M['thumbs']]
    return M


def t_memories3d(t, A, a, end):
    """Loop: the last shot's border lights up gold and it shrinks into a memory panel; the camera pulls back into the
    Speed Force, where every shot of the edit hangs as a panel along a curve, each swinging toward the lens as the
    camera races past (depth of field on the far ones, lightning streams, a slight roll); it reaches the first panel,
    which turns to face it, and the camera dives in to the start of the edit."""
    if t >= end: return layer(0, t - DUR)
    Mm = _mem3d_assets(); n = len(SHOTS); ps = 0.36; gap = 0.52 * W
    u = clamp((t - a) / (end - a)); zmax = 1 / ps
    xk = lambda k: k * gap; yk = lambda k: 0.06 * H * math.sin(k * 1.1)
    if u < 0.24: e = ease(u / 0.24); z = math.exp(math.log(zmax) * (1 - e)); camx = xk(n - 1); camy = yk(n - 1) * (1 - e)
    elif u < 0.76: e = ease((u - 0.24) / 0.52); z = 1.0 - 0.14 * math.sin(math.pi * e); camx = lerp(xk(n - 1), xk(0), e); camy = 0.0
    else: e = ein((u - 0.76) / 0.24); z = math.exp(math.log(zmax) * e); camx = xk(0); camy = yk(0) * e
    race = (0.24 <= u < 0.76) * math.sin(math.pi * clamp((u - 0.24) / 0.52))
    void = Mm['void']; vh, vw = void.shape[:2]
    Mv = np.float32([[z * 0.7, 0, W / 2 - (camx * 0.35 + vw * 0.3) * z * 0.7], [0, z * 0.7, H / 2 - vh / 2 * z * 0.7]])
    f = cv2.warpAffine(void, Mv, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    order = sorted(range(n), key=lambda k: -abs((xk(k) - camx) * z))                             # far panels first
    for k in order:
        sx = (xk(k) - camx) * z + W / 2; sy = (yk(k) - camy) * z + H / 2
        hw, hh = ps * z * W / 2, ps * z * H / 2
        if abs(sx - W / 2) > W / 2 + 2.2 * hw: continue
        off = max(-1.0, min(1.0, (sx - W / 2) / W)); yaw = off * math.radians(40) * (1 if z < 2.0 else 0)
        cw = math.cos(yaw); kL = 1 + 0.22 * math.sin(yaw); kR = 1 - 0.22 * math.sin(yaw)
        quad = np.float32([[sx - hw * cw, sy - hh * kL], [sx + hw * cw, sy - hh * kR], [sx + hw * cw, sy + hh * kR], [sx - hw * cw, sy + hh * kL]])
        live = (k == n - 1 and u < 0.28) or (k == 0 and u > 0.72)
        if live: img = layer(n - 1, t) if k == n - 1 else layer(0, t - DUR)
        else: img = Mm['thumbs_b'][k] if abs(off) > 0.35 else Mm['thumbs'][k]
        hI, wI = img.shape[:2]
        Pm = cv2.getPerspectiveTransform(np.float32([[0, 0], [wI, 0], [wI, hI], [0, hI]]), quad)
        pim = cv2.warpPerspective(img, Pm, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        pm = cv2.warpPerspective(np.ones((hI, wI), np.float32), Pm, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        edge_on = 1.0 if not (k == n - 1 and u < 0.24) else ease(u / 0.20)
        gfade = 1.0 if z < 2.2 else max(0.0, (2.7 - z) / 0.5)
        glow = np.clip(cv2.GaussianBlur(pm, (0, 0), 16) - pm, 0, 1) * 1.8 * edge_on * gfade
        rim = np.clip(pm - cv2.erode(pm, np.ones((5, 5), np.uint8)), 0, 1) * edge_on * gfade
        dim = 1.0 - 0.35 * min(1.0, abs(off) * 1.6) * (0 if live else 1)
        f = f * (1 - pm[..., None]) + pim * pm[..., None] * dim + glow[..., None] * GOLD_GLOW + rim[..., None] * np.float32([0.6, 0.9, 1.0])
    if race > 0.02:
        rng = np.random.default_rng(int(t * FPS)); segs = []
        for j in range(5):
            y = H * (0.12 + 0.19 * j) + rng.normal(0, 20); xs = np.linspace(-40, W + 40, 30)
            segs.append((np.stack([xs, y + 50 * np.sin(xs / 170 + j + t * 9) + rng.normal(0, 7, 30)], 1), 0.8 if j % 2 else 1.0))
        f = add(f, draw_bolts(segs, 3.0, glow=(GOLD_GLOW, BLUE_GLOW)[int(t * 20) % 2], amp=0.8 * race))
        kb = int(1 + 70 * race) | 1; f = cv2.blur(f, (kb, 1))
    f = f + (np.exp(-(((XX - W / 2) / (0.7 * W)) ** 2 + ((YY - H / 2) / (0.5 * H)) ** 2)) * 0.15 * (z < 1.5))[..., None] * np.float32([0.9, 0.7, 0.4])
    return f


# ============================================================ Flash kit XI (Still Friends v7)
# ---- the speed pass (0:10)
def t_speedpass(t, c, A, B, a, b):
    """Barry comes past the lens: a white-hot point far off on the left grows into a tapered red-gold comet that
    sweeps across and past the camera -- speed-line texture through its body, the picture smeared sideways in its
    wake, natural gold lightning along its flanks, a tight flare at its head, a jolt as it passes -- and the next shot
    is behind it."""
    p = clamp((t - a) / (b - a)); pc = (c - a) / (b - a)
    hx = lerp(0.06 * W, 1.45 * W, ease(p) * 0.80 + 0.20 * p)
    near = math.exp(-((p - pc) / 0.20) ** 2)
    yc = H * (0.50 + 0.05 * math.sin(p * 3.0)); hh0 = 14 + 0.55 * H * near
    src = layer(A, min(t, c - 1e-3)) if t < c else layer(B, t)
    if t >= c: src = src * (0.6 + 0.4 * ease(clamp((t - c) / (b - c) * 1.6)))
    back = np.clip(hx - XX, 0, None)                                                              # distance behind the head
    taper = np.exp(-back / (0.65 * W)); hh = hh0 * (0.30 + 0.70 * taper)                           # comet: widest at the head
    dy = np.abs(YY - yc) / np.maximum(hh, 1.0); ahead = ss(hx + 90, hx - 30, XX)
    wake = np.exp(-dy ** 2 * 0.5) * ahead * np.exp(-back / (1.1 * W))
    sm = cv2.blur(src, (int(1 + 150 * near) | 1, 1))
    f = src * (1 - wake[..., None] * 0.9) + sm * wake[..., None] * 0.9
    if 'sp_rows' not in _K4:
        r_ = np.random.default_rng(9).random(H).astype(np.float32); _K4['sp_rows'] = np.convolve(r_, np.ones(5) / 5, 'same')
    rows = _K4['sp_rows'][:, None]
    prof = np.exp(-dy ** 2 * 2.0) * ahead * taper
    tex = 0.65 + 0.7 * rows                                                                       # speed lines through the body
    body = np.clip(prof * (0.55 + 0.40 * near) * tex, 0, 0.95)[..., None]
    inner = np.exp(-dy ** 2 * 7.0)[..., None]
    col = np.float32([0.06, 0.10, 0.80]) * (1 - inner) + np.float32([0.25, 0.60, 1.0]) * inner
    f = f * (1 - body) + col * body
    core = np.exp(-dy ** 2 * 40) * ahead * taper * (0.5 + 0.5 * near)
    f = f + core[..., None] * np.float32([0.80, 0.93, 1.0])
    rng = np.random.default_rng(int(t * FPS)); segs = []
    for sg in (-1, 1):                                                                            # natural lightning along both flanks
        for j in range(2):
            x0 = max(-60, hx - rng.uniform(0.5, 1.1) * W); x1 = hx - rng.uniform(0, 0.1) * W
            y0 = yc + sg * hh0 * 0.35 * rng.uniform(0.6, 1.0); y1 = yc + sg * hh0 * rng.uniform(0.55, 0.85)
            segs.append((bolt_path((x0, y0), (x1, y1), rng, 6, 0.07), 1.0 if j == 0 else 0.6))
    f = add(f, draw_bolts(segs, 2.5 + 2.5 * near, amp=1.1))
    r2 = ((XX - hx) / (40 + 120 * near)) ** 2 + ((YY - yc) / (30 + 90 * near)) ** 2
    f = f + (np.exp(-r2) * 0.8)[..., None] * np.float32([0.7, 0.9, 1.0]) + (np.exp(-r2 / 6) * 0.35)[..., None] * np.float32([0.15, 0.30, 1.0])
    f = f + (np.exp(-((YY - yc) / 4) ** 2) * np.exp(-np.abs(XX - hx) / (250 + 350 * near)) * 0.55)[..., None] * np.float32([1.0, 0.85, 0.7])
    jolt = int(14 * near * math.sin(t * 95)); f = np.roll(f, jolt, 0)
    return f


# ---- the emblem flare (the drop)
def t_emblemflare(t, c, A, B, a, b):
    """The drop, in one continuous move: the camera keeps pushing into Barry's face while a point of blue-white light
    swells at the centre and throws rays across him; at the drop the light opens outward (centre first, soft edge)
    onto the real chest emblem tight in frame, a horizontal flare streaking through it and lightning crawling along
    its outline, and the camera keeps moving -- pulling back off the emblem to reveal Barry in the suit."""
    key = ('eflare', B, round(c, 3))
    if key not in _K4:
        ref = layer(B, c + 0.25); ex, ey, Re = _bright_blob(ref)
        _K4[key] = (ex, ey, Re, _emblem_contour(ref, ex, ey, Re))
    ex, ey, Re, conts = _K4[key]
    zmax = min(2.3, 0.55 * W / max(Re, 1.0)); P0 = (W / 2, H / 2)
    r = np.sqrt((XX - P0[0]) ** 2 + (YY - P0[1]) ** 2); th = np.arctan2(YY - P0[1], XX - P0[0])
    def tight(tt, v):
        z = 1 + (zmax - 1) * (1 - ease(v)) ** 1.4
        sx, sy = lerp(W / 2, ex, ease(v)), lerp(H / 2, ey, ease(v))
        M = np.float32([[z, 0, sx - z * ex], [0, z, sy - z * ey]])
        im = cv2.warpAffine(layer(B, tt), M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return im * (1 - 0.22 * (z - 1) / max(zmax - 1, 1e-3)), M, z
    def radial(f, k):
        if k < 0.02: return f
        acc = f.copy()
        for j in (1, 2, 3, 4):
            s = 1 + 0.03 * k * j; Mr = np.float32([[s, 0, P0[0] * (1 - s)], [0, s, P0[1] * (1 - s)]])
            acc += cv2.warpAffine(f, Mr, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return acc / 5
    ts = c - 0.34; tw0, tw1 = c - 0.09, c + 0.05
    if t < tw1:
        u = clamp((t - a) / (c - a)); g = ease(clamp((t - ts) / (c - ts)))
        fa = layer(A, min(t, c - 1e-3), 1 + 0.16 * ease(u))
        fa = radial(fa, g)
        rays = (0.5 + 0.5 * np.cos(th * 23 + 2.0 * np.sin(th * 5 + t * 3))) ** 10 * np.exp(-r / (80 + 900 * g)) * g
        glow = np.exp(-(r / (30 + 260 * g)) ** 2) * (0.4 + 1.2 * g) * (t >= ts)
        f = fa + (rays * 0.55 + glow)[..., None] * np.float32([1.0, 0.85, 0.65])
        if t >= tw0:                                                                            # the light opens onto the emblem
            w = ease(clamp((t - tw0) / (tw1 - tw0))); Rw = lerp(0, RFULL * 1.15, w)
            m = ss(Rw + 220, Rw - 220, r)[..., None]
            fb, M, z = tight(c, 0.0)
            f = f * (1 - m) + (fb + glow[..., None] * np.float32([1.0, 0.85, 0.65]) * 0.5) * m
        return f + 0.25 * math.exp(-((t - c) / 0.05) ** 2)
    v = clamp((t - c) / (b - c)); fb, M, z = tight(t, v)
    fb = radial(fb, 0.6 * (1 - ease(clamp(v / 0.35))))
    amp = 1 - ease(clamp((v - 0.30) / 0.70))
    rng = np.random.default_rng(int(t * FPS)); segs = []
    for cc in conts:
        P = cc @ M[:, :2].T + M[:, 2]
        for _ in range(3):
            i0 = rng.integers(0, len(P)); i1 = min(len(P) - 1, i0 + rng.integers(20, 60)); seg = P[i0:i1:3]
            if len(seg) > 2: segs.append((seg + rng.normal(0, 3 + 5 * z / zmax, seg.shape), 0.8))
    f = add(fb, draw_bolts(segs, 3.4, glow=BLUE_GLOW, core=np.float32([1, 1, 1]), amp=1.3 * amp))
    sxy = M[:, :2] @ np.array([ex, ey]) + M[:, 2]
    f = f + (np.exp(-((YY - sxy[1]) / 6) ** 2) * np.exp(-np.abs(XX - sxy[0]) / 420) * 1.0 * amp)[..., None] * np.float32([1.0, 0.9, 0.75])
    rr = np.sqrt((XX - sxy[0]) ** 2 + (YY - sxy[1]) ** 2); Rr = Re * z * (1.0 + 3.0 * eout(clamp(v / 0.5)))
    return f + (np.exp(-((rr - Rr) / 12) ** 2) * 0.8 * (1 - clamp(v / 0.5)))[..., None] * BLUE_GLOW


# ---- Speed Force memories, v3 (loop)
def _sf_void(t, camx, z, speed):
    """Animated Speed Force background at half resolution: a violet-navy depth gradient, three parallax layers of
    flowing gold and blue energy filaments, drifting motes, all moving with the camera."""
    h, w = H // 2, W // 2; ys = np.arange(h, dtype=np.float32)[:, None]; xs = np.arange(w, dtype=np.float32)[None, :]
    base = np.zeros((h, w, 3), np.float32)
    g = np.exp(-((ys - h / 2) / (0.55 * h)) ** 2)
    base += g[..., None] * np.float32([0.22, 0.06, 0.10]) + (1 - g)[..., None] * np.float32([0.10, 0.03, 0.03])
    lines = np.zeros((h, w, 3), np.float32); rng = np.random.default_rng(77)
    for layer_i, (par, cnt, wd, br) in enumerate(((0.25, 7, 1, 0.35), (0.6, 6, 1, 0.6), (1.2, 4, 2, 0.9))):
        off = camx * par * 0.5 / max(z, 0.4)
        for j in range(cnt):
            y0 = rng.uniform(0.05, 0.95) * h; amp_ = rng.uniform(8, 40); fr = rng.uniform(0.004, 0.012); ph = rng.uniform(0, 6)
            xx = np.arange(0, w + 8, 8, dtype=np.float32)
            yy = y0 + amp_ * np.sin((xx + off) * fr + ph + t * (1.5 + layer_i)) + 6 * np.sin((xx + off) * fr * 3.1 + ph)
            col = (np.float32([0.25, 0.65, 1.0]), np.float32([1.0, 0.72, 0.35]))[(j + layer_i) % 2] * br
            m = np.zeros((h, w), np.float32); cv2.polylines(m, [np.int32(np.stack([xx, yy], 1))], False, 1.0, wd, cv2.LINE_AA)
            lines += m[..., None] * col
    glow = cv2.GaussianBlur(lines, (0, 0), 4) * 1.4 + cv2.GaussianBlur(lines, (0, 0), 14) * 0.8
    out = base + lines * 0.8 + glow
    mo = np.zeros((h, w), np.float32)
    for k in range(120):
        par = rng.uniform(0.3, 1.6); x = (rng.uniform(0, w * 3) - camx * par * 0.5 / max(z, 0.4)) % (w * 1.0); y = rng.uniform(0, h)
        L = 1 + int(min(40, speed * par / 300)); cv2.line(mo, (int(x), int(y)), (int(x + L), int(y)), float(rng.uniform(0.3, 1.0)), 1, cv2.LINE_AA)
    out += (cv2.GaussianBlur(mo, (0, 0), 0.8) * 1.2)[..., None] * np.float32([0.6, 0.85, 1.0])
    return cv2.resize(out, (W, H), interpolation=cv2.INTER_LINEAR)


def t_memories5(t, A, a, end):
    """Loop: a gold light sweeps the last shot as its edges light up and it pulls back into a panel; the camera rises
    into the Speed Force -- every shot of the edit hanging as a glowing memory along a gentle arc, panels banking toward
    the lens as it accelerates past (far ones soft, lightning leaping between them, energy filaments streaming by,
    chromatic fringing at top speed) -- then it slows at the first memory, which turns square to the lens, and the
    camera glides into it: the start of the edit."""
    if t >= end: return layer(0, t - DUR)
    Mm = _mem3d_assets(); n = len(SHOTS); ps = 0.38; gap = 0.50 * W
    u = clamp((t - a) / (end - a)); zmax = 1 / ps
    xk = lambda k: k * gap; yk = lambda k: -0.05 * H * math.sin(k * 0.9)
    S = lambda x: x * x * x * (x * (6 * x - 15) + 10)                                             # smootherstep
    sp = S(clamp((u - 0.10) / 0.78)); camx = lerp(xk(n - 1), xk(0), sp)
    zo = 1 - S(clamp(u / 0.26)); zi = S(clamp((u - 0.76) / 0.24)) ** 1.6
    z = math.exp(math.log(zmax) * max(zo, zi)) * (1 - 0.10 * math.sin(math.pi * sp))
    kk = camx / gap; k0 = int(math.floor(kk)); fr_ = kk - k0
    camy = lerp(yk(k0), yk(min(n - 1, k0 + 1)), fr_) * (1 if 0.05 < sp < 0.95 else 1)
    speed = abs(xk(n - 1) - xk(0)) / (0.78 * (end - a)) * 30 * sp * (1 - sp) / 1.875                # px/s (smootherstep velocity)
    f = _sf_void(t, camx, z, speed)
    f = f * (0.35 + 0.65 * clamp((zmax - z) / (zmax - 1.0)))
    order = sorted(range(n), key=lambda k: -abs(xk(k) - camx))
    for k in order:
        sx = (xk(k) - camx) * z + W / 2; sy = (yk(k) - camy) * z + H / 2
        hw, hh = ps * z * W / 2, ps * z * H / 2
        if abs(sx - W / 2) > W / 2 + 2.4 * hw: continue
        off = max(-1.0, min(1.0, (sx - W / 2) / W)); full = clamp((z - 1.6) / 0.6)
        yaw = off * math.radians(45) * (1 - full); cw = math.cos(yaw); kL = 1 + 0.25 * math.sin(yaw); kR = 1 - 0.25 * math.sin(yaw)
        quad = np.float32([[sx - hw * cw, sy - hh * kL], [sx + hw * cw, sy - hh * kR], [sx + hw * cw, sy + hh * kR], [sx - hw * cw, sy + hh * kL]])
        live = (k == n - 1 and u < 0.30) or (k == 0 and u > 0.70)
        img = (layer(n - 1, t) if k == n - 1 else layer(0, t - DUR)) if live else (Mm['thumbs_b'][k] if abs(off) > 0.40 else Mm['thumbs'][k])
        hI, wI = img.shape[:2]
        Pm = cv2.getPerspectiveTransform(np.float32([[0, 0], [wI, 0], [wI, hI], [0, hI]]), quad)
        pim = cv2.warpPerspective(img, Pm, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        pm = cv2.warpPerspective(np.ones((hI, wI), np.float32), Pm, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        rimk = (1 - full) * (ease(clamp(u / 0.18)) if k == n - 1 else 1.0)
        glow = np.clip(cv2.GaussianBlur(pm, (0, 0), 20) - pm, 0, 1) * 2.0 * rimk
        rim = np.clip(pm - cv2.erode(pm, np.ones((5, 5), np.uint8)), 0, 1) * rimk
        dim = 1.0 - 0.40 * min(1.0, abs(off) * 1.5) * (0 if live else 1)
        f = f * (1 - pm[..., None]) + pim * pm[..., None] * dim + glow[..., None] * GOLD_GLOW + rim[..., None] * np.float32([0.65, 0.9, 1.0])
        if k == n - 1 and u < 0.22:                                                              # a light sweep across the glass
            gl = clamp(u / 0.20); d = ((XX - sx) / max(hw, 1) + (YY - sy) / max(hh, 1) * 0.5) - lerp(-1.8, 1.8, gl)
            f = f + (np.exp(-(d / 0.10) ** 2) * pm * 0.35)[..., None]
    race = math.sin(math.pi * sp) * (0.10 < u < 0.90)
    if race > 0.05:                                                                              # lightning leaping between memories
        rng = np.random.default_rng(int(t * FPS)); segs = []
        for k in range(n - 1):
            x0 = (xk(k) - camx) * z + W / 2 + ps * z * W / 2; x1 = (xk(k + 1) - camx) * z + W / 2 - ps * z * W / 2
            if x1 < -100 or x0 > W + 100 or rng.random() > 0.6: continue
            y0 = (yk(k) - camy) * z + H / 2 + rng.uniform(-0.3, 0.3) * ps * z * H; y1 = (yk(k + 1) - camy) * z + H / 2 + rng.uniform(-0.3, 0.3) * ps * z * H
            segs.append((bolt_path((x0, y0), (x1, y1), rng, 5, 0.25), 0.8))
        f = add(f, draw_bolts(segs, 2.6, amp=1.0 * race))
        sp_px = int(10 * race)                                                                     # chromatic fringing at speed
        if sp_px > 1: f = f.copy(); f[..., 2] = np.roll(f[..., 2], sp_px, 1); f[..., 0] = np.roll(f[..., 0], -sp_px, 1)
        kb = int(1 + 50 * race) | 1; f = cv2.blur(f, (kb, 1))
    if u > 0.80:                                                                                 # converging speed lines into the dive
        q = clamp((u - 0.80) / 0.20); rr = np.sqrt((XX - W / 2) ** 2 + (YY - H / 2) ** 2); th = np.arctan2(YY - H / 2, XX - W / 2)
        ln = (0.5 + 0.5 * np.cos(th * 60)) ** 30 * ss(0.25 * H, 0.9 * H, rr) * math.sin(math.pi * q)
        f = f + (ln * 0.35)[..., None] * np.float32([0.7, 0.9, 1.0])
    return f


# ============================================================ Flash kit XII (Still Friends v8)
# ---- zoom-through into the emblem (the drop)
def t_zoomthru(t, c, A, B, a, b):
    """The drop as one camera move: the camera accelerates into Barry's eyes with a building twist and zoom blur; on
    the drop it is already deep inside the next shot, tight on the real glowing emblem on his chest, and carries the
    same speed and rotation as it decelerates back out to the full shot. A one-frame flash and a few arcs off the
    emblem mark the hit."""
    key = ('zthru', A, B, round(c, 3))
    if key not in _K4:
        ref = layer(B, c + 0.25); ex, ey, Re = _bright_blob(ref)
        s = SHOTS[A]; lm = _landmarks(A, int(round((PRE + (c - 0.05) - s['start']) * FPS)))
        pa_ = ((lm[0][0] + lm[1][0]) / 2, (lm[0][1] + lm[1][1]) / 2) if lm is not None else (W / 2, 0.40 * H)
        _K4[key] = (ex, ey, Re, pa_, _emblem_contour(ref, ex, ey, Re))
    ex, ey, Re, (pax, pay), conts = _K4[key]
    L = math.log(4.0); t0 = c - 0.46; ROT = 14.0
    def view(src, px, py, z, ang, kblur):
        cx, cy = lerp(px, W / 2, min(1.0, (z - 1) / 1.2)), lerp(py, H / 2, min(1.0, (z - 1) / 1.2))
        acc = None; nb = 1 if kblur < 0.02 else 5
        for j in range(nb):
            zz = z * (1 + 0.10 * kblur * j / max(1, nb - 1)); aa = ang + 2.0 * kblur * j / max(1, nb - 1)
            M = cv2.getRotationMatrix2D((px, py), aa, zz); M[0, 2] += cx - px; M[1, 2] += cy - py
            im = cv2.warpAffine(src, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            acc = im if acc is None else acc + im
        return acc / nb, M
    if t < c:
        if t < t0: return layer(A, t, 1 + 0.05 * ease(clamp((t - a) / (t0 - a))))
        u = clamp((t - t0) / (c - t0)); k = ein(u)
        f, _ = view(layer(A, t, 1.05), pax, pay, math.exp(L * k), ROT * k, u ** 2)
        if u > 0.6:
            sp = int(10 * (u - 0.6) / 0.4); f = f.copy(); f[..., 2] = np.roll(f[..., 2], sp, 1); f[..., 0] = np.roll(f[..., 0], -sp, 1)
        return f + 0.30 * ein(clamp((t - (c - 0.03)) / 0.03))
    v = clamp((t - c) / (b - c)); k = 1 - eout(v)
    f, M = view(layer(B, t), ex, ey, math.exp(L * k), -ROT * k, k ** 2)
    if k > 0.4:
        sp = int(10 * (k - 0.4) / 0.6); f = f.copy(); f[..., 2] = np.roll(f[..., 2], sp, 1); f[..., 0] = np.roll(f[..., 0], -sp, 1)
    amp = 1 - ease(clamp(v / 0.6)); rng = np.random.default_rng(int(t * FPS)); segs = []
    for cc in conts:
        P = cc @ M[:, :2].T + M[:, 2]
        for _ in range(2):
            i0 = rng.integers(0, len(P)); i1 = min(len(P) - 1, i0 + rng.integers(20, 50)); seg = P[i0:i1:3]
            if len(seg) > 2: segs.append((seg + rng.normal(0, 4, seg.shape), 0.8))
    f = add(f, draw_bolts(segs, 3.0, glow=BLUE_GLOW, core=np.float32([1, 1, 1]), amp=1.1 * amp))
    return f + 0.28 * math.exp(-(t - c) * 30)


# ---- speed ramp (0:10)
def t_speedramp(t, c, A, B, a, b):
    """A speed ramp: the fireball shot drops into slow motion, then ramps to high speed (real frame-accumulated motion
    blur) as the camera whips after him; the next shot arrives mid-whip and settles with a slight overshoot, a gold
    lightning streak cracking across the cut."""
    sA = SHOTS[A]
    if t < c:
        u = clamp((t - a) / (c - a))
        g = 0.30 * u + 0.70 * ss(0.45, 1.0, u) * (u - 0.45) * 2.2                                 # slow, then ramping up
        tA = a + (c - a) * g; spd = 0.30 + 3.4 * ss(0.45, 1.0, u)
        n = 1 + int(5 * ss(0.5, 1.0, u)); span = 0.012 * spd * n
        acc = None
        for j in range(n):
            im = layer(A, min(sA['end'] + 0.45, tA + span * j / max(1, n - 1)), 1.0 + 0.08 * ease(u))
            acc = im if acc is None else acc + im
        f = acc / n
        w = ein(clamp((u - 0.55) / 0.45)); zz = 1 + 0.25 * w
        f = cv2.warpAffine(f, np.float32([[zz, 0, W / 2 * (1 - zz) + 0.10 * W * w], [0, zz, H / 2 * (1 - zz)]]), (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        kb = int(1 + 180 * w) | 1
        if kb > 2: f = cv2.blur(f, (kb, 1))
        return f + 0.10 * w
    v = clamp((t - c) / (b - c)); e = eout(v)
    over = 0.04 * math.sin(math.pi * clamp((v - 0.55) / 0.45)) * (v > 0.55)
    zz = 1 + 0.25 * (1 - e)
    f = cv2.warpAffine(layer(B, t), np.float32([[zz, 0, W / 2 * (1 - zz) - 0.10 * W * (1 - e) + over * W], [0, zz, H / 2 * (1 - zz)]]), (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    kb = int(1 + 180 * (1 - e) ** 2) | 1
    if kb > 2: f = cv2.blur(f, (kb, 1))
    amp = 1 - ease(clamp(v / 0.45))
    if amp > 0.02:
        rng = np.random.default_rng(int(t * FPS)); y = H * 0.52 + rng.normal(0, 6)
        xs = np.linspace(-40, W + 40, 34); pts = np.stack([xs, y + 22 * np.sin(xs / 140 + t * 20)], 1)
        f = add(f, draw_bolts([(bolt_path(pts[0], pts[-1], rng, 6, 0.05), 1.0)], 6.0, amp=1.4 * amp))
    return f + 0.25 * math.exp(-(t - c) * 26)


# ---- flight through the memories (loop)
def t_memtunnel(t, A, a, end):
    """Loop: the camera launches forward into the Speed Force -- the last shot swings off to the side as a memory panel
    and rushes past; every other shot of the edit hangs on a spiral corridor of panels turned toward the path, racing
    past the lens with streaks of light and lightning between them; the first shot waits at the end of the corridor
    and grows until it is the frame -- the start of the edit."""
    if t >= end: return layer(0, t - DUR)
    Mm = _mem3d_assets(); n = len(SHOTS); fpx = 1.15 * W
    dz = 0.85 * fpx; R3 = 1.05 * W
    u = clamp((t - a) / (end - a)); S = lambda x: x * x * x * (x * (6 * x - 15) + 10)
    zs = 0.0; zend_panel = zs + fpx + (n - 1) * dz + 0.6 * dz
    zc = lerp(zs, zend_panel - fpx, S(u))
    speed = (zend_panel - fpx - zs) * 30 * u * u * (1 - u) * (1 - u) / (end - a)
    f = np.zeros((H, W, 3), np.float32) + np.float32([0.08, 0.03, 0.04])
    r = np.sqrt((XX - W / 2) ** 2 + (YY - H / 2) ** 2); th = np.arctan2(YY - H / 2, XX - W / 2)
    f = f + (np.exp(-(r / (0.45 * H)) ** 2) * 0.35)[..., None] * np.float32([0.55, 0.35, 0.25])
    q = 2; st = np.zeros((H // q, W // q, 3), np.float32); rng = np.random.default_rng(31)          # streaks of light
    for i in range(170):
        ang = rng.uniform(0, 2 * math.pi); d0 = (rng.uniform(0, 1) + zc / (6 * dz) * rng.uniform(0.6, 1.4)) % 1.0
        r0 = 20 + d0 ** 2 * 1300; Ls = (20 + 500 * d0 ** 2) * min(1.0, speed / 6000 + 0.15)
        p0 = (W / 2 + r0 * math.cos(ang), H / 2 + r0 * math.sin(ang)); p1 = (W / 2 + (r0 + Ls) * math.cos(ang), H / 2 + (r0 + Ls) * math.sin(ang))
        col = (np.float32([1.0, 0.70, 0.30]), np.float32([0.35, 0.75, 1.0]))[i % 2] * float(0.3 + 0.7 * d0)
        cv2.line(st, (int(p0[0] / q), int(p0[1] / q)), (int(p1[0] / q), int(p1[1] / q)), tuple(float(x) for x in col), 1, cv2.LINE_AA)
    st = cv2.resize(st, (W, H)); f = f + st * 0.9 + cv2.GaussianBlur(st, (0, 0), 5) * 1.3
    panels = []
    s1 = ease(clamp(u / 0.30))                                                                    # the last shot swings aside
    panels.append((n - 1, (R3 * 0.9 * s1, -0.10 * H * s1, zs + fpx), -40 * s1))
    for j, k in enumerate(range(n - 2, 0, -1)):
        ang = j * 1.9 + 0.6; panels.append((k, (R3 * math.cos(ang), 0.75 * R3 * math.sin(ang), zs + fpx + (j + 1) * dz), -38 * math.copysign(1, math.cos(ang))))
    panels.append((0, (0.0, 0.0, zend_panel), 0.0))
    proj = []
    for k, (X, Y, Z), yaw in panels:
        d = Z - zc
        if d < 0.08 * fpx: continue
        proj.append((d, k, X, Y, Z, yaw))
    proj.sort(key=lambda p_: -p_[0])
    centers = {}
    for d, k, X, Y, Z, yaw in proj:
        ca, sa = math.cos(math.radians(yaw)), math.sin(math.radians(yaw)); hw, hh = W / 2, H / 2
        cs3 = [(X - hw * ca, Y - hh, Z - hw * sa), (X + hw * ca, Y - hh, Z + hw * sa), (X + hw * ca, Y + hh, Z + hw * sa), (X - hw * ca, Y + hh, Z - hw * sa)]
        if min(c3[2] - zc for c3 in cs3) < 0.05 * fpx: continue
        quad = np.float32([[W / 2 + fpx * c3[0] / (c3[2] - zc), H / 2 + fpx * c3[1] / (c3[2] - zc)] for c3 in cs3])
        if quad[:, 0].max() < -50 or quad[:, 0].min() > W + 50 or quad[:, 1].max() < -50 or quad[:, 1].min() > H + 50: continue
        live = (k == n - 1 and u < 0.35) or (k == 0 and u > 0.6)
        img = (layer(n - 1, t) if k == n - 1 else layer(0, t - DUR)) if live else Mm['thumbs'][k]
        hI, wI = img.shape[:2]
        Pm = cv2.getPerspectiveTransform(np.float32([[0, 0], [wI, 0], [wI, hI], [0, hI]]), quad)
        pim = cv2.warpPerspective(img, Pm, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        pm = cv2.warpPerspective(np.ones((hI, wI), np.float32), Pm, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        full = clamp((fpx / d - 0.75) / 0.25) if k in (0, n - 1) else 0.0
        fog = clamp(1.25 - d / (5 * dz))
        glow = np.clip(cv2.GaussianBlur(pm, (0, 0), 18) - pm, 0, 1) * 1.8 * (1 - full) * fog
        rim = np.clip(pm - cv2.erode(pm, np.ones((5, 5), np.uint8)), 0, 1) * (1 - full)
        f = f * (1 - pm[..., None]) + pim * pm[..., None] * (0.35 + 0.65 * fog) + glow[..., None] * GOLD_GLOW + rim[..., None] * np.float32([0.65, 0.9, 1.0])
        centers[k] = quad.mean(0)
    race = math.sin(math.pi * u) * (0.12 < u < 0.88)
    if race > 0.05 and len(centers) > 1:
        rng = np.random.default_rng(int(t * FPS)); segs = []; ks = sorted(centers)
        for k1, k2 in zip(ks, ks[1:]):
            if rng.random() < 0.22: segs.append((bolt_path(centers[k1], centers[k2], rng, 5, 0.12), 0.7))
        f = add(f, draw_bolts(segs, 2.0, amp=0.6 * race))
    return f


# ---- spray-paint X ("X gon' give it to ya", second time)
def _spray_assets():
    if 'spray' in _K4: return _K4['spray']
    rng = np.random.default_rng(13)
    N = rng.random((H, W)).astype(np.float32); Nb = cv2.GaussianBlur(N, (0, 0), 3); Nb = (Nb - Nb.mean()) / (Nb.std() + 1e-6)
    _K4['spray'] = (N, cv2.resize(fbm(H // 16, W // 16, 4, 77, 4), (W, H), interpolation=cv2.INTER_CUBIC) + 0.12 * Nb)
    return _K4['spray']


def t_spray(t, c, A, B, a, b):
    """A big red X is spray-painted across the shot in two strokes (overspray, wet sheen, runs dripping down); on
    the hit the X tears open and swells until the next shot fills the frame."""
    N, nz_ = _spray_assets(); cx, cy = W / 2, 0.42 * H; ex, ey = 0.40 * W, 0.24 * H
    strokes = (((cx - ex, cy - ey), (cx + ex, cy + ey), c - 0.40, c - 0.25), ((cx + ex, cy - ey), (cx - ex, cy + ey), c - 0.22, c - 0.07))
    u = clamp((t - c) / (b - c)) if t >= c else 0.0
    w = 120 + 2700 * ein(u)
    paint = np.zeros((H, W), np.float32); win = np.zeros((H, W), np.float32); sheen = np.zeros((H, W), np.float32)
    drips = np.zeros((H, W), np.float32); mist = np.zeros((H, W), np.float32)
    for j, ((x0, y0), (x1, y1), ts, te) in enumerate(strokes):
        if t < ts: continue
        p = ease(clamp((t - ts) / (te - ts))); dx, dy = x1 - x0, y1 - y0; L2 = dx * dx + dy * dy; L = math.sqrt(L2)
        sp = np.clip(((XX - x0) * dx + (YY - y0) * dy) / L2, 0, p)
        dist = np.sqrt((XX - x0 - sp * dx) ** 2 + (YY - y0 - sp * dy) ** 2)
        sgn = ((XX - x0) * dy - (YY - y0) * dx) / L
        dn = dist + 30 * (nz_ - 0.5)
        core = ss(w / 2 + 16, w / 2 - 16, dn)
        over_ = (N < 0.5 * np.exp(-np.clip(dist - w / 2, 0, None) / 38)) * (dist > w / 2 - 12) * (1 - u) * 0.85
        paint = np.maximum(paint, np.maximum(core, over_))
        sheen = np.maximum(sheen, np.exp(-((sgn - w * 0.22) / (10 + w * 0.05)) ** 2) * core)
        if t >= c: win = np.maximum(win, ss(w / 2 - 6, w / 2 - 34, dn))
        if t < te:
            hx, hy = x0 + p * dx, y0 + p * dy
            mist = np.maximum(mist, np.exp(-((XX - hx) ** 2 + (YY - hy) ** 2) / (2 * 75 ** 2)) * 0.45)
        rng = np.random.default_rng(40 + j)
        for k in range(7):                                                                   # runs drip down from the stroke
            s_ = rng.uniform(0.08, 0.95)
            if s_ > p: continue
            age = t - (ts + s_ * (te - ts)); Ld = min(380.0, 1400 * max(0.0, age - 0.03) ** 1.5 * rng.uniform(0.6, 1.3))
            xd = x0 + s_ * dx + rng.normal(0, 30); yd = y0 + s_ * dy + w * 0.35
            if Ld > 2:
                cv2.line(drips, (int(xd), int(yd)), (int(xd), int(yd + Ld)), 1.0, int(rng.integers(9, 15)), cv2.LINE_AA)
                cv2.circle(drips, (int(xd), int(yd + Ld)), int(rng.integers(8, 12)), 1.0, -1, cv2.LINE_AA)
    drips *= (1 - u)
    paint = np.maximum(paint, drips)
    red = np.float32([0.07, 0.05, 0.80])
    base = layer(A, min(t, c - 1e-3))
    f = base * (1 - paint[..., None]) + red * (0.85 + 0.15 * nz_)[..., None] * paint[..., None] + (sheen * 0.30)[..., None]
    f = f * (1 - mist[..., None] * 0.6) + red * mist[..., None] * 0.6
    if t >= c:
        fb = layer(B, t, 1.08 - 0.08 * eout(u))
        f = f * (1 - win[..., None]) + fb * win[..., None]
        f = f + 0.25 * (1 - eout(u * 4))
    return f


# ---- vinyl loop: the record scratches back to the start
def _vinyl_assets():
    if 'vinyl' in _K4: return _K4['vinyl']
    Rr = 0.44 * W; Rl = 0.34 * Rr; Rp = 0.80 * Rl
    D = int(2 * Rl + 8); c0 = D / 2
    trgb, tm = _polar_text('SIDE A  ·  MIAMI METRO RECORDS  ·  33 1/3 RPM', INTER_BLACK, 15, Rp + 6, Rl - 5, D, c0, (0.92, 0.92, 0.95), (0.10, 0.08, 0.66))
    Y, X = np.mgrid[0:D, 0:D].astype(np.float32) - c0; r = np.sqrt(X * X + Y * Y)
    lab = np.float32([0.10, 0.08, 0.66]) * (0.92 + 0.08 * cv2.resize(fbm(D // 8, D // 8, 3, 9, 4), (D, D)))[..., None]
    lab = lab * (1 - tm[..., None]) + trgb * tm[..., None]
    lab = lab * (1 - (np.exp(-((r - Rp - 2) / 1.5) ** 2) * 0.9)[..., None]) + (np.exp(-((r - Rp - 2) / 1.5) ** 2) * 0.9)[..., None]
    pl = 0.07 + 0.03 * cv2.resize(fbm(H // 4, W // 32, 3, 3, 4), (W, H))                       # brushed plinth
    pl = pl * (1 + 0.15 * np.sin(YY * 1.3 + 4 * cv2.resize(fbm(H // 32, W // 32, 3, 8, 4), (W, H))))
    _K4['vinyl'] = dict(Rr=Rr, Rl=Rl, Rp=Rp, lab=lab.astype(np.float32), D=D, plinth=np.dstack([pl, pl, pl * 1.05]).astype(np.float32),
                        piv=(W / 2 + 0.42 * W, H / 2 - 0.37 * H))
    return _K4['vinyl']


def _tonearm(f, P, ang, z, lift=0.0):
    """Silver tonearm: pivot base, counterweight, tube and headshell, rotated by ang (radians) about P (screen)."""
    vx, vy = -123.0, 435.0; ca, sa = math.cos(ang), math.sin(ang)
    nx_, ny_ = (vx * ca - vy * sa) * z, (vx * sa + vy * ca) * z
    N = (P[0] + nx_, P[1] + ny_); Cw = (P[0] - nx_ * 0.20, P[1] - ny_ * 0.20)
    m = np.zeros((H, W), np.float32); col = np.zeros((H, W, 3), np.float32); shd = np.zeros((H, W), np.float32)
    def disc(cx_, cy_, r_, v, al=1.0):
        mm = np.zeros((H, W), np.float32); cv2.circle(mm, (int(cx_), int(cy_)), max(1, int(r_)), 1.0, -1, cv2.LINE_AA)
        return mm * al, v
    parts = []
    parts.append(disc(P[0], P[1], 62 * z, np.float32([0.55, 0.55, 0.57])))
    parts.append(disc(P[0], P[1], 40 * z, np.float32([0.30, 0.30, 0.32])))
    parts.append(disc(Cw[0], Cw[1], 44 * z, np.float32([0.16, 0.16, 0.17])))
    tube = np.zeros((H, W), np.float32); cv2.line(tube, (int(P[0]), int(P[1])), (int(N[0]), int(N[1])), 1.0, max(2, int(16 * z)), cv2.LINE_AA)
    parts.append((tube, np.float32([0.70, 0.70, 0.72])))
    hl = np.zeros((H, W), np.float32); cv2.line(hl, (int(P[0] - 4 * z), int(P[1])), (int(N[0] - 4 * z), int(N[1])), 0.8, max(1, int(4 * z)), cv2.LINE_AA)
    hs_ = np.zeros((H, W), np.float32)
    ux, uy = (N[0] - P[0]), (N[1] - P[1]); Ln = math.hypot(ux, uy) + 1e-3; ux, uy = ux / Ln, uy / Ln
    box = cv2.boxPoints(((N[0] + ux * 10 * z, N[1] + uy * 10 * z), (40 * z, 76 * z), math.degrees(math.atan2(uy, ux)) - 90))
    cv2.fillConvexPoly(hs_, np.int32(box), 1.0, cv2.LINE_AA); parts.append((hs_, np.float32([0.08, 0.08, 0.09])))
    cart = np.zeros((H, W), np.float32)
    box = cv2.boxPoints(((N[0] + ux * 22 * z, N[1] + uy * 22 * z), (24 * z, 30 * z), math.degrees(math.atan2(uy, ux)) - 90))
    cv2.fillConvexPoly(cart, np.int32(box), 1.0, cv2.LINE_AA); parts.append((cart, np.float32([0.10, 0.10, 0.75])))
    for mm, v in parts: col = col * (1 - mm[..., None]) + v * mm[..., None]; m = np.maximum(m, mm)
    off = int((14 + 40 * lift) * z)
    shd = cv2.GaussianBlur(np.roll(np.roll(m, off, 0), off // 2, 1), (0, 0), (8 + 14 * lift) * z) * 0.6
    f = f * (1 - shd[..., None]) * (1 - m[..., None]) + col + hl[..., None]
    return f


def t_vinyl(t, A, a, end):
    """Loop: the last shot shrinks into the photo label of a spinning record on a turntable; the tonearm swings in
    and drops the needle, the record is scratched back (and the label is now the first shot), then the camera dives
    into the label as it settles upright onto the start of the edit."""
    if t >= end: return layer(0, t - DUR)
    g = _vinyl_assets(); Rr, Rl, Rp = g['Rr'], g['Rl'], g['Rp']
    T1, T2, T3 = a + 0.26, a + 0.48, a + 0.68
    zmax = RFULL / Rp; w = 324.0
    def spin(tt):
        if tt < T2: return w * (tt - a)
        s_ = clamp((tt - T2) / (T3 - T2)); a2 = w * (T2 - a)
        return a2 - 230 * ease(s_ / 0.45) if s_ < 0.45 else a2 - 230 + 330 * ease((s_ - 0.45) / 0.55)
    if t < T1: z = math.exp(math.log(zmax) * (1 - eout((t - a) / (T1 - a))))
    elif t < T3: z = 1.0 - 0.03 * math.sin(math.pi * (t - T1) / (T3 - T1))
    else: z = math.exp(math.log(zmax) * ein((t - T3) / (end - T3)))
    if t < T3: ang = spin(t)
    else:
        a3 = spin(T3); tgt = math.ceil((a3 + 90) / 360) * 360; ang = lerp(a3, tgt, ease((t - T3) / (end - T3)))
    switched = t >= T2 + 0.45 * (T3 - T2)
    cx, cy = W / 2, H / 2
    Mz = np.float32([[z, 0, cx - z * W / 2], [0, z, cy - z * H / 2]])
    f = cv2.warpAffine(g['plinth'], Mz, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    rr = np.sqrt((XX - cx) ** 2 + (YY - cy) ** 2) + 1e-3; rw = rr / z
    phs = np.arctan2(YY - cy, XX - cx); ar = math.radians(ang)
    # platter with strobe dots, the record (grooves + fixed anisotropic sheen), label
    m_pl = ss(1.085 * Rr + 1, 1.085 * Rr - 1, rw)
    dots = ((np.mod(np.degrees(phs - ar), 3.6) < 1.8) & (rw > 1.045 * Rr) & (rw < 1.075 * Rr))
    plat = 0.50 * (1 + 0.15 * np.sin(rw * 0.7)) * (1 - 0.75 * dots)
    f = f * (1 - m_pl[..., None]) + (plat * m_pl)[..., None] * np.float32([1.0, 1.0, 1.02])
    m_rec = ss(Rr + 1, Rr - 1, rw)
    smooth = ((rw > 0.955 * Rr) | (rw < 0.40 * Rr) | (np.abs(rw - 0.63 * Rr) < 3) | (np.abs(rw - 0.79 * Rr) < 3)).astype(np.float32)
    gro = 0.035 + (0.010 * np.sin(rw * 2.4) + 0.006 * np.sin(rw * 0.37)) * (1 - smooth)
    sheen = 0.40 * np.abs(np.cos(phs - math.radians(-135))) ** 6 * (1 - 0.6 * smooth) + 0.08 * np.abs(np.cos(phs - math.radians(-135))) ** 2
    rim = np.exp(-((rw - Rr + 2) / 2.5) ** 2) * 0.25
    rec = (gro + sheen + rim)[..., None] * np.float32([1.05, 1.0, 0.98])
    f = f * (1 - m_rec[..., None]) + rec * m_rec[..., None]
    D = g['D']; s_ = z
    M = cv2.getRotationMatrix2D((D / 2, D / 2), -ang, s_); M[0, 2] += cx - D / 2; M[1, 2] += cy - D / 2
    labw = cv2.warpAffine(g['lab'], M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    m_lab = ss(Rl + 0.8, Rl - 0.8, rw)[..., None]
    f = f * (1 - m_lab) + labw * m_lab
    live = layer(0, t - DUR) if switched else layer(len(SHOTS) - 1, t)
    f, _ = _live_circle(f, live, cx, cy, Rp * z, ang=-ang, feather=1.2)
    # tonearm: swings in from rest, drops the needle at T2
    P = (cx + (g['piv'][0] - cx) * z, cy + (g['piv'][1] - cy) * z)
    sw_ = eout(clamp((t - (T1 - 0.04)) / 0.16)); arm_ang = lerp(-0.38, 0.0, sw_)
    lift = 1 - ease(clamp((t - (T2 - 0.08)) / 0.08))
    if z < 2.5: f = _tonearm(f, P, arm_ang, z, lift)
    if z < 3:                                                                                # spindle through the label
        sp_ = np.exp(-(rr / (7 * z)) ** 4); f = f * (1 - sp_[..., None]) + sp_[..., None] * (0.55 + 0.4 * np.exp(-(((XX - cx + 2 * z) ** 2 + (YY - cy + 2 * z) ** 2) / (3 * z) ** 2)))[..., None]
    if T2 <= t < T2 + 0.06: f = f + 0.06 * (1 - (t - T2) / 0.06)
    k = ease(clamp((t - (T1 - 0.14)) / 0.18)) * (1 - ease(clamp((t - (T3 - 0.04)) / 0.14)))      # tilt into a turntable view
    if k > 1e-3:
        phi = math.radians(34) * k; f0 = 1800.0
        def pj(x, y):
            X_, Y_ = x - cx, y - cy; Z_ = -Y_ * math.sin(phi); q = f0 / (f0 + Z_)
            return (cx + X_ * q, cy + 0.06 * H * k + Y_ * math.cos(phi) * q)
        src = np.float32([[0, 0], [W, 0], [W, H], [0, H]]); dst = np.float32([pj(*p_) for p_ in src])
        Hm = cv2.getPerspectiveTransform(src, dst)
        wv = cv2.warpPerspective(f, Hm, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        wm = cv2.warpPerspective(np.ones((H, W), np.float32), Hm, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)[..., None]
        f = wv + g['plinth'] * (1 - wm)
    return f


# Transition registry: name -> (window(c, o) -> (start, end), render(t, c, A, B, o)).
# c is the cut time into shot B, o the shot's out_opts. Windows are relative to the cut,
# so the same transition works at any tempo and position.
def _iris_times(c):   # c = end of the last shot; the drop falls, swells and reveals shot 0
    return c - 0.60, c - 0.20, c + 0.12


# ==== Flash kit XIII ("Remedy": prismatic light, collage, title matte, the cortex wall) ====
def _slab_tex():
    """Speed Force light texture for the slabs: long horizontal streaks of fire and electricity (two palettes)."""
    if 'slabtex' in _K4: return _K4['slabtex']
    W2 = 2 * W
    n1 = cv2.resize(fbm(H // 24, W2 // 96, 5, 91, 4), (W2, H), interpolation=cv2.INTER_CUBIC)
    n2 = cv2.resize(fbm(H // 10, W2 // 160, 4, 17, 4), (W2, H), interpolation=cv2.INTER_CUBIC)
    n1 = (n1 - n1.min()) / (n1.max() - n1.min() + 1e-6); n2 = (n2 - n2.min()) / (n2.max() - n2.min() + 1e-6)
    s = np.clip(0.35 + 1.1 * n1 ** 2 + 0.5 * n2 ** 3, 0, 1.6).astype(np.float32)
    rng = np.random.default_rng(5); fil = np.zeros((H, W2), np.float32)              # bright filaments
    for i in range(60):
        y = rng.uniform(0, H); x0 = rng.uniform(0, W2); L = rng.uniform(150, 900)
        xs = np.linspace(x0, x0 + L, 40); ys = y + np.cumsum(rng.normal(0, 2.5, 40))
        cv2.polylines(fil, [np.int32(np.stack([xs, ys], 1) * 4)], False, float(rng.uniform(0.4, 1.0)), 2, cv2.LINE_AA, shift=2)
    fil = fil + cv2.GaussianBlur(fil, (0, 0), 6) * 1.5
    _K4['slabtex'] = (s, fil)
    return _K4['slabtex']


SLAB_COL = {'cyan': np.float32([1.00, 0.80, 0.25]), 'gold': np.float32([0.20, 0.72, 1.00]), 'red': np.float32([0.18, 0.12, 1.00]),
            'white': np.float32([0.92, 0.95, 1.00]), 'violet': np.float32([1.00, 0.35, 0.70])}
_TILT = math.tan(math.radians(13))


def _slab(f, t, y0, hh, head, length, col, tex_off=0.0, alpha=1.0, lead=True):
    """One light slab: a parallelogram band (tilted 13 degrees) shooting in from the right. Its head is crisp and
    white-hot, the body is coloured Speed Force texture with a hot core line, the tail fades out, and it throws a soft
    glow of its colour around it. y0 = band centre at the frame's centre column, hh = band height, head = x of the
    leading edge (slanted), length = body length."""
    v = YY + (XX - W / 2) * _TILT - y0                          # distance across the band
    across = 1 - ss(hh / 2 - 2.5, hh / 2 + 2.5, np.abs(v))
    if across.max() < 1e-3: return f
    x = XX - head + v * 0.35                                     # along the band, from the (slanted) head
    m = across * ss(-3, 3, x) * (1 - ss(length * 0.55, length, x)) * alpha
    if m.max() < 1e-3: return f
    s, fil = _slab_tex(); W2 = s.shape[1]
    sh = int((tex_off + head * 0.6) % W2); sx = np.roll(s, -sh, 1)[:, :W]; fx_ = np.roll(fil, -sh, 1)[:, :W]
    core = np.exp(-(v / (0.16 * hh + 1)) ** 2)
    headg = np.exp(-np.maximum(x, 0) / 60.0)
    C = SLAB_COL[col] if isinstance(col, str) else col
    fill = C[None, None, :] * (0.55 * sx + 0.8 * fx_)[..., None] + (0.55 * core + 0.9 * headg * lead)[..., None] * np.float32([1, 1, 1])
    glow = cv2.GaussianBlur(cv2.resize(m, (W // 4, H // 4), interpolation=cv2.INTER_AREA), (0, 0), 9)
    glow = cv2.resize(glow, (W, H), interpolation=cv2.INTER_LINEAR)
    return f * (1 - 0.92 * m[..., None]) + fill * m[..., None] + glow[..., None] * C * 0.55 * alpha


def _shift(img, dx, dy=0.0, z=1.0, cx=W / 2, cy=H / 2):
    if abs(dx) < 0.5 and abs(dy) < 0.5 and abs(z - 1) < 1e-4: return img
    M = np.float32([[z, 0, cx * (1 - z) + dx], [0, z, cy * (1 - z) + dy]])
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def _hblur(img, k):
    k = int(k) | 1
    return cv2.blur(img, (k, 1)) if k > 2 else img


def t_prism(t, c, A, B, a, b):
    """Light-slab wipe (after the reference): the camera slides left as thin slabs of Speed Force light -- cyan,
    gold, red -- shoot across the frame on the beat; a wide slab arrives carrying the next shot inside it, its
    white-hot edges sweeping open until it is the frame; the camera keeps its slide and settles, and a last red slab
    streaks over the new shot."""
    sp = 3.2 * W / 0.30                                          # slab speed (px/s), right to left
    def hx(t0): return W + 80 - (t - t0) * sp
    if t < c - 0.04:
        u = clamp((t - a) / (c - a)); dx = -0.10 * W * ein(u)
        f = _hblur(_shift(layer(A, t), dx, z=1 + 0.04 * u), 1 + 70 * u ** 2)
    else:
        f = None
    v = clamp((t - (c - 0.10)) / 0.22); e = eout(v)
    B_ = _hblur(_shift(layer(B, t), 0.10 * W * (1 - e), z=1.04 - 0.04 * e), 1 + 70 * (1 - e) ** 2)
    # B arrives inside a wide band (centre 0.47H) that opens to the full frame
    hb = lerp(0.22 * H, 2.4 * H, ein(clamp((t - (c - 0.12)) / 0.20)) ** 0.7)
    if f is None: f = B_
    else:
        vv = YY + (XX - W / 2) * _TILT - 0.47 * H; xx = XX - hx(c - 0.22) + vv * 0.35
        mB = (1 - ss(hb / 2 - 3, hb / 2 + 3, np.abs(vv))) * ss(-3, 3, xx)
        f = f * (1 - mB[..., None]) + B_ * mB[..., None]
        edge = np.exp(-((np.abs(vv) - hb / 2) / 6.0) ** 2) * ss(-3, 3, xx)
        f = f + (edge[..., None] * np.float32([1, 1, 1]) + cv2.GaussianBlur(edge, (0, 0), 14)[..., None] * SLAB_COL['cyan'] * 2.5) * 0.9
    for (y0, hh, t0, L, col, off) in ((0.10 * H, 0.050 * H, a, 1.9 * W, 'cyan', 0), (0.73 * H, 0.085 * H, a + 0.05, 1.6 * W, 'gold', 700),
                                      (0.62 * H, 0.030 * H, a + 0.10, 1.2 * W, 'white', 300), (0.86 * H, 0.060 * H, c - 0.02, 1.8 * W, 'red', 1300),
                                      (0.08 * H, 0.035 * H, c + 0.06, 1.3 * W, 'gold', 400)):
        if t0 <= t <= t0 + 0.50: f = _slab(f, t, y0, hh, hx(t0), L, col, off)
    return f + 0.08 * math.exp(-abs(t - c) * 30)


def t_slab(t, c, A, B, a, b, col='gold'):
    """Glue: one great slab of light streaks across the cut on the beat -- the frame behind its tail is already the
    next shot -- with a thin satellite slab above and below; the camera carries a short slide through it."""
    sp = 2.9 * W / 0.24
    hx = W + 60 - (t - a) * sp; L = 0.95 * W
    u = clamp((t - a) / (c - a)); v = clamp((t - c) / (b - c))
    A_ = _hblur(_shift(layer(A, t), -0.05 * W * ein(u)), 1 + 40 * u * u) if t < c + 0.10 else None
    B_ = _hblur(_shift(layer(B, t), 0.05 * W * (1 - eout(v))), 1 + 40 * (1 - eout(v)) ** 2)
    vv = YY + (XX - W / 2) * _TILT; xx = XX - hx + vv * 0.35
    if A_ is None: f = B_
    else:
        mB = ss(L * 0.30, L * 0.45, xx); f = A_ * (1 - mB[..., None]) + B_ * mB[..., None]
    f = _slab(f, t, 0.50 * H, 2.6 * H, hx, L * 0.55, col, 900, alpha=0.80)
    f = _slab(f, t, 0.10 * H, 0.04 * H, hx + 0.35 * W, 1.4 * W, 'white', 100)
    f = _slab(f, t, 0.84 * H, 0.05 * H, hx - 0.25 * W, 1.4 * W, 'cyan' if col != 'cyan' else 'red', 1500)
    return f


def _panel_src(i, tt, dy, z=1.0):
    return _shift(layer(i, tt), 0, dy, z, W / 2, 0.40 * H)


def t_triptych(t, c, A, B, a, b):
    """The drop as a collage: on the 808's three stutter hits, three panels slam in over the darkened shot -- the
    next three shots of the edit, each framed on its face, sliding in from alternate sides with speed blur and a
    flash -- the gaps between them glowing gold; on the drop the middle panel (the drop shot) bursts open to the full
    frame, pushing the others off the top and bottom, with a flash and a colour split."""
    hits = (c - 0.48, c - 0.30, c - 0.12)
    order = ((0, B + 1, 1), (2, B + 2, -1), (1, B, 1))         # (slot, shot, side)
    ph = H / 3.0; gap = 10
    if t < c:
        u = clamp((t - a) / (c - a))
        k = sum(1 for h in hits if t >= h)
        bump = sum(0.035 * math.exp(-(t - h) * 18) for h in hits if t >= h)
        f = _shift(layer(A, t), 0, 0, 1 + 0.03 * k + bump, W / 2, 0.40 * H) * (1 - 0.55 * ss(hits[0], hits[0] + 0.05, t))
        for j, (slot, sh, side) in enumerate(order):
            h = hits[j]
            if t < h: continue
            p = clamp((t - h) / 0.075); e = eout(p)
            dx = side * (1 - e) * W * 1.05 + side * 0.025 * W * math.sin(math.pi * clamp((t - h - 0.075) / 0.08)) * (t > h + 0.075)
            y0, y1 = int(slot * ph + gap / 2), int((slot + 1) * ph - gap / 2)
            dy = (slot + 0.5) * ph - 0.40 * H
            tt = (S0 := SHOTS[sh])['start'] + 0.15 + (t - h) if sh != B else t
            img = _panel_src(sh, tt, dy, 1.05)
            img = _hblur(img, 1 + 160 * (1 - e))
            img = img + 0.6 * math.exp(-(t - h) * 22)
            M = np.float32([[1, 0, dx], [0, 1, 0]])
            pm = np.zeros((H, W), np.float32); pm[y0:y1] = 1
            pm = cv2.warpAffine(pm, M, (W, H), borderValue=0); pim = cv2.warpAffine(img, M, (W, H), borderMode=cv2.BORDER_REFLECT)
            rim = np.clip(pm - cv2.erode(pm, np.ones((5, 5), np.uint8)), 0, 1)
            f = f * (1 - pm[..., None]) + pim * pm[..., None] + rim[..., None] * 0.9
            gl = cv2.GaussianBlur(cv2.resize(rim, (W // 4, H // 4), interpolation=cv2.INTER_AREA), (0, 0), 6)
            f = f + cv2.resize(gl, (W, H))[..., None] * GOLD_GLOW * 2.2
        return f
    # the drop: middle panel opens to the frame
    v = clamp((t - c) / 0.16); e = eout(v)
    top = lerp(ph + gap / 2, 0, e); bot = lerp(2 * ph - gap / 2, H, e)
    dyB = lerp(0.5 * H - 0.40 * H, 0, e)
    zB = 1.05 + 0.10 * (1 - eout(clamp((t - c) / (b - c))))
    Bf = _panel_src(B, t, dyB, zB)
    sp = int(14 * (1 - eout(clamp((t - c) / 0.25))))
    if sp > 0: Bf = Bf.copy(); Bf[..., 2] = np.roll(Bf[..., 2], sp, 1); Bf[..., 0] = np.roll(Bf[..., 0], -sp, 1)
    f = Bf
    if e < 0.999:
        for slot, sh in ((0, B + 1), (2, B + 2)):
            off = (top - (ph - gap / 2)) if slot == 0 else (bot - (2 * ph + gap / 2))
            y0, y1 = int(slot * ph + gap / 2 + off), int((slot + 1) * ph - gap / 2 + off)
            dy = (slot + 0.5) * ph - 0.40 * H + off
            img = _panel_src(sh, SHOTS[sh]['start'] + 0.15 + (t - (c - 0.48 + (0.18 if slot == 2 else 0))), dy, 1.05)
            img = cv2.blur(img, (1, int(1 + 120 * (1 - e)) | 1))
            m = np.zeros((H, W), np.float32); m[max(0, y0):max(0, min(H, y1))] = 1
            f = f * (1 - m[..., None]) + img * m[..., None]
        ln = np.zeros((H, W), np.float32)
        for yy in (top, bot):
            if 0 <= yy < H: ln[max(0, int(yy) - 3):int(yy) + 3] = 1
        f = f + cv2.GaussianBlur(ln, (0, 0), 3)[..., None] * (1 - e) * 1.5 + cv2.GaussianBlur(ln, (0, 0), 25)[..., None] * GOLD_GLOW * (1 - e) * 3
    return f + 0.18 * math.exp(-(t - c) * 20)


def _rgb_zoom(img, zs, cx, cy, ang=0.0):
    out = np.empty_like(img)
    for ch, z in enumerate(zs):
        M = cv2.getRotationMatrix2D((cx, cy), ang * (ch - 1), z)
        out[..., ch] = cv2.warpAffine(img[..., ch], M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return out


def _radial(img, z0, z1, cx, cy, n=6):
    acc = None
    for j in range(n):
        z = lerp(z0, z1, j / max(1, n - 1)); M = cv2.getRotationMatrix2D((cx, cy), 0, z)
        im = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        acc = im if acc is None else acc + im
    return acc / n


def t_chromasnap(t, c, A, B, a, b):
    """The breath before the hit: while the 808 drops out, the camera eases back off the face and the colour
    channels drift apart like light through a prism; on the beat it snaps forward -- a zoom-blur rush into the next
    shot, which lands with its channels split wide and slams them back together."""
    cx, cy = W / 2, 0.40 * H
    if t < c:
        u = clamp((t - a) / (c - a - 0.07)); k = ease(u)
        snap = clamp((t - (c - 0.07)) / 0.07)
        z = lerp(1.0, 1.06, k) * (1 + 0.55 * ein(snap))
        img = layer(A, t)
        sep = 0.035 * k + 0.05 * snap
        f = _rgb_zoom(img, (z * (1 - sep), z, z * (1 + sep)), cx, cy, 1.2 * k)
        if snap > 0: f = _radial(f, 1.0, 1 + 0.25 * snap, cx, cy)
        vig = 1 - 0.45 * k * np.clip(((XX - cx) ** 2 + (YY - cy) ** 2) / (0.8 * H) ** 2, 0, 1)
        return f * vig[..., None] + 0.06 * snap ** 2
    v = clamp((t - c) / (b - c)); e = eout(v)
    z = lerp(1.30, 1.0, e); sep = 0.07 * (1 - e) ** 1.5
    f = _rgb_zoom(layer(B, t), (z * (1 + sep), z, z * (1 - sep)), cx, cy, -1.5 * (1 - e))
    if e < 0.7: f = _radial(f, 1.0, 1 + 0.18 * (1 - e / 0.7), cx, cy, 5)
    return f + 0.10 * math.exp(-(t - c) * 22)


# ---- THE FLASH title as a window into the next shot
def _title_assets():
    if 'title' in _K4: return _K4['title']
    from PIL import Image, ImageDraw, ImageFont
    fnt = '/usr/share/fonts/opentype/inter/InterDisplay-BlackItalic.otf'
    word, small = 'FLASH', 'THE'
    F = ImageFont.truetype(fnt, 400); wd = F.getlength(word); size = int(400 * 0.94 * W / wd); F = ImageFont.truetype(fnt, size)
    Fs = ImageFont.truetype(fnt, int(size * 0.30))
    letters = []
    x = (W - F.getlength(word)) / 2; yb = int(0.47 * H)
    for ch in word:
        im = Image.new('L', (W, H), 0); ImageDraw.Draw(im).text((x, yb), ch, font=F, fill=255, anchor='ls')
        letters.append(np.asarray(im, np.float32) / 255); x += F.getlength(ch)
    trk = size * 0.10; ws = sum(Fs.getlength(ch) for ch in small) + trk * (len(small) - 1)
    x = (W - ws) / 2 - size * 0.04; ys = yb - int(size * 0.80)
    for ch in small:
        im = Image.new('L', (W, H), 0); ImageDraw.Draw(im).text((x, ys), ch, font=Fs, fill=255, anchor='ls')
        letters.append(np.asarray(im, np.float32) / 255); x += Fs.getlength(ch) + trk
    allm = np.clip(sum(letters), 0, 1)
    dt = cv2.distanceTransform((letters[1] > 0.5).astype(np.uint8), cv2.DIST_L2, 5)          # dive into the L's stem
    ty, tx = np.unravel_index(np.argmax(dt), dt.shape)
    ys_, xs_ = np.nonzero(allm > 0.5); top, bot = ys_.min(), ys_.max()
    grad = np.clip((YY - top) / max(1, bot - top), 0, 1)
    gold = (np.float32([0.55, 0.92, 1.00])[None, None] * (1 - grad[..., None]) + np.float32([0.05, 0.38, 0.95])[None, None] * grad[..., None])
    band = np.exp(-((YY - (top + 0.42 * (bot - top))) / 5.0) ** 2)
    gold = gold * (0.85 + 0.25 * np.sin(XX / 37.0 + YY / 61.0))[..., None] * 0.5 + gold * 0.5 + band[..., None] * 0.5
    _K4['title'] = dict(letters=letters, all=allm, target=(float(tx), float(ty)), gold=gold.astype(np.float32))
    return _K4['title']


def t_title(t, c, A, B, a, b):
    """A graphic into a transition: THE FLASH builds over the quiet bar -- each letter of FLASH punches up out of a
    baseline with speed blur, burnished gold with a hot rim, THE tracking in above -- a glint runs across the metal;
    on the 808's return the letters turn into windows onto the next shot and the camera dives into the L's stem until
    the next shot is the frame."""
    T = _title_assets(); L = T['letters']
    tx, ty = T['target']
    td = c - 0.30                                                # the dive: one eased move, landing on the beat
    k = clamp((t - td) / 0.30); S5k = k * k * k * (k * (6 * k - 15) + 10)
    zdive = math.exp(math.log(40.0) * (0.6 * S5k + 0.4 * k * k))
    # build the title mask (per-letter animation)
    m = np.zeros((H, W), np.float32)
    for j, lm in enumerate(L):
        t0 = a + 0.04 + 0.075 * j if j < 5 else a + 0.40 + 0.05 * (j - 5)
        p = clamp((t - t0) / 0.14)
        if p <= 0: continue
        e = eout(p)
        if j < 5:
            dy = (1 - e) * 0.16 * H; M = np.float32([[1, 0, 0], [0, 1, dy]])
            mm = cv2.warpAffine(lm, M, (W, H)); kb = int(1 + 70 * (1 - e)) | 1
            if kb > 2: mm = cv2.blur(mm, (1, kb))
            mm = mm * (YY < 0.47 * H + 0.02 * H)                  # rises from behind the baseline
        else:
            sc = 1 + 0.6 * (1 - e); M = cv2.getRotationMatrix2D((W / 2, 0.47 * H - 0.25 * W), 0, sc)
            mm = cv2.warpAffine(lm, M, (W, H)) * e
        m = np.maximum(m, mm)
    winB = ss(td - 0.06, td + 0.08, t)                           # letters become windows onto the next shot
    ztot = zdive
    if ztot > 1.0001:
        cxp, cyp = lerp(tx, W / 2, S5k), lerp(ty, H / 2, S5k)
        M = np.float32([[ztot, 0, cxp - ztot * tx], [0, ztot, cyp - ztot * ty]])
        m = cv2.warpAffine(m, M, (W, H), flags=cv2.INTER_LINEAR)
        gold = cv2.warpAffine(T['gold'], M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    else:
        gold = T['gold']
    zA = 1 + 0.06 * clamp((t - a) / (c - a)) + 2.0 * S5k
    A_ = _shift(layer(A, t), 0, 0, zA, W / 2, 0.42 * H)
    if k > 0: A_ = _radial(A_, 1.0, 1 + 0.10 * math.sin(math.pi * k), W / 2, H / 2, 4)
    dim = (1 - 0.40 * ss(a, a + 0.30, t)) * (1 - 0.5 * S5k)
    zb = 1.10 - 0.10 * eout(clamp((t - td) / (b - td)))
    Bf = _shift(layer(B, t), 0, 0, zb, W / 2, 0.42 * H)
    fill = gold * (1 - winB) + Bf * (1 + 0.35 * (1 - S5k)) * winB
    sweep = np.exp(-(((XX - (lerp(-0.3, 1.3, clamp((t - (a + 0.60)) / 0.30)) * W)) + (YY - 0.4 * H) * 0.5) / 26.0) ** 2) * (1 - winB)
    rimw = np.clip(m - cv2.erode(m, np.ones((7, 7), np.uint8)), 0, 1) * (1 - ss(td + 0.10, td + 0.25, t))
    glow = cv2.resize(cv2.GaussianBlur(cv2.resize(m, (W // 4, H // 4), interpolation=cv2.INTER_AREA), (0, 0), 7), (W, H))
    shadow = cv2.resize(cv2.GaussianBlur(cv2.resize(m, (W // 4, H // 4), interpolation=cv2.INTER_AREA), (0, 0), 4), (W, H))
    shadow = np.roll(shadow, (12, 8), (0, 1)) * (1 - winB)
    f = A_ * dim * (1 - 0.6 * shadow[..., None])
    f = f + glow[..., None] * GOLD_GLOW * (0.55 * (1 - winB) + 0.25) * (1 - k)
    f = f * (1 - m[..., None]) + (fill + sweep[..., None] * 0.9) * m[..., None]
    f = f + rimw[..., None] * np.float32([0.75, 0.95, 1.0]) * 0.9
    return f


# ---- the cortex: the edit was on STAR Labs' screens all along (loop)
def _cortex_assets():
    if 'cortex' in _K4: return _K4['cortex']
    n = len(SHOTS); Sx, Sy = 1.13 * W, 1.10 * H
    screens = []
    for r in range(-2, 3):
        for j in range(-2, 4):
            if (j, r) == (0, 0): k = n - 1
            elif (j, r) == (1, -1): k = 0
            else: k = 1 + ((j + 2) * 5 + (r + 2) * 3) % (n - 2)
            screens.append(dict(k=k, X=j * Sx, Y=r * Sy, d=math.hypot(j, r), off=((j * 7 + r * 3) % 5) * 0.11))
    rng = np.random.default_rng(3); bg = np.zeros((H, W, 3), np.float32)
    g = np.exp(-((YY - 0.45 * H) / (0.9 * H)) ** 2)[..., None]
    bg = bg + np.float32([0.10, 0.04, 0.015]) + g * np.float32([0.16, 0.07, 0.02])
    sl = (0.5 + 0.5 * np.cos(YY * 2 * math.pi / 3.0)).astype(np.float32)
    _K4['cortex'] = dict(screens=screens, bg=bg, Sx=Sx, Sy=Sy, scan=sl)
    return _K4['cortex']


def t_cortex(t, A, a, end):
    """Loop: the last shot is a screen -- the camera pulls back and it becomes one monitor of the STAR Labs cortex
    wall, the screens around it flickering on one after another with every other moment of the edit, cool light
    spilling into the haze; the camera glides diagonally across the wall to the screen showing the first shot, a gold
    trace runs around its bezel, and it pushes in until that screen is the frame -- the start of the edit."""
    if t >= end: return layer(0, t - DUR)
    C = _cortex_assets(); n = len(SHOTS); fpx = 1.15 * W
    u = clamp((t - a) / (end - a)); S5 = lambda x: x * x * x * (x * (6 * x - 15) + 10)
    w = S5(ss(0.18, 0.88, u))
    s0 = [s for s in C['screens'] if s['k'] == 0][0]; sL = [s for s in C['screens'] if (s['X'], s['Y']) == (0, 0)][0]
    camx, camy = lerp(sL['X'], s0['X'], w), lerp(sL['Y'], s0['Y'], w)
    D = fpx * (1 + 1.75 * math.sin(math.pi * S5(u)) ** 2)
    roll = 2.5 * math.sin(math.pi * S5(u)) * math.sin(math.pi * w + 0.6)
    f = C['bg'].copy()
    spill = np.zeros((H // 4, W // 4), np.float32)
    lines = []
    for s in C['screens']:
        sc = fpx / D; cxs = W / 2 + (s['X'] - camx) * sc; cys = H / 2 + (s['Y'] - camy) * sc
        hw, hh = W / 2 * sc, H / 2 * sc
        if cxs + hw * 1.1 < 0 or cxs - hw * 1.1 > W or cys + hh * 1.1 < 0 or cys - hh * 1.1 > H: continue
        live_end = s is sL; live_0 = s is s0
        on_t = a + 0.10 + 0.07 * s['d'] + 0.05 * s['off']
        pw = 1.0 if live_end or live_0 else ss(on_t, on_t + 0.10, t)
        bez = 0.035 * W * sc
        x0, y0, x1, y1 = cxs - hw - bez, cys - hh - bez, cxs + hw + bez, cys + hh + bez
        cv2.rectangle(f, (int(x0), int(y0)), (int(x1), int(y1)), (0.035, 0.03, 0.03), -1)
        cv2.rectangle(f, (int(x0), int(y0)), (int(x1), int(y1)), (0.20, 0.17, 0.15), max(1, int(2 * sc)), cv2.LINE_AA)
        if pw <= 0.001: continue
        if live_end: img = layer(n - 1, t)
        elif live_0: img = layer(0, t - DUR)
        else:
            sh = SHOTS[s['k']]; img = layer(s['k'], sh['start'] + 0.2 + s['off'] + (t - a))
        open_v = ss(0, 0.6, pw); ohh = max(1.0, hh * open_v)
        Ms = np.float32([[2 * hw / W, 0, cxs - hw], [0, 2 * ohh / H, cys - ohh]])
        full = clamp((sc - 0.82) / 0.18)
        pim = cv2.warpAffine(img, Ms, (W, H), flags=cv2.INTER_AREA if sc < 0.7 else cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        pm = cv2.warpAffine(np.ones((H // 4, W // 4), np.float32), np.float32([[2 * hw / W * 4, 0, cxs - hw], [0, 2 * ohh / H * 4, cys - ohh]]), (W, H), flags=cv2.INTER_LINEAR)
        look = 1 - full
        tint = pim * (1 - 0.18 * look) + look * 0.18 * pim.mean(2, keepdims=True) * np.float32([1.25, 1.0, 0.75])
        scan = C['scan'] if sc < 0.55 else 1.0
        tint = tint * (1 - 0.12 * look * (1 - (scan if np.ndim(scan) == 0 else scan[..., None])))
        flick = 1 + 1.5 * (1 - ss(on_t + 0.03, on_t + 0.16, t)) * (not (live_end or live_0))
        sheen = np.exp(-(((XX - cxs) - (YY - cys) * 0.6 + hw * 0.4) / (0.35 * hw + 1)) ** 2) * 0.10 * look
        f = f * (1 - pm[..., None]) + (tint * flick + sheen[..., None]) * pm[..., None]
        spill += cv2.resize(pm, (W // 4, H // 4), interpolation=cv2.INTER_AREA) * float(np.clip(pim.mean(), 0, 1)) * look * pw
        if live_0: lines.append((cxs, cys, hw, hh, bez))
    sp = cv2.resize(cv2.GaussianBlur(spill, (0, 0), 14), (W, H))
    f = f + sp[..., None] * np.float32([0.95, 0.70, 0.40]) * 0.9
    for cxs, cys, hw, hh, bez in lines:                          # a gold trace runs once around the first shot's bezel
        p = clamp((u - 0.45) / 0.30); fade = 1 - ss(0.80, 0.97, u)
        if p > 0 and fade > 0:
            per = [(cxs - hw - bez / 2, cys - hh - bez / 2), (cxs + hw + bez / 2, cys - hh - bez / 2), (cxs + hw + bez / 2, cys + hh + bez / 2), (cxs - hw - bez / 2, cys + hh + bez / 2), (cxs - hw - bez / 2, cys - hh - bez / 2)]
            seg = np.array(per, np.float32); Ls = np.r_[0, np.cumsum(np.hypot(*np.diff(seg, axis=0).T))]
            s_ = np.linspace(0, Ls[-1] * p, 80); pts = np.stack([np.interp(s_, Ls, seg[:, 0]), np.interp(s_, Ls, seg[:, 1])], 1)
            tr = np.zeros((H, W), np.float32); cv2.polylines(tr, [np.int32(pts * 4)], False, 1.0, max(2, int(bez * 0.35)), cv2.LINE_AA, shift=2)
            hd = np.zeros((H, W), np.float32); cv2.circle(hd, (int(pts[-1][0]), int(pts[-1][1])), max(3, int(bez * 0.6)), 1.0, -1, cv2.LINE_AA)
            f = f + (tr[..., None] * np.float32([0.55, 0.85, 1.0]) + cv2.GaussianBlur(tr + 2 * hd, (0, 0), 10)[..., None] * GOLD_GLOW * 1.6) * fade
    haze = np.exp(-((YY - H * 0.95) / (0.3 * H)) ** 2)[..., None] * np.float32([0.10, 0.05, 0.02]) * (1 - clamp((fpx / D - 0.82) / 0.18))
    f = f + haze
    if abs(roll) > 0.02:
        M = cv2.getRotationMatrix2D((W / 2, H / 2), roll, 1.0 + abs(roll) * 0.012)
        f = cv2.warpAffine(f, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return f


# ==== Flash kit XIV (Remedy v2): beat-synced overlays, stutter, shear, anamorphic flare, collage, cortex surge ====
def slabhit_fx(f, t, s):
    """A thin slab of light streaks across the top or bottom band (never the face line) on every kick and snare."""
    for th in sorted(set(_hits('KICKS')) | set(_hits('SNARES'))):
        if not (0 <= t - th < 0.24) or not (s['start'] - 0.05 <= th < s['end']): continue
        k = int(round((th - GRID0) / HALF))
        y0 = (0.085 if k % 2 == 0 else 0.905) * H
        hx = (W + 60 - (t - th) * 2.7 * W / 0.20) if k % 4 < 2 else (-60 + (t - th) * 2.7 * W / 0.20)
        if k % 4 < 2: f = _slab(f, t, y0, 0.038 * H, hx, 1.3 * W, ('cyan', 'gold', 'red', 'white')[k % 4], 300 * k)
        else:                                                    # left to right: mirror the frame, draw, mirror back
            f = np.ascontiguousarray(_slab(np.ascontiguousarray(f[:, ::-1]), t, y0, 0.038 * H, W - hx, 1.3 * W, ('cyan', 'gold', 'red', 'white')[k % 4], 300 * k)[:, ::-1])
    return f


GAPS = (11, 15, 19, 27, 31, 35)                                  # the 808's silent steps


def gapsplit_fx(f, t, s):
    """On the 808's silent step the picture breaks into three slanted slabs that slide apart (lit seams, the middle
    one pushed in), hold and drift through the silence, and snap back together on the next hit."""
    for gs in GAPS:
        tg = GRID0 + gs * HALF
        if not (s['start'] + 0.1 <= tg < s['end'] - 0.1) or not (tg <= t < tg + HALF): continue
        u = (t - tg) / HALF
        hold = s.get('out') == 'shear' and tg + HALF >= s['end'] - 0.01    # the shear carries the split on
        amp = eout(clamp(u / 0.14)) * (1.0 if hold else 1 - ein(clamp((u - 0.84) / 0.16)))
        if amp < 0.01: return f
        v = YY + (XX - W / 2) * _TILT
        b1, b2 = 0.36 * H, 0.66 * H
        m1 = ss(b1 - 1.5, b1 + 1.5, v); m2 = ss(b2 - 1.5, b2 + 1.5, v)
        dxs = (-0.055 * W * amp - 0.01 * W * u, 0.045 * W * amp + 0.01 * W * u, -0.05 * W * amp - 0.012 * W * u)
        bands = [_shift(f, dxs[0], 0, 1.0 + 0.02 * amp, W / 2, 0.42 * H), _shift(f, dxs[1], 0, 1.0 + 0.07 * amp, W / 2, 0.42 * H), _shift(f, dxs[2], 0, 1.0 + 0.02 * amp, W / 2, 0.42 * H)]
        out = bands[0] * (1 - m1[..., None]) + bands[1] * (m1 - m2)[..., None] + bands[2] * m2[..., None]
        seam = np.exp(-((v - b1) / 2.2) ** 2) + np.exp(-((v - b2) / 2.2) ** 2)
        gl = cv2.resize(cv2.GaussianBlur(cv2.resize(seam, (W // 4, H // 4), interpolation=cv2.INTER_AREA), (0, 0), 5), (W, H))
        return out * (1 - 0.10 * amp) + (seam[..., None] * 0.9 + gl[..., None] * SLAB_COL['gold'] * 2.0) * amp
    return f


def emblempulse_fx(f, i, t, s):
    """On every big kick inside the shot the real chest emblem (found per frame as the brightest chest blob) flares and
    throws a ring of light that expands and fades."""
    hits = [th for th in _hits('BUMP_KICKS') + _hits('SNARES') if s['start'] - 0.02 <= th <= t and t - th < 0.40]
    if not hits: return f
    k = int(round((PRE + t - s['start']) * FPS)) // 4; key = ('embp', i, k)
    if key not in _K4: _K4[key] = _emblem_low(shot(i, s['start'] + (k * 4) / FPS - PRE))
    ex, ey, Re = _K4[key]; Re = min(max(Re, 0.05 * W), 0.16 * W)
    add_ = np.zeros((H, W), np.float32); ring = np.zeros((H, W), np.float32)
    for th in hits:
        u = (t - th) / 0.40; R = Re * (1.0 + 2.6 * eout(u)); a = (1 - u) ** 1.5
        cv2.circle(ring, (int(ex), int(ey)), int(R), float(a), max(2, int(5 * (1 - u))), cv2.LINE_AA)
        add_ += a * np.exp(-((XX - ex) ** 2 + (YY - ey) ** 2) / (2 * (0.8 * Re) ** 2)) * math.exp(-(t - th) / 0.10)
    gl = cv2.resize(cv2.GaussianBlur(cv2.resize(ring, (W // 4, H // 4), interpolation=cv2.INTER_AREA), (0, 0), 4), (W, H))
    return f + ring[..., None] * np.float32([0.85, 0.95, 1.0]) + gl[..., None] * GOLD_GLOW * 2.2 + add_[..., None] * np.float32([0.6, 0.85, 1.0]) * 0.8


def t_stutter(t, c, A, B, a, b):
    """Flash frames: on the 16ths before the cut the next shot strobes in for two frames at a time (pushed in, colour
    split), the outgoing shot pushing in between; on the beat it lands with a short settle."""
    cx, cy = W / 2, 0.42 * H
    if t < c:
        u = clamp((t - a) / (c - a))
        for tf in (c - 0.24, c - 0.12, c - 0.06):
            if tf <= t < tf + 0.034:
                im = _rgb_zoom(layer(B, t), (1.27, 1.25, 1.23), cx, cy)
                Ai = layer(A, t, 1 + 0.08 * u)
                im = im * float(np.clip((Ai.mean() * 1.12 + 0.02) / (im.mean() + 1e-3), 0.5, 1.2))
                return 0.25 * Ai + 0.75 * im
        return _rgb_zoom(layer(A, t), (1 + 0.08 * u - 0.006 * u, 1 + 0.08 * u, 1 + 0.08 * u + 0.006 * u), cx, cy)
    v = clamp((t - c) / (b - c)); e = eout(v); z = lerp(1.16, 1.0, e); sp = 0.012 * (1 - e)
    return _rgb_zoom(layer(B, t), (z * (1 + sp), z, z * (1 - sp)), cx, cy) + 0.08 * math.exp(-(t - c) * 25)


def t_shear(t, c, A, B, a, b):
    """The frame shears along two slanted seams (the slab angle): the three bands slide away in alternating directions
    one after another, each carrying the next shot in behind it with speed blur, the seams glowing white and gold."""
    v = YY + (XX - W / 2) * _TILT
    b1, b2 = 0.36 * H, 0.66 * H
    m = [1 - ss(b1 - 1.5, b1 + 1.5, v), ss(b1 - 1.5, b1 + 1.5, v) - ss(b2 - 1.5, b2 + 1.5, v), ss(b2 - 1.5, b2 + 1.5, v)]
    out = np.zeros((H, W, 3), np.float32); A_ = layer(A, t) if t < c + 0.16 else None; B_ = layer(B, t)
    seams = 0.0
    for k in range(3):
        s0 = c - 0.12 + 0.06 * k; d = (-1, 1, -1)[k]
        p = clamp((t - s0) / 0.13); e = eout(p) if p > 0 else 0.0
        if p <= 0:
            pre = ein(clamp((t - a) / (s0 - a))) * 0.02 * W * d
            im = _shift(A_, pre)
        elif p >= 1: im = B_
        else:
            offA = d * W * e; offB = -d * W * (1 - e)
            Ai = _hblur(_shift(A_, offA), 1 + 120 * (1 - abs(2 * e - 1))) if A_ is not None else B_
            Bi = _hblur(_shift(B_, offB), 1 + 120 * (1 - e))
            mb = ss(-2, 2, XX - offB) if d < 0 else 1 - ss(-2, 2, XX - (W + offB))   # B's edge travels with it
            im = Ai * (1 - mb[..., None]) + Bi * mb[..., None]
        out += im * m[k][..., None]
    sl = np.exp(-((v - b1) / 2.0) ** 2) + np.exp(-((v - b2) / 2.0) ** 2)
    gl = cv2.resize(cv2.GaussianBlur(cv2.resize(sl, (W // 4, H // 4), interpolation=cv2.INTER_AREA), (0, 0), 6), (W, H))
    amp = math.sin(math.pi * clamp((t - a) / (b - a)))
    return out + (sl[..., None] + gl[..., None] * SLAB_COL['gold'] * 2.4) * amp + 0.12 * math.exp(-abs(t - c) * 30)


def t_flare(t, c, A, B, a, b):
    """Cinematic anamorphic flare: a blue-white light source drifts across below the face line, its long horizontal
    streak and lens ghosts brightening; at the cut it blooms over the frame and the next shot comes out of the glare."""
    u = clamp((t - a) / (b - a)); px = lerp(1.15 * W, -0.15 * W, u); py = 0.60 * H
    blo = math.exp(-((t - c) / 0.07) ** 2)
    f = layer(A, t) if t < c else layer(B, t)
    if t >= c: f = f * (1 + 0.15 * math.exp(-(t - c) / 0.10))
    inten = 0.5 + 1.1 * math.exp(-((t - c) / 0.14) ** 2)
    dy = YY - py; dx = XX - px
    streak = np.exp(-(dy / 5.0) ** 2) * np.exp(-np.abs(dx) / (0.9 * W)) * 1.2 + np.exp(-(dy / 26.0) ** 2) * np.exp(-np.abs(dx) / (0.55 * W)) * 0.6
    core = np.exp(-(dx ** 2 + dy ** 2) / (2 * 22.0 ** 2)) * 2.0 + np.exp(-(dx ** 2 + dy ** 2) / (2 * 120.0 ** 2)) * 0.5
    flare = streak[..., None] * np.float32([1.0, 0.72, 0.38]) + core[..., None] * np.float32([1.0, 0.95, 0.9])
    for k, (sc, r, col) in enumerate(((-0.6, 70, (1.0, 0.6, 0.3)), (-1.1, 34, (0.4, 0.9, 1.0)), (0.45, 50, (0.9, 0.5, 1.0)))):
        gx, gy = W / 2 + (px - W / 2) * sc, H / 2 + (py - H / 2) * sc
        disc = np.clip(1 - np.sqrt((XX - gx) ** 2 + (YY - gy) ** 2) / r, 0, 1) ** 0.6
        flare = flare + disc[..., None] * np.float32(col) * 0.10
    return f + flare * inten * 0.45 + blo * 0.18


def _cover(img, x0, y0, x1, y1, fy=0.40):
    """Map a full frame into the rectangle (cover-crop, the face line kept in view): returns image and mask."""
    pw, ph = max(2.0, x1 - x0), max(2.0, y1 - y0); s = max(pw / W, ph / H)
    ty = min(y0, max(y1 - s * H, (y0 + y1) / 2 - s * fy * H))
    M = np.float32([[s, 0, (x0 + x1) / 2 - s * W / 2], [0, s, ty]])
    im = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_AREA if s < 0.6 else cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    m = np.zeros((H, W), np.float32); m[int(max(0, y0)):int(min(H, y1)), int(max(0, x0)):int(min(W, x1))] = 1
    return im, m


def t_collage(t, c, A, B, a, b):
    """The reference's collage frame: the shot shrinks into a panel and four more panels pop in on the 16ths (other
    moments of the edit, live), white-bordered on a dark, blurred ground; on the beat the next shot's panel takes the
    frame while the rest slide away."""
    L = [(0.04, 0.05, 0.60, 0.44, A), (0.64, 0.05, 0.96, 0.27, 2), (0.64, 0.30, 0.96, 0.58, 7), (0.04, 0.48, 0.60, 0.95, B), (0.64, 0.61, 0.96, 0.95, 12)]
    pops = (a, c - 0.24, c - 0.18, c - 0.12, c - 0.06)
    if t < c:
        u = clamp((t - a) / 0.14); e = eout(u)
        bg = cv2.GaussianBlur(cv2.resize(layer(A, t), (W // 8, H // 8), interpolation=cv2.INTER_AREA), (0, 0), 3)
        f = cv2.resize(bg, (W, H)) * lerp(1.0, 0.30, e)
    v = clamp((t - c) / 0.18); ev = eout(v)
    if t >= c:
        bg = cv2.GaussianBlur(cv2.resize(layer(B, t), (W // 8, H // 8), interpolation=cv2.INTER_AREA), (0, 0), 3)
        f = cv2.resize(bg, (W, H)) * lerp(0.30, 1.0, ev)
    order = [0, 1, 2, 4, 3]
    for j in order:
        x0, y0, x1, y1, sh = L[j]
        if t < pops[j]: continue
        if j == 0:
            if t >= c: continue
            r = (lerp(0, x0, e) * W, lerp(0, y0, e) * H, lerp(1, x1, e) * W, lerp(1, y1, e) * H)
            img = layer(A, t)
        elif j == 3:
            r0 = (x0 * W, y0 * H, x1 * W, y1 * H)
            r = tuple(lerp(r0[q], (0, 0, W, H)[q], ev) for q in range(4)) if t >= c else r0
            img = layer(B, t)
        else:
            if t >= c:
                dxo = (1.2 * W) * ev; r = (x0 * W + dxo, y0 * H, x1 * W + dxo, y1 * H)
            else: r = (x0 * W, y0 * H, x1 * W, y1 * H)
            sh_ = SHOTS[sh]; img = layer(sh, sh_['start'] + 0.25 + (t - pops[j]))
        pp = eout(clamp((t - pops[j]) / 0.08)) if j else 1.0
        if pp < 1:
            cxr, cyr = (r[0] + r[2]) / 2, (r[1] + r[3]) / 2; sc = lerp(0.80, 1.0, pp)
            r = (cxr + (r[0] - cxr) * sc, cyr + (r[1] - cyr) * sc, cxr + (r[2] - cxr) * sc, cyr + (r[3] - cyr) * sc)
        im, m = _cover(img, *r)
        if f is None: f = np.zeros((H, W, 3), np.float32)
        if j and pp < 1: im = im + 0.5 * (1 - pp)
        bw = 5
        mo = np.zeros((H, W), np.float32); mo[int(max(0, r[1] - bw)):int(min(H, r[3] + bw)), int(max(0, r[0] - bw)):int(min(W, r[2] + bw))] = 1
        full = (j == 3 and ev > 0.98)
        if not full:
            sh_m = cv2.resize(cv2.GaussianBlur(cv2.resize(mo, (W // 4, H // 4), interpolation=cv2.INTER_AREA), (0, 0), 5), (W, H))
            f = f * (1 - 0.6 * np.roll(sh_m, (10, 10), (0, 1))[..., None])
            f = f * (1 - mo[..., None]) + np.float32([0.95, 0.96, 1.0]) * mo[..., None] * (1 - ev * (j == 3))
        f = f * (1 - m[..., None]) + im * m[..., None]
    return f + 0.10 * math.exp(-abs(t - c) * 28) * (t >= c)


# ---- the cortex surge (loop)
def t_cortex2(t, A, a, end):
    """Loop: the camera hauls back and the last shot becomes the centre monitor of STAR Labs' curved cortex wall (the
    camera swinging round the curve, every screen playing another moment of the edit); a surge of Speed Force
    lightning jumps from screen to screen and each one it reaches flashes over to the first shot; when the whole wall
    is the first shot the camera rushes back into the centre screen and lands on the start of the edit."""
    if t >= end: return layer(0, t - DUR)
    n = len(SHOTS); fpx = 1.15 * W; Sx, Sy = 1.12 * W, 1.10 * H; Rc = 3.2 * W
    u = clamp((t - a) / (end - a)); S5 = lambda x: x * x * x * (x * (6 * x - 15) + 10)
    if 'cx2' not in _K4:
        scr = []
        for r in range(-2, 3):
            for j in range(-3, 4):
                th = j * Sx / Rc
                X, Z = Rc * math.sin(th), -Rc * (1 - math.cos(th))           # the wall curves toward the camera
                d = math.hypot(j, r * 1.2)
                k = n - 1 if (j, r) == (0, 0) else 1 + ((j + 3) * 5 + (r + 2) * 3) % (n - 2)
                scr.append(dict(j=j, r=r, X=X, Y=r * Sy, Z=Z, yaw=-th, d=d, k=k))
        _K4['cx2'] = scr
    scr = _K4['cx2']
    pull = S5(clamp(u / 0.42)); push = S5(clamp((u - 0.62) / 0.38))
    D = fpx * (1 + 2.3 * (pull - push))
    sw = math.sin(math.pi * S5(u))
    yaw = math.radians(16) * sw * math.sin(math.pi * clamp(u / 0.9)); camx = -0.9 * W * sw * math.sin(math.pi * clamp(u / 0.9)); camy = -0.10 * H * sw
    roll = 3.0 * sw
    cz = -D
    cy_, sy_ = math.cos(yaw), math.sin(yaw)
    def proj(P):
        x, y, z = P[0] - camx, P[1] - camy, P[2] - cz
        xr, zr = cy_ * x - sy_ * z, sy_ * x + cy_ * z
        return xr, y, zr
    f = np.zeros((H, W, 3), np.float32) + np.float32([0.09, 0.035, 0.012]) + np.exp(-((YY - 0.5 * H) / (0.8 * H)) ** 2)[..., None] * np.float32([0.14, 0.06, 0.02])
    tswitch = lambda s_: a + (end - a) * (0.40 + 0.055 * s_['d'])     # the surge spreads outward from the centre
    spill = np.zeros((H // 4, W // 4), np.float32); cents = {}
    vis = []
    for s_ in scr:
        hw, hh = W / 2, H / 2; ca, sa = math.cos(s_['yaw']), math.sin(s_['yaw'])
        cs = [(s_['X'] - hw * ca, s_['Y'] - hh, s_['Z'] - hw * sa), (s_['X'] + hw * ca, s_['Y'] - hh, s_['Z'] + hw * sa),
              (s_['X'] + hw * ca, s_['Y'] + hh, s_['Z'] + hw * sa), (s_['X'] - hw * ca, s_['Y'] + hh, s_['Z'] - hw * sa)]
        pc = [proj(p) for p in cs]
        if min(p[2] for p in pc) < 0.05 * fpx: continue
        q = np.float32([[W / 2 + fpx * p[0] / p[2], H / 2 + fpx * p[1] / p[2]] for p in pc])
        if q[:, 0].max() < -40 or q[:, 0].min() > W + 40 or q[:, 1].max() < -40 or q[:, 1].min() > H + 40: continue
        vis.append((sum(p[2] for p in pc), s_, q))
    vis.sort(key=lambda v_: -v_[0])
    centre = None
    for depth, s_, q in vis:
        ts = tswitch(s_); is_c = (s_['j'], s_['r']) == (0, 0)
        if t >= ts: img = layer(0, t - DUR) if is_c else layer(0, SHOTS[0]['start'] + 0.3 + (t - ts) + 0.05 * s_['d'])
        elif is_c: img = layer(n - 1, t)
        else: img = layer(s_['k'], SHOTS[s_['k']]['start'] + 0.25 + 0.1 * s_['d'] + (t - a))
        hI, wI = img.shape[:2]
        Pm = cv2.getPerspectiveTransform(np.float32([[0, 0], [wI, 0], [wI, hI], [0, hI]]), q)
        area = abs(cv2.contourArea(q)) / (W * H)
        small = area < 0.25
        if small: img = cv2.resize(img, (W // 2, H // 2), interpolation=cv2.INTER_AREA); Pm = cv2.getPerspectiveTransform(np.float32([[0, 0], [W // 2, 0], [W // 2, H // 2], [0, H // 2]]), q)
        pim = cv2.warpPerspective(img, Pm, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        bq = q.mean(0) + (q - q.mean(0)) * 1.045
        mb = np.zeros((H, W), np.float32); cv2.fillConvexPoly(mb, np.int32(bq * 4), 1.0, cv2.LINE_AA, shift=2)
        mi = np.zeros((H, W), np.float32); cv2.fillConvexPoly(mi, np.int32(q * 4), 1.0, cv2.LINE_AA, shift=2)
        full = clamp((area - 0.80) / 0.20)
        f = f * (1 - mb[..., None]) + mb[..., None] * np.float32([0.035, 0.03, 0.03])
        fl = 1 + 1.1 * math.exp(-max(0.0, t - ts) / 0.06) * (t >= ts) * (1 - full)
        look = 1 - full
        sheen = np.exp(-(((XX - q[:, 0].mean()) - (YY - q[:, 1].mean()) * 0.6) / (0.25 * (q[:, 0].max() - q[:, 0].min()) + 1)) ** 2) * 0.08 * look
        f = f * (1 - mi[..., None]) + (pim * fl + sheen[..., None]) * mi[..., None]
        rim = np.clip(mb - mi, 0, 1) * look
        f = f + rim[..., None] * np.float32([0.25, 0.20, 0.16]) * (1 + 3 * math.exp(-max(0.0, t - ts) / 0.08) * (t >= ts))
        spill += cv2.resize(mi, (W // 4, H // 4), interpolation=cv2.INTER_AREA) * float(min(1.0, pim[mi > 0.5].mean() if (mi > 0.5).any() else 0)) * look
        cents[(s_['j'], s_['r'])] = (q.mean(0), ts)
        if is_c: centre = q
    f = f + cv2.resize(cv2.GaussianBlur(spill, (0, 0), 12), (W, H))[..., None] * np.float32([0.9, 0.65, 0.38]) * 0.7
    segs = []                                                     # the surge: lightning jumps from screen to screen
    rng = np.random.default_rng(int(t * FPS))
    for (j, r), (p, ts) in cents.items():
        if (j, r) == (0, 0) or not (ts - 0.07 <= t < ts + 0.04): continue
        if abs(j) >= abs(r) and j != 0: jj, rr = j - (1 if j > 0 else -1), r
        else: jj, rr = j, r - (1 if r > 0 else -1)
        if (jj, rr) in cents:
            segs.append((bolt_path(cents[(jj, rr)][0], p, rng, 5, 0.10), 1.0))
    if segs: f = add(f, draw_bolts(segs, 4.0, amp=1.2))
    if centre is not None and push > 0:                           # rush into the centre screen
        k = math.sin(math.pi * clamp((u - 0.62) / 0.38)); cx_, cy_c = centre[:, 0].mean(), centre[:, 1].mean()
        if k > 0.02: f = _radial(f, 1.0, 1 + 0.06 * k, cx_, cy_c, 5)
    if abs(roll) > 0.02:
        M = cv2.getRotationMatrix2D((W / 2, H / 2), roll, 1.0 + abs(roll) * 0.012)
        f = cv2.warpAffine(f, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return f + 0.07 * math.exp(-((u - 0.40) / 0.04) ** 2)


# ==== Flash kit XV (Remedy v3): one of a kind -- echo whip, light burn, card flip, grey shock, strip fall, emblem lock ====
def t_echowhip(t, c, A, B, a, b):
    """A whip with speed-force afterimages: the camera whips left and the shot leaves a trail of red-to-gold echoes
    of itself; the next shot whips in from the right trailing its own echoes, which catch up and settle."""
    RED, GOLD = np.float32([0.15, 0.20, 1.0]), np.float32([0.15, 0.75, 1.0])
    if t < c:
        u = clamp((t - a) / (c - a)); dx = -0.55 * W * ein(u); src = layer(A, t); sgn = 1
    else:
        u = 1 - clamp((t - c) / (b - c)); dx = 0.55 * W * (1 - eout(1 - u)); src = layer(B, t); sgn = -1
    f = _hblur(_shift(src, dx), 1 + 140 * u ** 2)
    for k in range(1, 5):
        w = 0.75 * (1 - k / 5) * u ** 0.6
        if w < 0.01: continue
        e = _hblur(_shift(src, dx + sgn * k * 0.15 * W * u), 1 + 60 * u)
        f = f + e * lerp(RED, GOLD, k / 4) * w
    return f / (1 + 0.45 * u) + 0.06 * math.exp(-abs(t - c) * 30)


def t_burn(t, c, A, B, a, b):
    """Light burn: the brightest parts of the fiery shot (the lightning, the fire) flare up and burn through first,
    a white-hot gold edge eating outward through the darker areas, and the next shot is underneath."""
    key_ = ('burn', A, round(c, 3))
    if key_ not in _K4:
        rng = np.random.default_rng(7); n = cv2.resize(fbm(H // 16, W // 16, 4, 23, 4), (W, H), interpolation=cv2.INTER_CUBIC)
        _K4[key_] = ((n - n.min()) / (n.max() - n.min() + 1e-6)).astype(np.float32)
    nz = _K4[key_]
    p = clamp((t - a) / (b - a))
    Af = layer(A, t, 1 + 0.05 * p); Bf = layer(B, t, 1.06 - 0.06 * eout(clamp((t - c) / (b - c))))
    lum = cv2.GaussianBlur(cv2.resize(luma(Af), (W // 4, H // 4), interpolation=cv2.INTER_AREA), (0, 0), 3)
    lum = cv2.resize(lum, (W, H)); lum = lum / (lum.max() + 1e-6)
    key = 0.72 * lum + 0.28 * nz
    th = lerp(1.05, -0.08, ease(p) ** 0.9)
    m = ss(th - 0.035, th + 0.035, key)
    edge = np.exp(-((key - th) / 0.03) ** 2)
    hot = Af * (1 + 1.2 * np.exp(-((key - th) / 0.10) ** 2)[..., None])
    f = hot * (1 - m[..., None]) + Bf * m[..., None]
    gl = cv2.resize(cv2.GaussianBlur(cv2.resize(edge, (W // 4, H // 4), interpolation=cv2.INTER_AREA), (0, 0), 5), (W, H))
    return f + edge[..., None] * np.float32([0.75, 0.92, 1.0]) * 1.2 + gl[..., None] * GOLD_GLOW * 1.6


def t_flip(t, c, A, B, a, b):
    """The frame is a card: it turns on its vertical axis in perspective, a glint running down its edge as it goes
    side-on on the beat, and the next shot is on the back; the camera eases back a little during the turn."""
    if t < c: th = 90.0 * clamp((t - a) / (c - a)) ** 2; img = layer(A, t)
    else: v = clamp((t - c) / (b - c)); th = 90.0 + 90.0 * (1 - (1 - v) ** 2); img = layer(B, t)
    p = clamp((t - a) / (b - a)); s = 1 - 0.20 * math.sin(math.pi * p)
    D = 1.8 * W; ang = math.radians(th if th <= 90 else th - 180)
    cs, sn = math.cos(ang), math.sin(ang)
    pts = []
    for x, y in ((-W / 2, -H / 2), (W / 2, -H / 2), (W / 2, H / 2), (-W / 2, H / 2)):
        X, Z = x * cs, x * sn
        pts.append([W / 2 + s * D * X / (D + Z), H / 2 + s * D * y / (D + Z)])
    q = np.float32(pts)
    bg = cv2.resize(cv2.GaussianBlur(cv2.resize(img, (W // 8, H // 8), interpolation=cv2.INTER_AREA), (0, 0), 4), (W, H)) * 0.35
    Pm = cv2.getPerspectiveTransform(np.float32([[0, 0], [W, 0], [W, H], [0, H]]), q)
    pim = cv2.warpPerspective(img, Pm, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    m = np.zeros((H, W), np.float32); cv2.fillConvexPoly(m, np.int32(q * 4), 1.0, cv2.LINE_AA, shift=2)
    shade = 1 - 0.45 * abs(sn)
    f = bg * (1 - m[..., None]) + pim * m[..., None] * shade
    gl = math.exp(-((abs(th - 90)) / 22.0) ** 2)
    if gl > 0.02:
        ed = np.zeros((H, W), np.float32)
        xe = q[1] if (th <= 90) else q[0]; xe2 = q[2] if (th <= 90) else q[3]
        cv2.line(ed, (int(xe[0]), int(xe[1])), (int(xe2[0]), int(xe2[1])), 1.0, 4, cv2.LINE_AA)
        f = f + (ed[..., None] + cv2.GaussianBlur(ed, (0, 0), 18)[..., None] * GOLD_GLOW * 3) * gl
    return f


def t_greyshock(t, c, A, B, a, b):
    """Into the black-and-white shot: through the 808's silent step the colour drains out of the frame from the
    edges in, the camera creeping forward; on the beat the grey shot slams in on a shockwave -- a ring of
    displacement expanding from his face."""
    cx, cy = W / 2, 0.40 * H
    r = np.sqrt((XX - cx) ** 2 + ((YY - cy) * 0.8) ** 2) / (0.75 * H)
    if t < c:
        u = ease(clamp((t - a) / (c - a)))
        f = layer(A, t, 1 + 0.05 * u)
        g = luma(f)[..., None]
        d = np.clip(u * 1.7 - (1 - r) * 0.9, 0, 1)[..., None]
        return f * (1 - d) + g * d
    v = clamp((t - c) / (b - c)); R = 1.5 * eout(v)
    rr = np.sqrt((XX - cx) ** 2 + (YY - cy) ** 2) / H
    disp = 0.045 * H * np.exp(-((rr - R) / 0.05) ** 2) * (1 - v)
    ux, uy = (XX - cx) / (rr * H + 1e-3), (YY - cy) / (rr * H + 1e-3)
    mx = (XX - ux * disp).astype(np.float32); my = (YY - uy * disp).astype(np.float32)
    f = cv2.remap(layer(B, t, 1.08 - 0.08 * eout(v)), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    ring = np.exp(-((rr - R) / 0.012) ** 2) * (1 - v)
    return f + ring[..., None] * 0.45 + 0.10 * math.exp(-(t - c) * 25)


def t_strips(t, c, A, B, a, b):
    """Colour comes back: the grey frame breaks into vertical strips that drop away one after another from the centre
    out, with speed blur and lit edges, the colour shot behind them."""
    N = 8; sw = W / N
    Af = layer(A, t) if t < c + 0.22 else None
    Bf = layer(B, t, 1.06 - 0.06 * eout(clamp((t - (c - 0.12)) / (b - c + 0.12))))
    f = Bf.copy()
    order = sorted(range(N), key=lambda k: abs(k - (N - 1) / 2))
    for rank, k in enumerate(order):
        s0 = c - 0.14 + 0.03 * rank
        p = clamp((t - s0) / 0.14)
        if p >= 1 or Af is None: continue
        dy = 1.15 * H * ein(p) * (1 if k % 2 else -1) * (1 if rank % 3 else 1)
        x0, x1 = int(k * sw), int((k + 1) * sw)
        col = Af[:, x0:x1]
        if abs(dy) > 0.5:
            col = cv2.warpAffine(col, np.float32([[1, 0, 0], [0, 1, dy]]), (x1 - x0, H), borderMode=cv2.BORDER_CONSTANT, borderValue=(-1, -1, -1))
            kb = int(1 + 160 * p) | 1
            valid = (col[..., 0] >= 0).astype(np.float32)
            col = np.clip(col, 0, None)
            if kb > 2: col = cv2.blur(col, (1, kb)); valid = cv2.blur(valid, (1, kb))
            f[:, x0:x1] = f[:, x0:x1] * (1 - valid[..., None]) + col
        else:
            f[:, x0:x1] = col
        if 0 < p < 1:
            f[:, max(0, x0 - 1):x0 + 2] += 0.6 * (1 - p)
    return f + 0.12 * math.exp(-abs(t - c) * 30)


def _emblem_low(img):
    """The chest emblem where it may sit low in a close framing: the brightest compact, round, near-white blob in the
    lower half (skin highlights are warmer and less round)."""
    l = cv2.GaussianBlur(luma(img), (0, 0), 4); sat = img.max(2) - img.min(2)
    sc = l * (1 - np.clip(sat * 2.2, 0, 1)); sc[:int(0.50 * H)] = 0; sc[:, :int(0.15 * W)] = 0; sc[:, int(0.85 * W):] = 0
    thr = np.percentile(sc[int(0.5 * H):], 99.0); m = (sc >= thr).astype(np.uint8)
    n, lab, st, cen = cv2.connectedComponentsWithStats(m)
    if n <= 1: return _bright_blob(img)
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    return float(cen[k][0]), float(cen[k][1]), float(max(st[k, cv2.CC_STAT_WIDTH], st[k, cv2.CC_STAT_HEIGHT])) * 0.55


def t_emblemlock(t, c, A, B, a, b):
    """Emblem to emblem: the camera pushes into the real emblem on his chest as it glows; on the beat the next shot
    opens out of that same emblem -- a ring of light expanding from it -- and its own emblem is locked onto the spot,
    then the framing eases back to normal."""
    key_ = ('elock', A, B, round(c, 3))
    if key_ not in _K4: _K4[key_] = (_emblem_low(layer(A, c - 0.05)), _emblem_low(layer(B, c + 0.05)))
    (ax, ay, ra), (bx, by, rb) = _K4[key_]
    ra = min(max(ra, 0.05 * W), 0.15 * W); rb = min(max(rb, 0.05 * W), 0.15 * W)
    u = clamp((t - a) / (c - a)); zA = 1 + 0.7 * ein(u)
    if t < c - 0.05:
        f = _shift(layer(A, t), 0, 0, zA, ax, ay)
        gl = np.exp(-((XX - ax) ** 2 + (YY - ay) ** 2) / (2 * (ra * zA * 0.9) ** 2)) * ein(u) * 1.2
        return f + gl[..., None] * np.float32([0.7, 0.9, 1.0])
    v = clamp((t - c) / (b - c)); e = eout(v)
    zAe = 1 + 0.7
    sB = lerp(ra * zAe / rb, 1.0, e)
    px, py = lerp(ax, bx, e), lerp(ay, by, e)
    M = np.float32([[sB, 0, px - sB * bx], [0, sB, py - sB * by]])
    Bf = cv2.warpAffine(layer(B, t), M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    Af = _shift(layer(A, t), 0, 0, zA if t < c else zAe * (1 + 0.3 * e), ax, ay)
    Rc = max(math.hypot(ax - x_, ay - y_) for x_ in (0, W) for y_ in (0, H)) + 30
    R = lerp(ra * zAe * 0.4, Rc, eout(clamp((t - (c - 0.05)) / 0.24)))
    d = np.sqrt((XX - ax) ** 2 + (YY - ay) ** 2)
    m = 1 - ss(R - 3, R + 3, d)
    f = Af * (1 - m[..., None]) + Bf * m[..., None]
    ring = np.exp(-((d - R) / 5.0) ** 2); gl = np.exp(-((d - R) / 40.0) ** 2)
    k = 1 - ss(0.5, 1.0, clamp((t - (c - 0.05)) / 0.24))
    return f + (ring[..., None] * 0.9 + gl[..., None] * GOLD_GLOW * 0.9) * k + 0.18 * math.exp(-abs(t - c) * 25)


TRANSITIONS = {
    'cut':       (lambda c, o: (c, c), None),
    'leaf':      (lambda c, o: (c - .15, c + .15), lambda t, c, A, B, o: t_leaf(t, c, A, B, o.get('ang', -38), o.get('seed', 3))),
    'blur':      (lambda c, o: (c - .15, c + .15), lambda t, c, A, B, o: t_blur(t, c, A, B)),
    'glass':     (lambda c, o: (c - .10, c + .09), lambda t, c, A, B, o: t_glass(t, c - .10, c + .09, A, B, ang=o.get('ang', 32), rev=o.get('rev', False), bright=o.get('bright', 1.0))),
    'smoke':     (lambda c, o: (c - .198, c + .252), lambda t, c, A, B, o: t_smoke(t, c, A, B, c - .198, c + .252)),
    'dip':       (lambda c, o: (c - .097, c + .173), lambda t, c, A, B, o: t_dip(t, c, A, B, c - .097, c + .173)),
    'slidewipe': (lambda c, o: (c - .45, c + .02), lambda t, c, A, B, o: t_slidewipe(t, A, B, c - .45, c + .02)),
    'blood':     (lambda c, o: (c - .29, c + .32), lambda t, c, A, B, o: t_blood(t, A, B, c - .29, c, c + .32)),
    'spin':      (lambda c, o: (c - .17, c + .17), lambda t, c, A, B, o: t_spin(t, c, A, B)),
    'polaroid':  (lambda c, o: (c - .10, c + .56), lambda t, c, A, B, o: t_polaroid(t, c, A, B)),
    'drops':     (lambda c, o: (c - .26, c + .20), lambda t, c, A, B, o: t_drops(t, c, A, B, c - .26, c + .20)),
    'whip':      (lambda c, o: (c - o.get('dur', .20) / 2, c + o.get('dur', .20) / 2),
                  lambda t, c, A, B, o: t_whip(t, c, A, B, dur=o.get('dur', .20), d=o.get('d', 1), vertical=o.get('vertical', False), amount=o.get('amount', 0.9))),
    'red':       (lambda c, o: (c - .14, c + .38), lambda t, c, A, B, o: t_red(t, c, A, B, c - .14, c + .38)),
    'confetti':  (lambda c, o: (c - .22, c + .38), lambda t, c, A, B, o: t_confetti(t, c, A, B, c - .22, c + .38)),
    'shatter':   (lambda c, o: (c - .12, c + .34), lambda t, c, A, B, o: t_shatter(t, c, A, B, c - .12, c + .34, o.get('cx', 0.5), o.get('cy', 0.40))),
    'speedlines': (lambda c, o: (c - .16, c + .20), lambda t, c, A, B, o: t_speedlines(t, c, A, B, c - .16, c + .20, o.get('d', 1))),
    'phase':     (lambda c, o: (c - .20, c + .20), lambda t, c, A, B, o: t_phase(t, c, A, B, c - .20, c + .20)),
    'emblem':    (lambda c, o: (c - .60, c + .36), lambda t, c, A, B, o: t_emblem(t, c, A, B, c - .60, c + .36)),
    'emblem_real': (lambda c, o: (c - .80, c + .32), lambda t, c, A, B, o: t_emblem_real(t, c, A, B, c - .80, c + .32)),
    'strike':    (lambda c, o: (c - .22, c + .30), lambda t, c, A, B, o: t_strike(t, c, A, B, c - .22, c + .30)),
    'ink':       (lambda c, o: (c - .35, c + .30), lambda t, c, A, B, o: t_ink(t, c, A, B, c - .35, c + .30)),
    'slices':    (lambda c, o: (c - .14, c + .30), lambda t, c, A, B, o: t_slices(t, c, A, B, c - .14, c + .30)),
    'orange':    (lambda c, o: (c - .42, c + .42), lambda t, c, A, B, o: t_orange(t, c, A, B, c - .42, c + .42)),
    'syringe':   (lambda c, o: (c - .78, c + .45), lambda t, c, A, B, o: t_syringe(t, c, A, B, c - .78, c + .45)),
    'sunburst':  (lambda c, o: (c - .32, c + .30), lambda t, c, A, B, o: t_sunburst(t, c, A, B, c - .32, c + .30, o.get('sx', 0.80), o.get('sy', 0.13))),
    'clouds':    (lambda c, o: (c - .32, c + .32), lambda t, c, A, B, o: t_clouds(t, c, A, B, c - .32, c + .32)),
    'runby':     (lambda c, o: (c - .26, c + .26), lambda t, c, A, B, o: t_runby(t, c, A, B, c - .26, c + .26, o.get('y', 0.52))),
    'newspaper': (lambda c, o: (c - .62, c + .36), lambda t, c, A, B, o: t_newspaper(t, c, A, B, c - .62, c + .36)),
    'badge':     (lambda c, o: (c - .55, c + .30), lambda t, c, A, B, o: t_badge(t, c, A, B, c - .55, c + .30)),
    'photo':     (lambda c, o: (c - .55, c + .30), lambda t, c, A, B, o: t_photo(t, c, A, B, c - .55, c + .30)),
    'casefile':  (lambda c, o: (c - .38, c + .38), lambda t, c, A, B, o: t_casefile(t, c, A, B, c - .38, c + .38)),
    'strings':   (lambda c, o: (c - .55, c + .38), lambda t, c, A, B, o: t_strings(t, c, A, B, c - .55, c + .38, o.get('px', 0.5), o.get('py', 0.40))),
    'bag':       (lambda c, o: (c - .45, c + .38), lambda t, c, A, B, o: t_bag(t, c, A, B, c - .45, c + .38)),
    'crown':     (lambda c, o: (c - .40, c + .42), lambda t, c, A, B, o: t_crown(t, c, A, B, c - .40, c + .42, o.get('px', 0.5), o.get('py', 0.46))),
    'bloodwater': (lambda c, o: (c - .45, c + .45), lambda t, c, A, B, o: t_bloodwater(t, c, A, B, c - .45, c + .45)),
    'blinds':    (lambda c, o: (c - .36, c + .40), lambda t, c, A, B, o: t_blinds(t, c, A, B, c - .36, c + .40)),
    'kaleido':   (lambda c, o: (c - .45, c + .40), lambda t, c, A, B, o: t_kaleido(t, c, A, B, c - .45, c + .40)),
    'tape':      (lambda c, o: (c - .34, c + .30), lambda t, c, A, B, o: t_tape(t, c, A, B, c - .34, c + .30)),
    'rorschach': (lambda c, o: (c - .30, c + .45), lambda t, c, A, B, o: t_rorschach(t, c, A, B, c - .30, c + .45)),
    'mirror':    (lambda c, o: (c - .45, c + .40), lambda t, c, A, B, o: t_mirror(t, c, A, B, c - .45, c + .40)),
    'knife':     (lambda c, o: (c - .30, c + .34), lambda t, c, A, B, o: t_knife(t, c, A, B, c - .30, c + .34)),
    'scope':     (lambda c, o: (c - .40, c + .38), lambda t, c, A, B, o: t_scope(t, c, A, B, c - .40, c + .38, o.get('px', 0.5), o.get('py', 0.5))),
    'plastic':   (lambda c, o: (c - .30, c + .40), lambda t, c, A, B, o: t_plastic(t, c, A, B, c - .30, c + .40)),
    'water':     (lambda c, o: (c - .279, c + .421), lambda t, c, A, B, o: t_water(t, A, B, c - .279, c, c + .421)),
    # Dexter kit IV (X Gon' Give It To Ya)
    'sub':       (lambda c, o: (c - .55, c + .30), lambda t, c, A, B, o: t_sub(t, c, A, B, c - .55, c + .30)),
    'ducttape':  (lambda c, o: (c - .40, c + .36), lambda t, c, A, B, o: t_ducttape(t, c, A, B, c - .40, c + .36)),
    'mugshot':   (lambda c, o: (c - .26, c + .48), lambda t, c, A, B, o: t_mugshot(t, c, A, B, c - .26, c + .48)),
    'xslash':    (lambda c, o: (c - .32, c + .40), lambda t, c, A, B, o: t_xslash(t, c, A, B, c - .32, c + .40)),
    'cuffs':     (lambda c, o: (c - .42, c + .36), lambda t, c, A, B, o: t_cuffs(t, c, A, B, c - .42, c + .36)),
    'luminol':   (lambda c, o: (c - .45, c + .35), lambda t, c, A, B, o: t_luminol(t, c, A, B, c - .45, c + .35)),
    'rollup':    (lambda c, o: (c - .40, c + .14), lambda t, c, A, B, o: t_rollup(t, c, A, B, c - .40, c + .14)),
    'bulb':      (lambda c, o: (c - .32, c + .40), lambda t, c, A, B, o: t_bulb(t, c, A, B, c - .32, c + .40)),
    # Dexter kit VI (Miami night drive)
    'pass':      (lambda c, o: (c - .18, c + .18), lambda t, c, A, B, o: t_pass(t, c, A, B, c - .18, c + .18)),
    'outrun':    (lambda c, o: (c - .30, c + .36), lambda t, c, A, B, o: t_outrun(t, c, A, B, c - .30, c + .36)),
    'window':    (lambda c, o: (c - .40, c + .32), lambda t, c, A, B, o: t_window(t, c, A, B, c - .40, c + .32)),
    'streetlights': (lambda c, o: (c - .52, c + .22), lambda t, c, A, B, o: t_streetlights(t, c, A, B, c - .52, c + .22)),
    'wiper':     (lambda c, o: (c - .42, c + .24), lambda t, c, A, B, o: t_wiper(t, c, A, B, c - .42, c + .24)),
    'sunwindow': (lambda c, o: (c - .40, c + .30), lambda t, c, A, B, o: t_sunwindow(t, c, A, B, c - .40, c + .30)),
    'mirror2':   (lambda c, o: (c - .40, c + .40), lambda t, c, A, B, o: t_mirror2(t, c, A, B, c - .40, c + .40)),
    'flaps':     (lambda c, o: (c - .20, c + .25), lambda t, c, A, B, o: t_flaps(t, c, A, B, c - .20, c + .25)),
    'neontrace': (lambda c, o: (c - .50, c + .40), lambda t, c, A, B, o: t_neontrace(t, c, A, B, c - .50, c + .40)),
    'rackfocus': (lambda c, o: (c - .40, c + .36), lambda t, c, A, B, o: t_rackfocus(t, c, A, B, c - .40, c + .36)),
    'tear':      (lambda c, o: (c - .14, c + .32), lambda t, c, A, B, o: t_tear(t, c, A, B, c - .14, c + .32)),
    'tee':       (lambda c, o: (c - .40, c + .34), lambda t, c, A, B, o: t_tee(t, c, A, B, c - .40, c + .34)),
    'drain':     (lambda c, o: (c - .46, c + .40), lambda t, c, A, B, o: t_drain(t, c, A, B, c - .46, c + .40)),
    'lensdrop':  (lambda c, o: (c - .50, c + .40), lambda t, c, A, B, o: t_lensdrop(t, c, A, B, c - .50, c + .40)),
    'castoff':   (lambda c, o: (c - .30, c + .36), lambda t, c, A, B, o: t_castoff(t, c, A, B, c - .30, c + .36)),
    'bleed':     (lambda c, o: (c - .42, c + .38), lambda t, c, A, B, o: t_bleed(t, c, A, B, c - .42, c + .38)),
    'splatburst': (lambda c, o: (c - .06, c + .30), lambda t, c, A, B, o: t_splatburst(t, c, A, B, c - .06, c + .30)),
    'trophy':    (lambda c, o: (c - .62, c + .42), lambda t, c, A, B, o: t_trophy(t, c, A, B, c - .62, c + .42)),
    'filmstrip': (lambda c, o: (c - .36, c + .36), lambda t, c, A, B, o: t_filmstrip(t, c, A, B, c - .36, c + .36)),
    'lighttrails': (lambda c, o: (c - .34, c + .34), lambda t, c, A, B, o: t_lighttrails(t, c, A, B, c - .34, c + .34)),
    'panes':     (lambda c, o: (c - .10, c + .30), lambda t, c, A, B, o: t_panes(t, c, A, B, c - .10, c + .30)),
    'codepage':  (lambda c, o: (c - .40, c + .40), lambda t, c, A, B, o: t_codepage(t, c, A, B, c - .40, c + .40)),
    'eyedive':   (lambda c, o: (c - .62, c + .34), lambda t, c, A, B, o: t_eyedive(t, c, A, B, c - .62, c + .34)),
    'glove':     (lambda c, o: (c - .60, c + .34), lambda t, c, A, B, o: t_glove(t, c, A, B, c - .60, c + .34)),
    'carve':     (lambda c, o: (c - .62, c + .40), lambda t, c, A, B, o: t_carve(t, c, A, B, c - .62, c + .40)),
    # Flash kit VII (Are We Still Friends?)
    'accel':     (lambda c, o: (c - .74, c + .46), lambda t, c, A, B, o: t_accel(t, c, A, B, c - .74, c + .46)),
    'tornado':   (lambda c, o: (c - .40, c + .40), lambda t, c, A, B, o: t_tornado(t, c, A, B, c - .40, c + .40)),
    'flashtime': (lambda c, o: (c - .48, c + .32), lambda t, c, A, B, o: t_flashtime(t, c, A, B, c - .48, c + .32)),
    'ringpop':   (lambda c, o: (c - .46, c + .40), lambda t, c, A, B, o: t_ringpop(t, c, A, B, c - .46, c + .40)),
    # Flash kit VIII
    'mirage':    (lambda c, o: (c - .74, c + .46), lambda t, c, A, B, o: t_mirage(t, c, A, B, c - .74, c + .46)),
    'trailbolt': (lambda c, o: (c - .56, c + .38), lambda t, c, A, B, o: t_trailbolt(t, c, A, B, c - .56, c + .38)),
    'disperse':  (lambda c, o: (c - .48, c + .40), lambda t, c, A, B, o: t_disperse(t, c, A, B, c - .48, c + .40)),
    'remnants':  (lambda c, o: (c - .34, c + .36), lambda t, c, A, B, o: t_remnants(t, c, A, B, c - .34, c + .36)),
    'phaseslice': (lambda c, o: (c - .34, c + .34), lambda t, c, A, B, o: t_phaseslice(t, c, A, B, c - .34, c + .34)),
    # Flash kit IX
    'emblemmatch': (lambda c, o: (c - .80, c + .46), lambda t, c, A, B, o: t_emblemmatch(t, c, A, B, c - .80, c + .46)),
    'warp':      (lambda c, o: (c - .40, c + .36), lambda t, c, A, B, o: t_warp(t, c, A, B, c - .40, c + .36)),
    'flashpaper': (lambda c, o: (c - .62, c + .36), lambda t, c, A, B, o: t_newspaper(t, c, A, B, c - .62, c + .36, sprite=_flash_paper_sprite)),
    # Flash kit X
    'emblempull': (lambda c, o: (c - .70, c + .50), lambda t, c, A, B, o: t_emblempull(t, c, A, B, c - .70, c + .50)),
    'redmatch':  (lambda c, o: (c - .34, c + .40), lambda t, c, A, B, o: t_redmatch(t, c, A, B, c - .34, c + .40)),
    'futurepaper': (lambda c, o: (c - 1.20, c + .40), lambda t, c, A, B, o: t_futurepaper(t, c, A, B, c - 1.20, c + .40)),
    'boltiris':  (lambda c, o: (c - .36, c + .20), lambda t, c, A, B, o: t_boltiris(t, c, A, B, c - .36, c + .20)),
    # Flash kit XI
    'speedpass': (lambda c, o: (c - .36, c + .36), lambda t, c, A, B, o: t_speedpass(t, c, A, B, c - .36, c + .36)),
    'emblemflare': (lambda c, o: (c - .70, c + .50), lambda t, c, A, B, o: t_emblemflare(t, c, A, B, c - .70, c + .50)),
    # Flash kit XII
    'zoomthru':  (lambda c, o: (c - .70, c + .46), lambda t, c, A, B, o: t_zoomthru(t, c, A, B, c - .70, c + .46)),
    'speedramp': (lambda c, o: (c - .60, c + .36), lambda t, c, A, B, o: t_speedramp(t, c, A, B, c - .60, c + .36)),
    # Flash kit XIII (Remedy)
    'prism':     (lambda c, o: (c - .30, c + .34), lambda t, c, A, B, o: t_prism(t, c, A, B, c - .30, c + .34)),
    'slab':      (lambda c, o: (c - .12, c + .14), lambda t, c, A, B, o: t_slab(t, c, A, B, c - .12, c + .14, o.get('col', 'gold'))),
    'triptych':  (lambda c, o: (c - .50, c + .30), lambda t, c, A, B, o: t_triptych(t, c, A, B, c - .50, c + .30)),
    'chromasnap': (lambda c, o: (c - .44, c + .28), lambda t, c, A, B, o: t_chromasnap(t, c, A, B, c - .44, c + .28)),
    'title':     (lambda c, o: (c - 1.44, c + .30), lambda t, c, A, B, o: t_title(t, c, A, B, c - 1.44, c + .30)),
    # Flash kit XIV (Remedy v2)
    'stutter':   (lambda c, o: (c - .26, c + .20), lambda t, c, A, B, o: t_stutter(t, c, A, B, c - .26, c + .20)),
    'shear':     (lambda c, o: (c - .20, c + .22), lambda t, c, A, B, o: t_shear(t, c, A, B, c - .20, c + .22)),
    'flare':     (lambda c, o: (c - .22, c + .26), lambda t, c, A, B, o: t_flare(t, c, A, B, c - .22, c + .26)),
    'collage':   (lambda c, o: (c - .32, c + .34), lambda t, c, A, B, o: t_collage(t, c, A, B, c - .32, c + .34)),
    # Flash kit XV (Remedy v3)
    'echowhip':  (lambda c, o: (c - .22, c + .22), lambda t, c, A, B, o: t_echowhip(t, c, A, B, c - .22, c + .22)),
    'burn':      (lambda c, o: (c - .30, c + .20), lambda t, c, A, B, o: t_burn(t, c, A, B, c - .30, c + .20)),
    'flip':      (lambda c, o: (c - .20, c + .24), lambda t, c, A, B, o: t_flip(t, c, A, B, c - .20, c + .24)),
    'greyshock': (lambda c, o: (c - .46, c + .30), lambda t, c, A, B, o: t_greyshock(t, c, A, B, c - .46, c + .30)),
    'strips':    (lambda c, o: (c - .18, c + .22), lambda t, c, A, B, o: t_strips(t, c, A, B, c - .18, c + .22)),
    'emblemlock': (lambda c, o: (c - .30, c + .20), lambda t, c, A, B, o: t_emblemlock(t, c, A, B, c - .30, c + .20)),
    # Dexter kit V (the kill ritual)
    'saw':       (lambda c, o: (c - .42, c + .36), lambda t, c, A, B, o: t_saw(t, c, A, B, c - .42, c + .36)),
    'ziptie':    (lambda c, o: (c - .36, c + .24), lambda t, c, A, B, o: t_ziptie(t, c, A, B, c - .36, c + .24)),
    'toolroll':  (lambda c, o: (c - .46, c + .40), lambda t, c, A, B, o: t_toolroll(t, c, A, B, c - .46, c + .40)),
    'fingerprint': (lambda c, o: (c - .40, c + .42), lambda t, c, A, B, o: t_fingerprint(t, c, A, B, c - .40, c + .42)),
    'bag2':      (lambda c, o: (c - .45, c + .40), lambda t, c, A, B, o: t_bag2(t, c, A, B, c - .45, c + .40)),
    'projector': (lambda c, o: (c - .30, c + .10), lambda t, c, A, B, o: t_projector(t, c, A, B, c - .30, c + .10)),
    'click':     (lambda c, o: (c - .06, c + .10), lambda t, c, A, B, o: t_click(t, c, A, B, c - .06, c + .10)),
    'lightsup':  (lambda c, o: (c - .12, c + .25), lambda t, c, A, B, o: t_lightsup(t, c, A, B, c - .12, c + .25)),
    'flatline':  (lambda c, o: (c - .30, c + .28), lambda t, c, A, B, o: t_flatline(t, c, A, B, c - .30, c + .28)),
    'target':    (lambda c, o: (c - .52, c + .42), lambda t, c, A, B, o: t_target(t, c, A, B, c - .52, c + .42)),
    'spray':     (lambda c, o: (c - .42, c + .35), lambda t, c, A, B, o: t_spray(t, c, A, B, c - .42, c + .35)),
}


LOOPS = ('iris', 'tunnel', 'boltsplit', 'clock', 'rewind', 'slidebox', 'vinyl', 'frost', 'mirror', 'shades', 'breach', 'struck', 'whiploop', 'memories', 'memories3d', 'memories5', 'memtunnel', 'cortex', 'cortex2')


NO_MBLUR = {'futurepaper', 'stutter'}          # transitions that draw their own motion blur (sub-frame averaging would double them)
_NOMB = set()


def build_trans():
    out = []; n = len(SHOTS)
    for i, s in enumerate(SHOTS):
        name = s.get('out', 'cut')
        if name in LOOPS:
            if i != n - 1: raise ValueError('%s is a loop ending: only the last shot can use it' % name)
            continue
        if i == n - 1: raise ValueError('the last shot must use a loop ending: %s' % ', '.join(LOOPS))
        if name not in TRANSITIONS: raise ValueError('shot %d: unknown transition %r' % (i, name))
        c = CUTS[i + 1]; o = s.get('out_opts', {}); win, fn = TRANSITIONS[name]
        if fn is not None:
            a, b = win(c, o); out.append((a, b, (lambda fn, c, A, B, o: lambda t: fn(t, c, A, B, o))(fn, c, i, i + 1, o)))
            if name in NO_MBLUR: _NOMB.add((a, b))
    for i, s in enumerate(SHOTS):                       # punch-ins on the first frames of a shot
        if 'punch' in s and i > 0:
            p = s['punch']; c = CUTS[i]; d = p.get('dur', .20)
            out.append((c, c + d, (lambda c, i, p, d: lambda t: t_punch(t, c, i, dur=d, z=p.get('z', 1.22), flash=p.get('flash', 0.14)))(c, i, p, d)))
    last = n - 1; lp = SHOTS[-1].get('out')
    if lp == 'iris':
        a, fall, end = _iris_times(CUTS[-1]); out.append((a, DUR + 1, lambda t: t_iris(t, last, 0, a, fall, end)))
    elif lp == 'slidebox':
        c = CUTS[-1]; a, end = c - 0.85, c + 0.10
        out.append((a, DUR + 1, lambda t: t_slidebox(t, last, a, end)))
    elif lp == 'cortex2':
        c = CUTS[-1]; a, end = c - 1.30, c + 0.20
        out.append((a, DUR + 1, lambda t: t_cortex2(t, last, a, end)))
    elif lp == 'cortex':
        c = CUTS[-1]; a, end = c - 1.25, c + 0.20
        out.append((a, DUR + 1, lambda t: t_cortex(t, last, a, end)))
    elif lp == 'memtunnel':
        c = CUTS[-1]; a, end = c - 1.20, c + 0.22
        out.append((a, DUR + 1, lambda t: t_memtunnel(t, last, a, end)))
    elif lp == 'memories5':
        c = CUTS[-1]; a, end = c - 0.80, c + 0.22
        out.append((a, DUR + 1, lambda t: t_memories5(t, last, a, end)))
    elif lp == 'memories3d':
        c = CUTS[-1]; a, end = c - 0.80, c + 0.22
        out.append((a, DUR + 1, lambda t: t_memories3d(t, last, a, end)))
    elif lp == 'memories':
        c = CUTS[-1]; a, end = c - 0.85, c + 0.20
        out.append((a, DUR + 1, lambda t: t_memories(t, last, a, end)))
    elif lp == 'whiploop':
        c = CUTS[-1]; a, end = c - 0.70, c + 0.30
        out.append((a, DUR + 1, lambda t: t_whiploop(t, last, a, end)))
    elif lp == 'struck':
        c = CUTS[-1]; a, end = c - 0.85, c + 0.10
        out.append((a, DUR + 1, lambda t: t_struck(t, last, a, end)))
    elif lp == 'breach':
        c = CUTS[-1]; a, end = c - 0.85, c + 0.10
        out.append((a, DUR + 1, lambda t: t_breach(t, last, a, end)))
    elif lp == 'shades':
        c = CUTS[-1]; a, end = c - 0.85, c + 0.10
        out.append((a, DUR + 1, lambda t: t_shades(t, last, a, end)))
    elif lp == 'mirror':
        c = CUTS[-1]; a, end = c - 0.85, c + 0.10
        out.append((a, DUR + 1, lambda t: t_mirror(t, last, a, end)))
    elif lp == 'frost':
        c = CUTS[-1]; a, end = c - 0.90, c + 0.10
        out.append((a, DUR + 1, lambda t: t_frost(t, last, a, end)))
    elif lp == 'vinyl':
        c = CUTS[-1]; a, end = c - 0.85, c + 0.10
        out.append((a, DUR + 1, lambda t: t_vinyl(t, last, a, end)))
    elif lp == 'rewind':
        c = CUTS[-1]; a, end = c - 0.50, c + 0.10
        out.append((a, DUR + 1, lambda t: t_rewind(t, a, end)))
    elif lp == 'clock':
        c = CUTS[-1]; a, op, end = c - 0.80, c - 0.30, c + 0.12
        out.append((a, DUR + 1, lambda t: t_clock(t, last, 0, a, op, end)))
    elif lp == 'boltsplit':
        c = CUTS[-1]; a, cc, end = c - 0.40, c - 0.12, c + 0.12
        out.append((a, DUR + 1, lambda t: t_boltsplit(t, last, 0, a, cc, end)))
    elif lp == 'tunnel':
        c = CUTS[-1]; a, mid, end = c - 0.55, c - 0.22, c + 0.10
        out.append((a, DUR + 1, lambda t: t_tunnel(t, last, 0, a, mid, end)))
    out.sort(key=lambda w: w[0])
    for (a0, b0, _), (a1, b1, _) in zip(out, out[1:]):
        if a1 < b0 - 1e-3: raise ValueError('transitions overlap at %.2f-%.2f s: make the shot between them longer' % (a1, b0))
    return out


TRANS = build_trans()


def base_index(t):
    for i in range(len(SHOTS)):
        if C[i] <= t < C[i + 1]: return i
    return len(SHOTS) - 1


def frame(t):
    import config as _c, typo
    for a, b, fn in TRANS:
        if a <= t < b:
            if getattr(_c, 'MBLUR', False) and (a, b) not in _NOMB:   # 180-degree shutter: average two sub-frames
                d = 0.25 / FPS; f = 0.5 * (fn(max(a, t - d)) + fn(min(b - 1e-4, t + d)))
            else: f = fn(t)
            break
    else:
        f = layer(base_index(t), t)
    f = letterbox(f, t)
    return typo.overlay(f, t, getattr(_c, 'TEXT', None), W, H)
