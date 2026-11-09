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
    s = SHOTS[i]; f = shot(i, t, ez)
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
        else: raise ValueError('unknown fx %r on shot %d' % (name, i))
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


# Transition registry: name -> (window(c, o) -> (start, end), render(t, c, A, B, o)).
# c is the cut time into shot B, o the shot's out_opts. Windows are relative to the cut,
# so the same transition works at any tempo and position.
def _iris_times(c):   # c = end of the last shot; the drop falls, swells and reveals shot 0
    return c - 0.60, c - 0.20, c + 0.12


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
    'scope':     (lambda c, o: (c - .40, c + .38), lambda t, c, A, B, o: t_scope(t, c, A, B, c - .40, c + .38, o.get('px', 0.5), o.get('py', 0.5))),
    'plastic':   (lambda c, o: (c - .30, c + .40), lambda t, c, A, B, o: t_plastic(t, c, A, B, c - .30, c + .40)),
    'water':     (lambda c, o: (c - .279, c + .421), lambda t, c, A, B, o: t_water(t, A, B, c - .279, c, c + .421)),
}


def build_trans():
    out = []; n = len(SHOTS)
    for i, s in enumerate(SHOTS):
        name = s.get('out', 'cut')
        if name == 'iris':
            if i != n - 1: raise ValueError('iris is the loop ending: only the last shot can use it')
            continue
        if i == n - 1: raise ValueError('the last shot must use out=\'iris\' (the loop)')
        if name not in TRANSITIONS: raise ValueError('shot %d: unknown transition %r' % (i, name))
        c = CUTS[i + 1]; o = s.get('out_opts', {}); win, fn = TRANSITIONS[name]
        if fn is not None:
            a, b = win(c, o); out.append((a, b, (lambda fn, c, A, B, o: lambda t: fn(t, c, A, B, o))(fn, c, i, i + 1, o)))
    for i, s in enumerate(SHOTS):                       # punch-ins on the first frames of a shot
        if 'punch' in s and i > 0:
            p = s['punch']; c = CUTS[i]; d = p.get('dur', .20)
            out.append((c, c + d, (lambda c, i, p, d: lambda t: t_punch(t, c, i, dur=d, z=p.get('z', 1.22), flash=p.get('flash', 0.14)))(c, i, p, d)))
    a, fall, end = _iris_times(CUTS[-1]); last = n - 1
    if SHOTS[-1].get('out') == 'iris':
        out.append((a, DUR + 1, lambda t: t_iris(t, last, 0, a, fall, end)))
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
    for a, b, fn in TRANS:
        if a <= t < b: return fn(t)
    return layer(base_index(t), t)
