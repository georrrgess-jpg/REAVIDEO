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
        elif name == 'glint': f = glint_fx(f, t, s)
        elif name == 'ghost': f = ghost_fx(f, i, t, s)
        elif name == 'trails': f = trails_fx(f, i, t, s)
        elif name == 'crackle_build': f = crackle_fx(f, i, t, s, build=True)
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
    rng = np.random.default_rng(int(t * FPS / 3) + 17 * i); segs = []
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
    """Zoom bump on each strong kick inside the shot: +4.5% that settles in ~0.1 s."""
    z = 0.0
    if t < s['start'] and any(abs(tk - s['start']) < 1e-6 for tk in _hits('KICKS')):
        return 0.045                                     # loop tail of shot 0: hold the downbeat bump so the seam matches
    for tk in _hits('KICKS'):
        if s['start'] - 1e-6 <= tk <= t: z = max(z, 0.045 * math.exp(-(t - tk) / 0.09))
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


def ghost_fx(f, i, t, s):
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
        out = out + ((ed * 0.9 + glow * 1.4) * a0 * (0.45 + 0.8 * pulse))[..., None] * np.float32([0.98, 0.97, 1.0])
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
}


LOOPS = ('iris', 'tunnel', 'boltsplit', 'clock', 'rewind')


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
    for i, s in enumerate(SHOTS):                       # punch-ins on the first frames of a shot
        if 'punch' in s and i > 0:
            p = s['punch']; c = CUTS[i]; d = p.get('dur', .20)
            out.append((c, c + d, (lambda c, i, p, d: lambda t: t_punch(t, c, i, dur=d, z=p.get('z', 1.22), flash=p.get('flash', 0.14)))(c, i, p, d)))
    last = n - 1; lp = SHOTS[-1].get('out')
    if lp == 'iris':
        a, fall, end = _iris_times(CUTS[-1]); out.append((a, DUR + 1, lambda t: t_iris(t, last, 0, a, fall, end)))
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
        if a <= t < b: f = fn(t); break
    else:
        f = layer(base_index(t), t)
    return typo.overlay(f, t, getattr(_c, 'TEXT', None), W, H)
