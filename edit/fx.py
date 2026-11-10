"""Effect primitives and procedural sprites for the edit. All images are float32 BGR in 0..1."""
import math, json
import numpy as np, cv2
from config import *

class _Meta(dict):
    """clips/meta.json, loaded on first use (extract.py writes it)."""
    def __missing__(self, i):
        if not dict.__len__(self):
            self.update({int(k): v for k, v in json.load(open('clips/meta.json')).items()})
            if i in self: return self[i]
        raise KeyError('shot %s has no clip: run python extract.py %s' % (i, i))


META = _Meta()
YY, XX = np.mgrid[0:H, 0:W].astype(np.float32)


def ss(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def clamp(u): return min(max(u, 0.0), 1.0)
def ease(u): u = clamp(u); return u * u * (3 - 2 * u)
def eout(u): u = clamp(u); return 1 - (1 - u) ** 3
def ein(u): u = clamp(u); return u ** 3
def lerp(a, b, u): return a + (b - a) * u


def fbm(h, w, octaves=5, seed=1, base=4):
    r = np.random.default_rng(seed)
    out = np.zeros((h, w), np.float32); amp = 1.0; tot = 0.0
    for o in range(octaves):
        n = base * 2 ** o
        g = r.random((max(2, int(n * h / w)) + 1, n + 1)).astype(np.float32)
        out += amp * cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC); tot += amp; amp *= 0.5
    out /= tot
    return (out - out.min()) / (out.max() - out.min())


def tile_fbm(h, w, **kw):
    """fbm that tiles vertically (cross-faded) so it can be scrolled forever."""
    n = fbm(h + h // 4, w, **kw); o = h // 4
    a = np.linspace(0, 1, o, dtype=np.float32)[:, None]
    out = n[:h].copy(); out[:o] = n[:o] * a + n[h:h + o] * (1 - a)
    return out


# ---------------------------------------------------------------- clips / grade
_cache = {}


def clip_frame(i, k):
    m = META[i]; k = int(min(max(k, m['lo']), m['hi']))
    key = (i, k)
    if key not in _cache:
        if len(_cache) > 10: _cache.pop(next(iter(_cache)))
        _cache[key] = cv2.imread('clips/%02d/%04d.jpg' % (i, k))
    return _cache[key]


def grade(img, s):
    f = img.astype(np.float32) / 255.0
    b, g, r = f[..., 0], f[..., 1], f[..., 2]
    l = (0.114 * b + 0.587 * g + 0.299 * r)[..., None]
    if 'rth' in s:                                                   # strict red key: only saturated reds (blood, red light), not skin
        wr = np.clip((r - np.maximum(g, b) - 1.2 * np.maximum(g - b, 0) - s['rth']) * 6, 0, 1)[..., None]   # orange (g >> b) is not red
    else:
        wr = np.clip((r - np.maximum(g, b)) * 5, 0, 1)[..., None]      # keep reds alive in a muted grade
    se = s['sat'] + (max(s['sat'], s['rk']) - s['sat']) * wr
    if 'yk' in s:                                                    # optionally keep golds / lightning alive too
        wy = (np.clip((np.minimum(r, g) - b) * 4 - 0.15, 0, 1) * np.clip(r * 2 - 0.4, 0, 1))[..., None] * (1 - wr)
        se = se + (max(s['sat'], s['yk']) - s['sat']) * wy
    f = l + (f - l) * se
    f = (f - 0.5) * s['con'] + 0.5
    f *= np.float32(s['tint']) * s['gain']
    f += np.float32([0.022, 0.012, 0.0]) * (1 - np.clip(l, 0, 1))  # teal shadows
    return np.clip(f, 0, 1)


_luts = {}


def grade_u8(img, s):
    """grade() for 8-bit frames via a full 256^3 lookup table built once per grade (identical output, ~10x faster)."""
    key = (s['sat'], s['rk'], s['con'], tuple(s['tint']), s['gain'], s.get('yk'), s.get('rth'))
    if key not in _luts:
        if len(_luts) > 2: _luts.pop(next(iter(_luts)))
        v = np.arange(256, dtype=np.uint8); lut = np.empty((256 * 256 * 256, 3), np.uint8)
        for b in range(0, 256, 16):                      # 16 blue planes at a time keeps memory low
            cube = np.stack(np.meshgrid(v[b:b + 16], v, v, indexing='ij'), -1).reshape(16 * 256, 256, 3)
            lut[b * 65536:(b + 16) * 65536] = (grade(cube, s) * 255 + 0.5).astype(np.uint8).reshape(-1, 3)
        _luts[key] = lut
    i = img.astype(np.uint32); idx = (i[..., 0] << 16) | (i[..., 1] << 8) | i[..., 2]
    return _luts[key][idx]


def zoom(img, z, ang=0.0, cx=W / 2, cy=H * 0.42):
    M = cv2.getRotationMatrix2D((cx, cy), ang, z)
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def shot(i, t, ez=1.0):
    s = SHOTS[i]
    k = round((PRE + t - s['start']) * FPS)
    u = (t - s['start']) / (s['end'] - s['start'])
    z = max(lerp(s['push'][0], s['push'][1], u), 1.0) * ez
    c = clip_frame(i, k)
    f = c.astype(np.float32) * np.float32(1 / 255) if META[i].get('graded') else grade(c, s)
    return zoom(f, z) if abs(z - 1) > 1e-4 else f


def blur(f, r):
    if r < 0.6: return f
    d = int(min(8, max(1, r // 3)))
    sm = cv2.resize(f, (W // d, H // d), interpolation=cv2.INTER_AREA) if d > 1 else f
    sm = cv2.GaussianBlur(sm, (0, 0), r / d)
    return cv2.resize(sm, (W, H), interpolation=cv2.INTER_LINEAR) if d > 1 else sm


def luma(f): return 0.114 * f[..., 0] + 0.587 * f[..., 1] + 0.299 * f[..., 2]


def affine(src_c, dst_c, ang, sc):
    """Forward 2x3 matrix: dst = sc*R(ang)*(src-src_c)+dst_c (ang in radians, image coords)."""
    c, s = math.cos(ang) * sc, math.sin(ang) * sc
    return np.float32([[c, -s, dst_c[0] - c * src_c[0] + s * src_c[1]],
                       [s, c, dst_c[1] - s * src_c[0] - c * src_c[1]]])


def over(f, rgba):
    """Composite premultiplied RGBA layer over f."""
    return f * (1 - rgba[..., 3:4]) + rgba[..., :3]


_proj = {}


def proj(ang_deg):
    if ang_deg not in _proj:
        th = math.radians(ang_deg)
        _proj[ang_deg] = XX * math.cos(th) + YY * math.sin(th)
    return _proj[ang_deg]


# ---------------------------------------------------------------- finishing
_fin = {}


def finish(f, n):
    if not _fin:
        _fin['vig'] = np.float32(1) * (1 - 0.42 * np.clip(((XX - W / 2) / (W * 0.78)) ** 2 + ((YY - H / 2) / (H * 0.70)) ** 2, 0, 1) ** 1.25)[..., None]
        r = np.random.default_rng(99)
        _fin['grain'] = [cv2.resize(r.normal(0, 0.030, (H * 2 // 3, W * 2 // 3)).astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)[..., None] for _ in range(8)]
    sm = cv2.resize(f, (W // 8, H // 8), interpolation=cv2.INTER_AREA)
    hl = cv2.GaussianBlur(np.clip(sm - 0.62, 0, 1), (0, 0), 7)
    f = f + cv2.resize(hl, (W, H), interpolation=cv2.INTER_LINEAR) * 0.45       # soft bloom
    f *= _fin['vig']; f += _fin['grain'][n % 8]
    np.clip(f, 0, 1, out=f); f *= 255; f += 0.5
    return f.astype(np.uint8)


# ---------------------------------------------------------------- sprites
def make_frond(L=2600, Wd=1100, seed=3):
    S = 2; r = np.random.default_rng(seed)
    col = np.zeros((L * S, Wd * S, 3), np.uint8); a = np.zeros((L * S, Wd * S), np.uint8)
    rach = lambda y: Wd / 2 + 70 * math.sin(y / L * math.pi * 1.1) - 35
    n = 78
    for j in range(n):
        y = 150 + (L - 300) * j / (n - 1)
        env = math.sin(math.pi * (0.07 + 0.90 * j / (n - 1))) ** 0.6
        for side in (-1, 1):
            ln = (Wd * 0.47) * env * r.uniform(0.85, 1.05)
            ang = math.radians(r.uniform(50, 66))
            x0 = rach(y); dx = side * math.sin(ang); dy = -math.cos(ang); droop = r.uniform(0.05, 0.22)
            qs = np.linspace(0, 1, 14)
            pc = [(x0 + dx * ln * q, y + dy * ln * q + droop * ln * q * q) for q in qs]
            wmax = r.uniform(15, 22); left = []; right = []
            for k, q in enumerate(qs):
                wq = wmax * math.sin(math.pi * min(1, q * 1.05 + 0.05)) ** 0.5 * (1 - 0.8 * q)
                kk = min(k, 12); tx = pc[kk + 1][0] - pc[kk][0]; ty = pc[kk + 1][1] - pc[kk][1]
                nn = math.hypot(tx, ty); nx, ny = -ty / nn, tx / nn
                left.append((pc[k][0] + nx * wq, pc[k][1] + ny * wq)); right.append((pc[k][0] - nx * wq, pc[k][1] - ny * wq))
            poly = np.int32(np.array(left + right[::-1]) * S)
            g = r.uniform(0.70, 1.15)
            cv2.fillPoly(col, [poly], (int(26 * g), int(86 * g), int(44 * g)), lineType=cv2.LINE_AA)
            cv2.fillPoly(a, [poly], 255, lineType=cv2.LINE_AA)
            cv2.polylines(col, [np.int32(np.array(pc) * S)], False, (int(48 * g), int(128 * g), int(84 * g)), S, lineType=cv2.LINE_AA)
    pts = np.int32([[rach(y) * S, y * S] for y in np.linspace(40, L - 60, 80)])
    cv2.polylines(col, [pts], False, (36, 96, 70), 11 * S, cv2.LINE_AA); cv2.polylines(a, [pts], False, 255, 11 * S, cv2.LINE_AA)
    col = cv2.resize(col, (Wd, L), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    a = cv2.resize(a, (Wd, L), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    shade = 0.62 + 0.55 * fbm(L, Wd, octaves=4, seed=seed + 1, base=3)
    col *= shade[..., None]
    col = cv2.GaussianBlur(col, (0, 0), 2.5); a = cv2.GaussianBlur(a, (0, 0), 2.5)   # foreground defocus
    return np.dstack([col * a[..., None], a])


def make_slide(w=1040, h=340, seed=5):
    """Glass microscope slide with a blood sample and frosted label (premultiplied RGBA, 2x res) + body matte."""
    r = np.random.default_rng(seed)
    body = np.zeros((h, w), np.uint8); rad = 22
    cv2.rectangle(body, (rad, 0), (w - rad, h), 255, -1); cv2.rectangle(body, (0, rad), (w, h - rad), 255, -1)
    for c in ((rad, rad), (w - rad, rad), (rad, h - rad), (w - rad, h - rad)): cv2.circle(body, c, rad, 255, -1, cv2.LINE_AA)
    bodyf = body.astype(np.float32) / 255
    pb = cv2.copyMakeBorder(body, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=0)      # pad so the outline is even on all sides
    edge = cv2.GaussianBlur(cv2.morphologyEx(pb, cv2.MORPH_GRADIENT, np.ones((7, 7), np.uint8)).astype(np.float32) / 255, (0, 0), 1.2)[8:-8, 8:-8] * bodyf
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    a = bodyf * 0.14; col = np.ones((h, w, 3), np.float32) * np.float32([0.98, 0.96, 0.93])
    band = np.exp(-(((xx - yy * 0.9) - w * 0.42) / 60) ** 2) * 0.16 * bodyf                 # specular streak
    a = np.maximum(a, band + a)
    a = np.maximum(a, edge * 0.60)
    # frosted label
    lab = np.zeros((h, w), np.float32); cv2.rectangle(lab, (w - 330, 18), (w - 18, h - 18), 1.0, -1)
    lab = cv2.GaussianBlur(lab, (0, 0), 2)
    lcol = (0.90 + 0.08 * fbm(h, w, 4, seed, 8))[..., None] * np.float32([1.0, 0.985, 0.97])
    col = col * (1 - lab[..., None]) + lcol * lab[..., None]; a = np.maximum(a, lab * 0.90)
    # blood sample
    cx, cy, R = w * 0.33, h / 2, 122
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    circ = ss(R + 2, R - 2, d)
    sp = fbm(h, w, 4, seed + 3, 90)
    pink = np.float32([0.66, 0.60, 0.88]); red = np.float32([0.12, 0.06, 0.60])
    k = ss(0.50, 0.60, sp)[..., None]
    ccol = pink * (1 - k) + red * k
    ccol *= (0.92 + 0.10 * ss(R * 0.5, R, d))[..., None]
    col = col * (1 - circ[..., None]) + ccol * circ[..., None]; a = np.maximum(a, circ * 0.86)
    a = np.clip(a, 0, 1)
    return np.dstack([col * a[..., None], a]), bodyf


def place(sprite, M):
    return cv2.warpAffine(sprite, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
