"""Kinetic lyric typography, drawn per frame from config.TEXT (section time).

TEXT = [dict(y=0.78, size=104, out=6.4, exit='fade'|'cut'|'blur'|'scatter',
             words=[(t_in, 'WORD', anim[, (b, g, r) colour]), ...]), ...]
A line is laid out once from all its words (centred), so later words never shift earlier ones.
Entrances:
  type     letters appear one by one, each popping in
  pop      the word punches in (scale overshoot)
  slide    the word rises in from below
  forever  letters appear slowly across a held note; tracking widens and letters breathe
  glitch   RGB-split, sliced and flickering for a quarter second, then steady
  impact   huge word slams in from 2.6x with a white flash and a colour split that settles
Exits run over the last moments before `out`: fade, cut, blur (grow + fade), scatter (letters fly apart).
White text with a black stroke and a soft shadow; keywords take their own colour (e.g. red).
"""
import math
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

FONT = '/usr/share/fonts/opentype/inter/InterDisplay-Black.otf'
WHITE = (0.96, 0.97, 0.97)
_fonts, _glyphs, _lines = {}, {}, {}


def _font(size):
    if size not in _fonts: _fonts[size] = ImageFont.truetype(FONT, size)
    return _fonts[size]


def _glyph(ch, size, col):
    """Premultiplied float RGBA sprite of one letter: fill, black stroke, soft drop shadow."""
    key = (ch, size, col)
    if key in _glyphs: return _glyphs[key]
    f = _font(size); sw = max(2, int(size * 0.075)); pad = sw + int(size * 0.25)
    l, t_, r, b = f.getbbox(ch, stroke_width=sw)
    w, h = r - l + 2 * pad, b - t_ + 2 * pad
    fill = Image.new('L', (w, h), 0); stroke = Image.new('L', (w, h), 0)
    ImageDraw.Draw(stroke).text((pad - l, pad - t_), ch, font=f, fill=255, stroke_width=sw, stroke_fill=255)
    ImageDraw.Draw(fill).text((pad - l, pad - t_), ch, font=f, fill=255)
    fa = np.asarray(fill, np.float32) / 255; sa = np.asarray(stroke, np.float32) / 255
    sh = cv2.GaussianBlur(np.roll(sa, int(size * 0.06), 0), (0, 0), size * 0.08) * 0.65
    a = np.clip(np.maximum(sa, sh), 0, 1)
    rgb = np.zeros((h, w, 3), np.float32) + fa[..., None] * np.float32(col)       # stroke and shadow are black
    spr = np.dstack([rgb, a]).astype(np.float32)
    origin = (pad - l, pad - t_)                                                    # where the pen sits in the sprite
    _glyphs[key] = (spr, origin)
    return _glyphs[key]


def _layout(li, line):
    """Pen x for every letter of every word, centred as a whole line."""
    if li in _lines: return _lines[li]
    f = _font(line['size']); space = f.getlength(' ') * 0.9
    words = []; x = 0.0
    for w in line['words']:
        txt = w[1]; xs = [x + f.getlength(txt[:j]) for j in range(len(txt))]
        words.append(xs); x += f.getlength(txt) + space
    total = x - space; asc, desc = f.getmetrics()
    _lines[li] = (words, total, asc, desc)
    return _lines[li]


def _ease_out(u): u = min(max(u, 0.0), 1.0); return 1 - (1 - u) ** 3
def _back(u, k=2.2): u = min(max(u, 0.0), 1.0); return 1 + (k + 1) * (u - 1) ** 3 + k * (u - 1) ** 2


def _put(canvas, spr, origin, px, py, scale=1.0, alpha=1.0, rot=0.0):
    """Composite a letter sprite with its pen origin at (px, py), scaled/rotated about the letter centre."""
    if alpha <= 0.003 or scale <= 0.01: return
    h, w = spr.shape[:2]; H, W = canvas.shape[:2]
    cx, cy = w / 2, h / 2
    M = cv2.getRotationMatrix2D((cx, cy), rot, scale)
    M[0, 2] += px - origin[0]; M[1, 2] += py - origin[1]
    ext = int(max(w, h) * scale * 0.75) + 4
    tx, ty = int(px - origin[0] + cx), int(py - origin[1] + cy)
    x0, y0, x1, y1 = max(0, tx - ext), max(0, ty - ext), min(W, tx + ext), min(H, ty + ext)
    if x1 <= x0 or y1 <= y0: return
    M[0, 2] -= x0; M[1, 2] -= y0
    patch = cv2.warpAffine(spr, M, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0) * alpha
    sub = canvas[y0:y1, x0:x1]
    sub[:] = sub * (1 - patch[..., 3:4]) + patch


def overlay(f, t, TEXT, W, H):
    if not TEXT: return f
    canvas = None
    for li, line in enumerate(TEXT):
        t0 = min(w[0] for w in line['words']); t_out = line['out']
        if t < t0 - 0.01 or t >= t_out: continue
        if canvas is None: canvas = np.zeros((H, W, 4), np.float32)
        size = line['size']; words, total, asc, desc = _layout(li, line)
        base_y = line['y'] * H + (asc - desc) * 0.5 - asc * 0.5          # vertical centre of caps on y
        extra = 0.0                                                        # 'forever' tracking widens the line: keep it centred
        for w in line['words']:
            if w[2] == 'forever' and t >= w[0]: extra += (len(w[1]) - 1) * 7.0 * min(t - w[0], 3.5)
        x_left = W / 2 - (total + extra) / 2
        ex = line.get('exit', 'fade'); ed = {'fade': 0.12, 'blur': 0.16, 'scatter': 0.32, 'cut': 0.0}[ex]
        eu = 0.0 if ed == 0 else min(max((t - (t_out - ed)) / ed, 0.0), 1.0)
        rng = np.random.default_rng(li * 97 + 3)
        for (w, xs) in zip(line['words'], words):
            tin, txt, kind = w[0], w[1], w[2]; col = w[3] if len(w) > 3 else WHITE
            u = t - tin
            if u < 0: continue
            n = len(txt)
            for j, ch in enumerate(txt):
                if ch == ' ': continue
                spr, org = _glyph(ch, size, tuple(col))
                px = x_left + xs[j]; py = base_y; sc = 1.0; al = 1.0; rot = 0.0
                if kind == 'type':
                    lu = (u - j * 0.045) / 0.07
                    if lu < 0: continue
                    sc = 1 + 0.5 * (1 - _ease_out(lu)); al = min(1.0, lu * 1.5)
                elif kind == 'pop':
                    sc = _back(u / 0.16) * 0.38 + 0.62 if u < 0.16 else 1.0; al = min(1.0, u / 0.05)
                elif kind == 'slide':
                    py += 54 * (1 - _ease_out(u / 0.16)); al = min(1.0, u / 0.10)
                elif kind == 'forever':
                    lu = (u - j * 0.12) / 0.18
                    if lu < 0: continue
                    al = min(1.0, lu); sc = 1 + 0.25 * (1 - _ease_out(lu))
                    px += j * 7.0 * min(u, 3.5)                                     # tracking widens over the held note
                    py += 3.5 * math.sin(t * 3.1 + j * 0.8)
                elif kind == 'glitch':
                    if u < 0.26:
                        if rng.random() < 0.18 + 0.5 * (1 - u / 0.26) * (int(t * 60) % 3 == 0): continue
                        px += rng.normal(0, 16 * (1 - u / 0.26)); py += rng.normal(0, 4 * (1 - u / 0.26))
                elif kind == 'impact':
                    sc = 1 + 1.6 * (1 - _ease_out(u / 0.11)) + 0.06 * min(u / 0.7, 1.0)
                    al = min(1.0, u / 0.04)
                    lc = W / 2; px = lc + (px - lc) * sc                            # scale the whole word about its centre
                    py = base_y + (asc * 0.5) * (sc - 1) * 0.0
                if ex == 'scatter' and eu > 0:
                    d = (j - (n - 1) / 2); px += d * 60 * eu ** 1.5; py += (rng.normal(0, 1) * 90 + 140) * eu ** 2
                    rot = rng.normal(0, 30) * eu; al *= 1 - eu
                elif ex == 'fade': al *= 1 - eu
                elif ex == 'blur': sc *= 1 + 0.18 * eu; al *= 1 - eu; px = W / 2 + (px - W / 2) * (1 + 0.18 * eu)
                _put(canvas, spr, org, px, py, sc, al, rot)
                if kind == 'impact' and u < 0.12:                                  # white flash on the letters as they land
                    _put(canvas, _glyph(ch, size, (1.0, 1.0, 1.0))[0], org, px, py, sc, (1 - u / 0.12) * 0.8)
        # colour split on glitch/impact words (whole line, decaying)
    if canvas is None: return f
    split = 0
    for line in TEXT:
        for w in line['words']:
            if w[2] in ('glitch', 'impact') and 0 <= t - w[0] < 0.3: split = max(split, int(14 * (1 - (t - w[0]) / 0.3)))
    if split:
        c2 = canvas.copy(); c2[..., 2] = np.roll(canvas[..., 2], split, 1); c2[..., 0] = np.roll(canvas[..., 0], -split, 1); canvas = c2
    return f * (1 - canvas[..., 3:4]) + canvas[..., :3]
