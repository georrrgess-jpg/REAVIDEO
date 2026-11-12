"""Soundtrack: the song section the reference uses (65.643 s in) plus quiet synthesized transition SFX."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.getcwd())   # run from an edit folder: its config.py wins, engine modules come from edit/
import subprocess, math
import numpy as np, soundfile as sf, scipy.signal as sg
from config import *

SR = 48000
N = int(round(DUR * SR))
raw = subprocess.run(['ffmpeg', '-v', 'error', '-ss', '%.4f' % SONG_OFFSET, '-i', SONG, '-t', '%.4f' % (DUR + 0.2),
                      '-f', 'f32le', '-ac', '2', '-ar', str(SR), '-'], stdout=subprocess.PIPE).stdout
music = np.frombuffer(raw, np.float32).reshape(-1, 2)[:N].astype(np.float64)
music *= 0.84 / np.abs(music).max()
fx = np.zeros((N, 2))
rng = np.random.default_rng(4)


def put(sig, t, pan=0.0, gain=1.0):
    i = int(round(t * SR))
    if i < 0: sig = sig[-i:]; i = 0
    n = min(len(sig), N - i)
    if n <= 0: return
    if sig.ndim == 1:
        l = math.cos((pan + 1) * math.pi / 4); r = math.sin((pan + 1) * math.pi / 4)
        fx[i:i + n, 0] += sig[:n] * l * gain; fx[i:i + n, 1] += sig[:n] * r * gain
    else:
        fx[i:i + n] += sig[:n] * gain


def whoosh(dur, f0, f1, q=1.2, peak=0.55, sweep_pan=True):
    """Noise through a swept band-pass with a swell envelope; returns stereo."""
    n = int(dur * SR); x = rng.standard_normal(n)
    u = np.linspace(0, 1, n); fc = f0 * (f1 / f0) ** u
    lo = hp = bp = 0.0; y = np.empty(n); damp = 1 / q
    fcoef = 2 * np.sin(np.pi * np.minimum(fc, SR / 6.5) / SR)
    for i in range(n):                                   # state-variable filter
        lo += fcoef[i] * bp; hp = x[i] - lo - damp * bp; bp += fcoef[i] * hp; y[i] = bp
    env = np.where(u < peak, (u / peak) ** 2.2, ((1 - u) / (1 - peak)) ** 1.6)
    y *= env; y /= np.abs(y).max() + 1e-9
    pan = (u * 2 - 1) * 0.7 if sweep_pan else np.zeros(n)
    return np.stack([y * np.cos((pan + 1) * np.pi / 4), y * np.sin((pan + 1) * np.pi / 4)], 1)


def boom(dur=0.9, f0=95, f1=36):
    n = int(dur * SR); u = np.arange(n) / SR
    f = f1 + (f0 - f1) * np.exp(-u * 9); ph = 2 * np.pi * np.cumsum(f) / SR
    y = np.sin(ph) * np.exp(-u * 4.2)
    nz = sg.lfilter(*sg.butter(2, 500 / (SR / 2)), rng.standard_normal(n)) * np.exp(-u * 30) * 0.8
    y = y + nz; return y / np.abs(y).max()


def shutter():
    n = int(0.16 * SR); y = np.zeros(n)
    for t0, d, g in [(0.0, 0.006, 1.0), (0.052, 0.005, 0.7), (0.075, 0.012, 0.35)]:
        i = int(t0 * SR); m = int(d * SR)
        y[i:i + m] += rng.standard_normal(m) * np.exp(-np.arange(m) / (m / 3)) * g
    y = sg.lfilter(*sg.butter(2, [1500 / (SR / 2), 9000 / (SR / 2)], 'band'), y)
    return y / np.abs(y).max()


def splat(dur=0.22):
    n = int(dur * SR); u = np.arange(n) / SR
    y = sg.lfilter(*sg.butter(2, 1100 / (SR / 2)), rng.standard_normal(n)) * np.exp(-u * 26)
    for _ in range(14):
        i = rng.integers(0, n - 300); y[i:i + 200] += rng.standard_normal(200) * np.exp(-np.arange(200) / 40) * rng.uniform(0.05, 0.25)
    return y / np.abs(y).max()


def bubbles(dur=0.7, count=34):
    n = int(dur * SR); y = np.zeros(n)
    for _ in range(count):
        d = rng.uniform(0.025, 0.07); m = int(d * SR); i = rng.integers(0, n - m)
        f0 = rng.uniform(280, 800); u = np.arange(m) / m
        y[i:i + m] += np.sin(2 * np.pi * np.cumsum(f0 * (1 + 0.9 * u)) / SR) * np.sin(np.pi * u) ** 2 * rng.uniform(0.3, 1.0)
    low = sg.lfilter(*sg.butter(2, 420 / (SR / 2)), rng.standard_normal(n)) * np.sin(np.pi * np.linspace(0, 1, n)) ** 1.5
    y = y + 1.6 * low; return y / np.abs(y).max()


def shimmer(dur=0.7):
    n = int(dur * SR); u = np.arange(n) / SR; y = np.zeros(n)
    for _ in range(16):
        t0 = rng.uniform(0, dur * 0.8); f = rng.uniform(3200, 8200)
        y += np.where(u > t0, np.sin(2 * np.pi * f * (u - t0)) * np.exp(-(u - t0) * 14), 0) * rng.uniform(0.3, 1)
    y *= np.linspace(0.2, 1, n); return y / np.abs(y).max()


def glass(dur=0.9):
    """Glass crash: bright noise burst plus scattered high tinkles."""
    n = int(dur * SR); u = np.arange(n) / SR
    y = sg.lfilter(*sg.butter(2, 2500 / (SR / 2), 'high'), rng.standard_normal(n)) * np.exp(-u * 9)
    for _ in range(26):
        t0 = rng.uniform(0.02, dur * 0.8); f = rng.uniform(3000, 9000); m = u >= t0
        y[m] += np.sin(2 * np.pi * f * (u[m] - t0)) * np.exp(-(u[m] - t0) * rng.uniform(25, 60)) * rng.uniform(0.2, 0.7)
    return y / np.abs(y).max()


def crinkle(dur=0.7):
    """Plastic sheet: band-passed noise gated by sparse crackles."""
    n = int(dur * SR)
    y = sg.lfilter(*sg.butter(2, [900 / (SR / 2), 6500 / (SR / 2)], 'band'), rng.standard_normal(n))
    gate = np.zeros(n)
    for _ in range(90):
        i = rng.integers(0, n - 400); gate[i:i + 400] += np.exp(-np.arange(400) / rng.uniform(30, 120)) * rng.uniform(0.3, 1)
    y = y * (0.15 + gate) * np.sin(np.pi * np.linspace(0, 1, n)) ** 0.7
    return y / np.abs(y).max()


def shing(dur=0.9):
    """Knife draw: rising metallic scrape into a ringing tail."""
    n = int(dur * SR); u = np.arange(n) / SR
    scrape = sg.lfilter(*sg.butter(2, [2500 / (SR / 2), 9000 / (SR / 2)], 'band'), rng.standard_normal(n)) * np.exp(-((u - 0.10) / 0.07) ** 2)
    ring = sum(np.sin(2 * np.pi * f * u + p) * a for f, a, p in ((2430, 1.0, 0), (3710, 0.6, 1), (5890, 0.4, 2), (8120, 0.25, 3)))
    ring = ring * np.where(u < 0.12, (u / 0.12) ** 2, np.exp(-(u - 0.12) * 5.5))
    y = 0.8 * scrape + 0.35 * ring
    return y / np.abs(y).max()


def zap(dur=0.35):
    """Electric crackle: dense clicks through a resonant band, fast decay."""
    n = int(dur * SR); u = np.arange(n) / SR; y = np.zeros(n)
    for _ in range(int(dur * 260)):
        i = rng.integers(0, n - 200); y[i:i + 200] += rng.standard_normal(200) * np.exp(-np.arange(200) / rng.uniform(8, 40)) * rng.uniform(0.2, 1)
    y = sg.lfilter(*sg.butter(2, [1200 / (SR / 2), 9500 / (SR / 2)], 'band'), y) * np.exp(-u * 5)
    y += 0.25 * np.sign(np.sin(2 * np.pi * 120 * u)) * np.exp(-u * 9)          # mains hum bite
    return y / np.abs(y).max()


def thunder(dur=1.2):
    """Lightning strike: sharp crack then low rumble."""
    n = int(dur * SR); u = np.arange(n) / SR
    crack = sg.lfilter(*sg.butter(2, 1500 / (SR / 2), 'high'), rng.standard_normal(n)) * np.exp(-u * 30)
    rumble = sg.lfilter(*sg.butter(2, 180 / (SR / 2)), rng.standard_normal(n)) * np.exp(-u * 2.8) * 6
    y = crack + rumble; return y / np.abs(y).max()


def buzz(dur=0.45):
    """Speed Force vibration: noise amplitude-modulated at ~55 Hz with a pitch wobble."""
    n = int(dur * SR); u = np.arange(n) / SR
    y = sg.lfilter(*sg.butter(2, [300 / (SR / 2), 4000 / (SR / 2)], 'band'), rng.standard_normal(n))
    y *= (0.5 + 0.5 * np.sin(2 * np.pi * 55 * u)) * np.sin(np.pi * u / dur) ** 0.8
    y += 0.4 * np.sin(2 * np.pi * np.cumsum(90 + 40 * np.sin(2 * np.pi * 7 * u)) / SR) * np.sin(np.pi * u / dur)
    return y / np.abs(y).max()


def tick(dur=0.08, f0=2400):
    """Soft glassy tick for text entrances."""
    n = int(dur * SR); u = np.arange(n) / SR
    y = np.sin(2 * np.pi * f0 * u) * np.exp(-u * 60) + 0.5 * np.sin(2 * np.pi * f0 * 1.5 * u) * np.exp(-u * 80)
    return y / np.abs(y).max()


def heartbeat():
    """Two low thumps (lub-dub)."""
    out = np.zeros(int(0.5 * SR))
    for t0, g in ((0.0, 1.0), (0.17, 0.7)):
        b = boom(0.25, 70, 38) * g; i = int(t0 * SR); out[i:i + len(b)] += b[:len(out) - i]
    return out / np.abs(out).max()


def squeak(dur=0.35):
    """Rubber plunger squeak: a wobbling resonant glide."""
    n = int(dur * SR); u = np.arange(n) / SR
    f = 900 + 500 * u / dur + 60 * np.sin(2 * np.pi * 23 * u)
    y = np.sin(2 * np.pi * np.cumsum(f) / SR) * (0.5 + 0.5 * np.sin(2 * np.pi * 31 * u)) * np.sin(np.pi * u / dur)
    y = sg.lfilter(*sg.butter(2, [600 / (SR / 2), 3500 / (SR / 2)], 'band'), y)
    return y / np.abs(y).max()


def downer(dur=0.9):
    """Woozy pitch-down whoosh (sedation)."""
    n = int(dur * SR); u = np.arange(n) / SR
    f = 420 * np.exp(-u * 2.2) + 50
    y = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.sin(np.pi * u / dur) ** 0.7
    y += 0.5 * sg.lfilter(*sg.butter(2, 700 / (SR / 2)), rng.standard_normal(n)) * np.sin(np.pi * u / dur)
    return y / np.abs(y).max()


def chime(dur=1.4, f0=880):
    """Clock chime: inharmonic bell partials with long decay."""
    n = int(dur * SR); u = np.arange(n) / SR
    y = sum(a * np.sin(2 * np.pi * f0 * r * u) * np.exp(-u * d) for r, a, d in ((1.0, 1.0, 2.5), (2.0, 0.5, 3.5), (2.76, 0.35, 5), (5.4, 0.2, 8)))
    return y / np.abs(y).max()


def wind(dur=0.7):
    """Soft rolling wind: low-passed noise with a slow swell."""
    n = int(dur * SR); u = np.linspace(0, 1, n)
    y = sg.lfilter(*sg.butter(2, 900 / (SR / 2)), rng.standard_normal(n)) * np.sin(np.pi * u) ** 1.2
    return y / np.abs(y).max()


def rewind_whine(dur=0.6):
    """Tape rewind: a rising, wobbling squeal with motor hiss."""
    n = int(dur * SR); u = np.arange(n) / SR
    f = 180 * (14 ** (u / dur)) * (1 + 0.03 * np.sin(2 * np.pi * 9 * u))
    y = np.sin(2 * np.pi * np.cumsum(f) / SR) * 0.6 + 0.4 * sg.lfilter(*sg.butter(2, [2000 / (SR / 2), 8000 / (SR / 2)], 'band'), rng.standard_normal(n))
    y *= np.sin(np.pi * u / dur) ** 0.5
    return y / np.abs(y).max()


def pluck(f0=220, dur=0.5):
    """Taut string twang: decaying harmonics with a slight pitch drop."""
    n = int(dur * SR); u = np.arange(n) / SR; f = f0 * (1 + 0.02 * np.exp(-u * 30))
    y = sum(np.sin(2 * np.pi * np.cumsum(f * h) / SR) * np.exp(-u * (6 + 3 * h)) / h for h in (1, 2, 3, 5))
    return y / np.abs(y).max()


def zipper(dur=0.25):
    """Zip-lock seal: a fast run of clicks rising in rate."""
    n = int(dur * SR); y = np.zeros(n); k = 0.0
    while k < dur:
        i = int(k * SR); m = min(120, n - i); y[i:i + m] += rng.standard_normal(m) * np.exp(-np.arange(m) / 20); k += 0.012 * (1 - 0.6 * k / dur)
    y = sg.lfilter(*sg.butter(2, [1500 / (SR / 2), 8000 / (SR / 2)], 'band'), y)
    return y / np.abs(y).max()


# ---- SFX follow the plan: each transition type has its sound, placed relative to its cut ----
def sfx_for(name, c, o, up):
    if name == 'leaf': put(whoosh(0.34, 700, 5200) if up else whoosh(0.34, 5200, 800), c - 0.20, gain=0.085)
    elif name == 'blur': put(whoosh(0.36, 300, 1500, q=0.8), c - 0.18, gain=0.06)
    elif name == 'glass': put(whoosh(0.26, 1500, 6500), c - 0.14, gain=0.075)
    elif name == 'smoke': put(whoosh(0.60, 160, 900, q=0.7, peak=0.4), c - 0.228, gain=0.11)
    elif name == 'slidewipe': put(whoosh(0.52, 400, 3800, peak=0.8), c - 0.48, gain=0.085)
    elif name == 'blood':
        put(whoosh(0.70, 220, 700, q=0.7, peak=0.4), c - 0.32, gain=0.12); put(bubbles(0.55, 18), c - 0.26, gain=0.05)
    elif name == 'spin': put(whoosh(0.42, 500, 4200, peak=0.5), c - 0.21, gain=0.095)
    elif name == 'polaroid':
        put(shutter(), c - 0.012, gain=0.16); put(whoosh(0.30, 2500, 600, peak=0.3), c + 0.30, gain=0.05)
    elif name == 'drops':
        put(splat(), c - 0.22, pan=-0.2, gain=0.13); put(splat(0.16), c - 0.10, pan=0.3, gain=0.09)
    elif name == 'whip':
        d = o.get('dur', 0.20); big = o.get('amount', 0.9) > 0.7
        f0, f1 = (900, 5200) if o.get('d', 1) > 0 else (5200, 900)
        if o.get('vertical'): f0, f1 = 1500, 5000
        put(whoosh(d + 0.04, f0, f1), c - d / 2 - 0.02, gain=0.09 if big else 0.06)
    elif name == 'red':
        put(whoosh(0.34, 250, 2600, peak=0.92), c - 0.33, gain=0.09); put(boom(), c, gain=0.26)
    elif name == 'speedlines':
        put(whoosh(0.40, 900, 7000, peak=0.45), c - 0.20, pan=0.5 * o.get('d', 1), gain=0.10); put(zap(0.2), c - 0.05, gain=0.04)
    elif name == 'phase':
        put(buzz(0.45), c - 0.22, gain=0.08); put(zap(0.25), c - 0.02, gain=0.05)
    elif name == 'emblem':
        put(whoosh(0.26, 2500, 7000, peak=0.6), c - 0.60, gain=0.05); put(shimmer(0.3), c - 0.55, gain=0.03)
        put(boom(0.3, 140, 60), c - 0.24, gain=0.09); put(zap(0.25), c - 0.24, gain=0.07)
        put(shimmer(0.25), c - 0.20, pan=0.3, gain=0.03); put(whoosh(0.36, 600, 5200, peak=0.2), c, gain=0.08)
    elif name == 'emblem_real':
        put(buzz(0.8), c - 0.80, gain=0.05); put(whoosh(0.5, 400, 4200, peak=0.9), c - 0.50, gain=0.07)
        put(zap(0.4), c - 0.06, gain=0.08); put(boom(0.45, 120, 45), c, gain=0.12)
    elif name == 'strike':
        put(zap(0.25), c - 0.22, gain=0.05); put(thunder(), c, gain=0.17)
    elif name == 'boltsplit':
        put(zap(0.3), c - 0.40, gain=0.06); put(whoosh(0.35, 800, 6000, peak=0.6), c - 0.15, gain=0.09); put(boom(0.4, 130, 50), c - 0.12, gain=0.10)
    elif name == 'tunnel':
        put(whoosh(0.62, 250, 6500, q=0.8, peak=0.9), c - 0.55, gain=0.10); put(zap(0.3), c - 0.05, gain=0.06)
    elif name == 'knife':
        put(whoosh(0.30, 600, 4200, peak=0.85), c - 0.30, pan=0.4, gain=0.07); put(shing(), c - 0.12, pan=0.2, gain=0.11)
        put(boom(0.35, 110, 50), c, gain=0.08)
    elif name == 'newspaper':
        put(whoosh(0.50, 300, 3000, q=0.6, peak=0.9), c - 0.62, gain=0.08); put(crinkle(0.25), c - 0.30, gain=0.05)
        put(boom(0.25, 140, 60), c - 0.12, gain=0.10); put(splat(0.05), c - 0.12, gain=0.07); put(whoosh(0.40, 400, 3500, peak=0.85), c, gain=0.06)
    elif name == 'badge':
        put(whoosh(0.30, 2500, 600, peak=0.5), c - 0.55, gain=0.05); put(tick(0.07, 1300), c - 0.27, gain=0.06); put(tick(0.05, 1900), c - 0.22, gain=0.04)
        put(shimmer(0.4), c - 0.30, pan=0.3, gain=0.03); put(whoosh(0.45, 300, 3500, peak=0.85), c - 0.15, gain=0.07)
    elif name == 'slidebox':
        put(whoosh(0.30, 3000, 600, peak=0.2), c - 0.85, gain=0.06)
        for k in range(14): put(tick(0.04, 3800 - 120 * k), c - 0.60 + 0.026 * k, pan=0.2, gain=0.03)
        put(whoosh(0.30, 500, 4500, peak=0.9), c - 0.25, gain=0.07); put(tick(0.15, 4400), c + 0.05, gain=0.05)
    elif name == 'photo':
        put(tick(0.05, 3200), c - 0.50, gain=0.04); put(tick(0.05, 3600), c - 0.42, gain=0.04)
        put(shutter(), c - 0.06, gain=0.17); put(whoosh(0.3, 5000, 900, peak=0.15), c, gain=0.05)
    elif name == 'casefile':
        put(whoosh(0.38, 900, 200, q=0.7, peak=0.8), c - 0.38, gain=0.08); put(boom(0.3, 140, 60), c, gain=0.09); put(splat(0.05), c, gain=0.05)
        put(whoosh(0.35, 200, 900, q=0.7, peak=0.2), c + 0.02, gain=0.06)
    elif name == 'strings':
        for k in range(9):
            put(splat(0.06), c - 0.55 + (0.03 + 0.32 * k / 8) * 0.55, pan=-0.6 + 0.15 * k, gain=0.045)
            put(pluck(196 * (1 + 0.06 * k)), c - 0.55 + (0.39 + 0.32 * k / 8) * 0.55, pan=-0.6 + 0.15 * k, gain=0.035)
        put(boom(0.35, 120, 50), c, gain=0.08); put(whoosh(0.35, 400, 4000, peak=0.2), c, gain=0.06)
    elif name == 'bag':
        put(whoosh(0.30, 2500, 500, peak=0.3), c - 0.45, gain=0.05); put(crinkle(0.35), c - 0.30, gain=0.07)
        put(zipper(0.20), c - 0.10, gain=0.07); put(whoosh(0.35, 1500, 250, peak=0.3), c, gain=0.07)
    elif name == 'crown':
        put(whoosh(0.30, 1200, 300, peak=0.85), c - 0.38, gain=0.04); put(splat(0.25), c - 0.08, gain=0.14); put(boom(0.3, 100, 45), c - 0.08, gain=0.07)
    elif name == 'bloodwater':
        put(wind(0.9), c - 0.45, gain=0.07); put(bubbles(0.8, 30), c - 0.40, gain=0.07); put(boom(0.6, 60, 35), c - 0.45, gain=0.06)
    elif name == 'blinds':
        for k in range(8): put(tick(0.04, 1500 + 120 * k), c - 0.36 + 0.035 * k, pan=-0.4 + 0.1 * k, gain=0.04)
        put(whoosh(0.30, 600, 2500, peak=0.7), c - 0.30, gain=0.05)
        for k in range(8): put(tick(0.04, 2400 - 120 * k), c + 0.02 + 0.035 * k, pan=0.4 - 0.1 * k, gain=0.04)
    elif name == 'rewind':
        put(rewind_whine(0.62), c - 0.52, gain=0.07); put(tick(0.06, 1200), c - 0.53, gain=0.08); put(tick(0.06, 1000), c + 0.09, gain=0.08)
    elif name == 'kaleido':
        put(shimmer(0.8), c - 0.45, pan=-0.3, gain=0.04); put(whoosh(0.85, 300, 4000, q=0.7, peak=0.5), c - 0.45, gain=0.07)
    elif name == 'tape':
        put(whoosh(0.30, 800, 5000, peak=0.6), c - 0.34, pan=0.5, gain=0.07); put(crinkle(0.35), c - 0.25, gain=0.06)
        put(crinkle(0.25), c + 0.02, pan=-0.4, gain=0.08); put(whoosh(0.25, 4000, 900, peak=0.2), c, gain=0.06)
    elif name == 'rorschach':
        put(splat(0.18), c - 0.30, gain=0.09); put(whoosh(0.7, 200, 900, q=0.6, peak=0.6), c - 0.28, gain=0.07)
    elif name == 'mirror':
        put(shimmer(0.5), c - 0.45, gain=0.04); put(tick(0.10, 3000), c - 0.02, gain=0.06)
        put(whoosh(0.40, 1800, 300, q=0.8, peak=0.25), c, pan=-0.5, gain=0.06); put(whoosh(0.40, 1800, 300, q=0.8, peak=0.25), c, pan=0.5, gain=0.06)
    elif name == 'sunburst':
        put(shimmer(0.7), c - 0.32, pan=0.5, gain=0.04); put(whoosh(0.45, 300, 6000, q=0.7, peak=0.85), c - 0.40, gain=0.07); put(boom(0.4, 110, 50), c, gain=0.06)
    elif name == 'clouds':
        put(wind(0.75), c - 0.38, pan=-0.3, gain=0.10); put(whoosh(0.6, 250, 1200, q=0.6, peak=0.5), c - 0.30, gain=0.05)
    elif name == 'runby':
        put(whoosh(0.40, 600, 7000, peak=0.6), c - 0.24, gain=0.11); put(zap(0.35), c - 0.10, pan=0.4, gain=0.06)
    elif name == 'clock':
        for k in range(10): put(tick(0.05, 2600), c - 0.80 + 0.5 * (k / 10) ** 0.6, pan=0.2, gain=0.05)
        put(whoosh(0.5, 4000, 300, q=0.7, peak=0.3), c - 0.30, gain=0.08); put(chime(1.2, 880), c - 0.05, gain=0.035)
    elif name == 'ink':
        put(whoosh(0.40, 180, 1200, q=0.7, peak=0.7), c - 0.36, gain=0.09); put(splat(0.14), c - 0.30, pan=-0.3, gain=0.07)
        put(splat(0.12), c - 0.20, pan=0.4, gain=0.06); put(whoosh(0.30, 1200, 300, peak=0.2), c, gain=0.06)
    elif name == 'orange':
        put(whoosh(0.30, 500, 3200, peak=0.8), c - 0.42, pan=0.5, gain=0.08); put(splat(0.20), c - 0.18, gain=0.12)
        put(whoosh(0.25, 4000, 9000, q=0.6, peak=0.2), c - 0.17, gain=0.04); put(whoosh(0.40, 3000, 600, peak=0.15), c, gain=0.07)
    elif name == 'slices':
        for k in range(5): put(shing(0.35), c - 0.14 + 0.018 * k, pan=-0.6 + 0.3 * k, gain=0.035)
        put(whoosh(0.30, 2000, 400, peak=0.2), c, gain=0.08); put(boom(0.3, 120, 50), c, gain=0.07)
    elif name == 'syringe':
        put(whoosh(0.30, 600, 3000, peak=0.85), c - 0.78, pan=0.5, gain=0.06); put(tick(0.12, 3200), c - 0.48, gain=0.06)
        put(squeak(0.30), c - 0.45, gain=0.035); put(heartbeat(), c - 0.40, gain=0.16); put(downer(0.9), c - 0.40, gain=0.08)
        put(heartbeat(), c + 0.05, gain=0.13); put(whoosh(0.35, 300, 1800, q=0.7, peak=0.6), c + 0.12, gain=0.05)
    elif name == 'shatter':
        for k in range(3): put(splat(0.03), c - 0.11 + 0.035 * k, pan=0.2 * k - 0.2, gain=0.05)
        put(glass(), c, gain=0.16); put(boom(0.35, 120, 50), c, gain=0.10)
    elif name == 'scope':
        put(whoosh(0.42, 400, 3200, peak=0.9), c - 0.42, gain=0.08); put(shutter(), c - 0.01, gain=0.07)
        put(whoosh(0.40, 3200, 500, peak=0.15), c, gain=0.07)
    elif name == 'plastic':
        put(crinkle(0.72), c - 0.30, pan=0.3, gain=0.10); put(whoosh(0.7, 300, 1400, q=0.7, peak=0.5), c - 0.30, gain=0.05)
    elif name == 'confetti':
        put(splat(0.09), c - 0.24, pan=0.1, gain=0.10); put(boom(0.25, 150, 70), c - 0.24, gain=0.07)
        put(whoosh(0.62, 2500, 7500, q=0.6, peak=0.3), c - 0.20, gain=0.05)
    elif name == 'water': put(bubbles(0.75), c - 0.30, gain=0.10)
    elif name == 'iris': put(whoosh(0.50, 300, 4200, peak=0.9), c - 0.35, gain=0.09)


for i, s in enumerate(SHOTS):
    o = s.get('out_opts', {}); name = s.get('out', 'cut'); c = CUTS[i + 1]
    sfx_for(name, c, o, up=math.cos(math.radians(o.get('ang', -38))) > 0)
    _hs = sorted(GRID0 + k_ * HALF for k_ in globals().get('KICKS', []) + globals().get('SNARES', []))
    _hits_in = [x for x in _hs if s['start'] + 0.12 <= x <= s['end'] - 0.15]
    for fxn in s.get('fx', ()):
        if fxn.startswith('stamp:'):
            t0 = _hits_in[0] if _hits_in else s['start'] + 0.25
            put(boom(0.25, 160, 60), t0, gain=0.10); put(splat(0.05), t0, gain=0.06)
        elif fxn == 'markers':
            for k_, t0 in enumerate((_hits_in or [s['start'] + 0.2])[:3]): put(tick(0.06, 1700 + 200 * k_), t0, pan=-0.5 + 0.5 * k_, gain=0.05)
        elif fxn == 'idbox':
            t0 = (_hits_in or [s['start'] + 0.1])[0]; put(tick(0.08, 2900), t0, gain=0.05)
            for k_ in range(8): put(tick(0.02, 4200), t0 + 0.13 + 0.03 * k_, gain=0.015)
    if 'crackle_blue' in s.get('fx', ()): put(zap(s['end'] - s['start']), s['start'], gain=0.03)
    if 'crackle_build' in s.get('fx', ()): put(zap(s['end'] - s['start']), s['start'], gain=0.035)
    if 'streaks' in s.get('fx', ()): put(shimmer(0.62), s['end'] - 0.62, pan=0.3, gain=0.030)
    if 'punch' in s and i > 0:
        big = s['punch'].get('z', 1.22) > 1.18
        put(boom(0.5, 80, 42) if big else boom(0.4, 70, 40), s['start'], gain=0.14 if big else 0.10)

for line in globals().get('TEXT', []):
    for w in line['words']:
        tin, kind = w[0], w[2]
        if kind in ('pop', 'slide'): put(tick(0.07, 2200 if kind == 'pop' else 1800), tin, pan=0.1, gain=0.035)
        elif kind == 'forever': put(shimmer(0.9), tin, pan=0.2, gain=0.03)
        elif kind == 'glitch': put(zap(0.2), tin, gain=0.04)
        elif kind == 'impact': put(boom(0.6, 90, 40), tin, gain=0.045)   # usually stacks with a transition hit on the drop

mix = music + fx
fade = int(0.008 * SR); ramp = np.linspace(0, 1, fade)[:, None]
mix[:fade] *= ramp; mix[-fade:] *= ramp[::-1]
pk = np.abs(mix).max()
if pk > 0.97: mix *= 0.97 / pk
import os; os.makedirs('out', exist_ok=True)
sf.write('out/audio.wav', mix.astype(np.float32), SR)
sf.write('out/sfx_only.wav', fx.astype(np.float32), SR)
print('music peak %.2f  sfx peak %.3f  mix peak %.2f  sfx rms/music rms %.3f' % (np.abs(music).max(), np.abs(fx).max(), np.abs(mix).max(), np.sqrt((fx ** 2).mean()) / np.sqrt((music ** 2).mean())))
