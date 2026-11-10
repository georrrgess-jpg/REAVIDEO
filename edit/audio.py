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
    if 'crackle_blue' in s.get('fx', ()): put(zap(s['end'] - s['start']), s['start'], gain=0.03)
    if 'crackle_build' in s.get('fx', ()): put(zap(s['end'] - s['start']), s['start'], gain=0.035)
    if 'streaks' in s.get('fx', ()): put(shimmer(0.62), s['end'] - 0.62, pan=0.3, gain=0.030)
    if 'punch' in s and i > 0:
        big = s['punch'].get('z', 1.22) > 1.18
        put(boom(0.5, 80, 42) if big else boom(0.4, 70, 40), s['start'], gain=0.14 if big else 0.10)

mix = music + fx
fade = int(0.008 * SR); ramp = np.linspace(0, 1, fade)[:, None]
mix[:fade] *= ramp; mix[-fade:] *= ramp[::-1]
pk = np.abs(mix).max()
if pk > 0.97: mix *= 0.97 / pk
import os; os.makedirs('out', exist_ok=True)
sf.write('out/audio.wav', mix.astype(np.float32), SR)
sf.write('out/sfx_only.wav', fx.astype(np.float32), SR)
print('music peak %.2f  sfx peak %.3f  mix peak %.2f  sfx rms/music rms %.3f' % (np.abs(music).max(), np.abs(fx).max(), np.abs(mix).max(), np.sqrt((fx ** 2).mean()) / np.sqrt((music ** 2).mean())))
