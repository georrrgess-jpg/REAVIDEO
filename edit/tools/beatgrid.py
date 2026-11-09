"""Beat grid for a song: tempo, half-beat grid, and the best-energy section to cut to.

  python beatgrid.py SONG [--len 22.65] [--ref REFERENCE_EDIT.mp4] [-o beats.json]

--ref: a reference edit that uses the same song; its audio is located inside the song
       and that section is used (this is how the Doomsday section 65.643 s was found).
Without --ref the section with the strongest, most even drum activity is suggested.
Output JSON: bpm, period, offset (song time of section start), grid (half-beat times
relative to the section start), beats_strength (hi-hat/snare onset per half-beat).
"""
import argparse, json, subprocess
import numpy as np, scipy.signal as ss

SR = 22050; HOP = 110; DT = HOP / SR


def load(path):
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', path, '-ac', '1', '-ar', str(SR), '-f', 'f32le', '-'],
                         stdout=subprocess.PIPE, check=True).stdout
    return np.frombuffer(raw, np.float32).astype(np.float64)


def onset(x, lo=0, hi=1e9):
    f, _, Z = ss.stft(x, SR, nperseg=1024, noverlap=1024 - HOP)
    M = np.log1p(50 * np.abs(Z[(f >= lo) & (f < hi)]))
    return np.maximum(0, np.diff(M, axis=1)).sum(0)


def tempo(o, lo_bpm=60, hi_bpm=180):
    o = o - o.mean()
    ac = ss.fftconvolve(o, o[::-1], 'full')[len(o) - 1:]
    lags = np.arange(len(ac)) * DT
    best = None
    for bpm in np.arange(lo_bpm, hi_bpm, 0.05):
        p = 60 / bpm
        s = sum(np.interp(k * p, lags, ac) for k in range(1, 9)) + 0.5 * np.interp(p / 2, lags, ac)
        if best is None or s > best[0]: best = (s, p)
    p = best[1]
    if 60 / p > 140: p *= 2          # prefer the felt (half-time) pulse for edits
    return p


def phase(o, p):
    t = np.arange(len(o)) * DT; best = None
    for ph in np.arange(0, p / 2, 0.002):
        s = np.interp(np.arange(ph, t[-1], p / 2), t, o).sum()
        if best is None or s > best[0]: best = (s, ph)
    return best[1]


def locate(song, ref):
    c = ss.fftconvolve(song, ref[::-1], 'valid')
    e = np.sqrt(ss.fftconvolve(song ** 2, np.ones(len(ref)), 'valid') * np.sum(ref ** 2)) + 1e-9
    n = c / e; i = int(np.argmax(np.abs(n)))
    return i / SR, float(n[i])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('song'); ap.add_argument('--len', type=float, default=22.65)
    ap.add_argument('--ref'); ap.add_argument('-o', default='beats.json')
    a = ap.parse_args()
    song = load(a.song); dur = len(song) / SR
    o_all = onset(song)
    p = tempo(o_all[: int(min(dur, 120) / DT)] if dur > 150 else o_all)
    if a.ref:
        ref = load(a.ref)
        off, corr = locate(song, ref); a.len = len(ref) / SR
        print('reference found in song at %.3f s (correlation %.2f)' % (off, corr))
    else:
        oh = onset(song, 2000, 9000); best = None
        step = p * 4
        for s0 in np.arange(8.0, dur - a.len - 4, step):
            seg = oh[int(s0 / DT): int((s0 + a.len) / DT)]
            q = np.array_split(seg, 6); m = np.array([x.mean() for x in q])
            score = m.mean() - 0.6 * m.std()          # loud and even drums, no breakdowns
            if best is None or score > best[0]: best = (score, s0)
        off, corr = best[1], None
        print('suggested section starts at %.2f s (%d:%05.2f)' % (off, off // 60, off % 60))
    seg = song[int(off * SR): int((off + a.len) * SR)]
    ph = (phase(onset(seg), p))
    grid = np.arange(ph, a.len, p / 2)
    oh = onset(seg, 2000, 9000)
    strength = [float(oh[max(0, int(g / DT) - 8): int(g / DT) + 9].max()) for g in grid]
    out = dict(song=a.song, bpm=round(60 / p, 2), period=p, offset=round(off, 3), length=a.len,
               grid=[round(float(g), 4) for g in grid], beats_strength=strength, ref_correlation=corr)
    json.dump(out, open(a.o, 'w'), indent=1)
    print('%.2f BPM, beat %.4f s, half-beat %.4f s, first half-beat at %.3f s; %d grid points -> %s'
          % (60 / p, p, p / 2, ph, len(grid), a.o))


if __name__ == '__main__':
    main()
