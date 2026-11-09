"""Map a song's structure and pick the edit section.

  python sections.py SONG [--bars 8] [-o section.json]

Prints energy per bar (low / mid / high bands) so intros, drops, bridges and fills stand out;
groups 8-bar sections by harmonic similarity (beat-aligned chroma) to find the chorus (the
group that includes the last full section, since songs usually end on the hook); then refines the
downbeat and lists kick / snare strength per half-beat for the chosen section, so the heaviest
transitions can go on the loudest hits.

Default pick: the last chorus with full drums, `--bars` long (8 bars = 64 half-beats), which loops
musically (its closing fill lands back on its own downbeat). Writes offset, half, bars and the
per-half-beat kick / snare strengths to -o. Use --start SEC to force a section start.
"""
import argparse, json, subprocess
import numpy as np, scipy.signal as ss

SR = 22050


def load(path, a=0, d=None):
    cmd = ['ffmpeg', '-v', 'error', '-ss', str(a)] + (['-t', str(d)] if d else []) + ['-i', path, '-ac', '1', '-ar', str(SR), '-f', 'f32le', '-']
    return np.frombuffer(subprocess.run(cmd, stdout=subprocess.PIPE, check=True).stdout, np.float32).astype(float)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('song'); ap.add_argument('--bars', type=int, default=8)
    ap.add_argument('--start', type=float); ap.add_argument('-o', default='section.json'); a = ap.parse_args()
    x = load(a.song); dur = len(x) / SR
    f, _, Z = ss.stft(x, SR, nperseg=2048, noverlap=2048 - 256); M = np.abs(Z); dt = 256 / SR
    low, mid, hi = (M[(f > lo) & (f < h)].sum(0) for lo, h in ((30, 120), (150, 2500), (4000, 11000)))
    on = np.maximum(0, np.diff(np.log1p(M).sum(0))); tt = np.arange(len(on)) * dt
    # tempo (60-180 BPM, prefer the felt pulse under 140) and beat phase
    o = on - on.mean(); ac = ss.fftconvolve(o, o[::-1], 'full')[len(o) - 1:]; lags = np.arange(len(ac)) * dt
    P = max(((sum(np.interp(k * p, lags, ac) for k in range(1, 9)), p) for p in 60 / np.arange(60, 180, 0.05)))[1]
    if 60 / P > 140: P *= 2
    lo_on = np.maximum(0, np.diff(np.log1p(low)))
    bp = max(((np.interp(np.arange(p, tt[-1], P), tt, lo_on).sum(), p) for p in np.arange(0, P, 0.005)))[1]
    db = bp + P * int(np.argmax([np.interp(np.arange(bp + q * P, tt[-1], 4 * P), tt, lo_on).mean() for q in range(4)]))
    bar = 4 * P; nb = int((dur - db) / bar)
    seg = lambda arr, s: arr[int(s / dt):int((s + bar) / dt)].mean()
    avg = mid.mean()
    print('%.2f BPM, bar %.2f s, first downbeat %.3f s\n bar   time   low  mid  high' % (60 / P, bar, db))
    lv = []
    for k in range(nb):
        s = db + k * bar; lv.append(seg(low, s))
        print('%4d %6.1f %5.2f %4.2f %5.2f %s' % (k, s, seg(low, s), seg(mid, s) / avg, seg(hi, s), '#' * int(seg(mid, s) / avg * 20)))
    # 8-bar sections starting where full drums enter, grouped by beat-aligned chroma
    f2, _, Z2 = ss.stft(x, SR, nperseg=8192, noverlap=8192 - 2048); M2 = np.abs(Z2); dt2 = 2048 / SR
    pc = np.zeros((12, M2.shape[1]))
    for i, fr in enumerate(f2):
        if 80 < fr < 4000: pc[int(round(12 * np.log2(fr / 440))) % 12] += M2[i]
    pc /= pc.sum(0, keepdims=True) + 1e-9
    def chroma(s): return np.array([pc[:, int((s + k * P) / dt2):int((s + (k + 1) * P) / dt2)].mean(1) for k in range(32)])
    def sim(A, B):
        A = A - A.mean(1, keepdims=True); B = B - B.mean(1, keepdims=True); return float((A * B).sum() / np.sqrt((A * A).sum() * (B * B).sum()))
    thr = 0.5 * max(lv); k0 = next(k for k, v in enumerate(lv) if v > thr)
    starts = [db + k * bar for k in range(k0, nb - 7, 8)]
    last = max(s for s in starts if seg(low, s) > thr)
    ch = {s: chroma(s) for s in starts}
    chorus = [s for s in starts if sim(ch[s], ch[last]) > 0.85]
    print('\n8-bar sections (similarity to the last full section):')
    for s in starts: print('  %3d:%04.1f  %.2f %s' % (s // 60, s % 60, sim(ch[s], ch[last]), '<- chorus' if s in chorus else ''))
    start = a.start if a.start is not None else max(chorus)
    # refine to the nearest half-beat with a fine phase search around the section
    y = load(a.song, max(0, start - 8), a.bars * bar + 16); hop = 64; d3 = hop / SR
    f3, _, Z3 = ss.stft(y, SR, nperseg=1024, noverlap=1024 - hop); L = np.log1p(50 * np.abs(Z3))
    band = lambda lo, h: np.maximum(0, np.diff(L[(f3 >= lo) & (f3 < h)], axis=1)).sum(0)
    o3 = band(0, 1e9); t3 = np.arange(len(o3)) * d3 + max(0, start - 8)
    ph = max(((np.interp(np.arange(t3[0] + p, t3[-1], P / 2), t3, o3).sum(), p) for p in np.arange(0, P / 2, 0.001)))[1]
    start = t3[0] + ph + round((start - t3[0] - ph) / (P / 2)) * P / 2
    kick, snare = band(30, 120), band(150, 400) + 0.5 * band(1500, 5000)
    pk = lambda o, t: o[max(0, int((t - t3[0]) / d3) - 12):int((t - t3[0]) / d3) + 13].max()
    g = [start + i * P / 2 for i in range(a.bars * 8 + 1)]
    K = np.array([pk(kick, t) for t in g]); S = np.array([pk(snare, t) for t in g]); K /= K.max(); S /= S.max()
    loud = sorted(range(len(g)), key=lambda i: -(K[i] + S[i]))[:8]
    print('\nsection %d:%06.3f-%d:%06.3f (%d bars, %.2f s); loudest half-beats: %s' % (start // 60, start % 60, (start + a.bars * bar) // 60,
          (start + a.bars * bar) % 60, a.bars, a.bars * bar, ', '.join('#%d (%.2f s)' % (i, i * P / 2) for i in sorted(loud))))
    json.dump(dict(song=a.song, bpm=round(60 / P, 2), offset=round(start, 3), half=round(P / 2, 5), bars=a.bars, length=round(a.bars * bar, 3),
                   kick=K.round(2).tolist(), snare=S.round(2).tolist(), loudest=sorted(loud)), open(a.o, 'w'), indent=1)
    print('-> %s  (config: SONG_OFFSET = offset, GRID0 = 0, HALF = half, TAIL = length - (last grid point * HALF + 0.077))' % a.o)


if __name__ == '__main__':
    main()
