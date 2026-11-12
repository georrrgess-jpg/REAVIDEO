"""Rank catalogued shots by real on-screen movement (dense optical flow), to avoid shots that read as stills.

  python edit/tools/motion.py PACK [PACK ...] [--min-face 0.25] [-o media/motion.json]

For each shot of 1.5 s or more with a usable face (from media/scores.json, written by score.py): mean
Farneback flow magnitude over its middle 1.5 s at 192x108 (pixels per frame at that size), plus the flow
inside the face box (head movement). Writes a list sorted by total motion.
"""
import argparse, json, os, sys
import numpy as np, cv2
from multiprocessing import Pool


def flow_stats(job):
    pack, s = job; cv2.setNumThreads(1)
    cap = cv2.VideoCapture('media/scenepacks/%s.mp4' % pack); fps = cap.get(cv2.CAP_PROP_FPS) or 30
    mid = (s['a'] + s['b']) / 2; t0 = max(s['a'] + 0.1, mid - 0.75); cap.set(cv2.CAP_PROP_POS_MSEC, t0 * 1000)
    prev = None; mags = []; n = int(1.5 * fps); step = max(1, int(round(fps / 12)))
    for k in range(n):
        ok, f = cap.read()
        if not ok: break
        if k % step: continue
        g = cv2.cvtColor(cv2.resize(f, (192, 108)), cv2.COLOR_BGR2GRAY)
        if prev is not None:
            fl = cv2.calcOpticalFlowFarneback(prev, g, None, 0.5, 2, 9, 2, 5, 1.1, 0)
            m = np.sqrt((fl ** 2).sum(-1))
            if m.mean() > 1e-3: mags.append(m.mean())                 # skip repeated frames
        prev = g
    return dict(pack=pack, i=s['i'], a=s['a'], b=s['b'], face=s['face'], fx=s['fx'], sharp=s['sharp'], motion=round(float(np.mean(mags)) if mags else 0.0, 3))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('packs', nargs='+'); ap.add_argument('--min-face', type=float, default=0.25)
    ap.add_argument('--scores', default='media/scores.json'); ap.add_argument('-o', default='media/motion.json'); a = ap.parse_args()
    S = json.load(open(a.scores))
    jobs = [(p, s) for p in a.packs for s in S.get(p, []) if s and s['b'] - s['a'] >= 1.5 and s['face'] >= a.min_face and s['sharp'] >= 100]
    with Pool(os.cpu_count() or 4) as pool: res = pool.map(flow_stats, jobs)
    res.sort(key=lambda r: -r['motion']); json.dump(res, open(a.o, 'w'), indent=0)
    print(len(res), 'shots ranked ->', a.o)
