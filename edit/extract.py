"""Cut every planned shot out of its scene pack as a face-tracked, graded 9:16 clip.

  python extract.py            # all shots (parallel, one process per core)
  python extract.py 3 7        # only these shots, e.g. after changing their src, framing or grade

For each shot this decodes src - PRE .. end + POST, tracks the face, crops a 9:16 window, upscales
it (bicubic + unsharp mask), applies the shot's grade, and stores high-quality 4:4:4 JPEG
frames in clips/NN/. It also finds the usable frame range (stops at hard cuts inside the source
clip), writes clips/meta.json and makes out/sheets/clips.jpg to check framing.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.getcwd())   # run from an edit folder: its config.py wins, engine modules come from edit/
import json, os, subprocess, sys
from multiprocessing import Pool
import numpy as np, cv2
from config import *

FACE_MODEL = os.path.expanduser('~/.cache/rea-video/models/face_detection_yunet_2023mar.onnx')
JPEG = [cv2.IMWRITE_JPEG_QUALITY, 97, cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444]
_det = {}


def probe(path):
    o = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height',
                        '-of', 'csv=p=0', path], stdout=subprocess.PIPE, text=True, check=True).stdout
    return tuple(int(v) for v in o.strip().split(',')[:2])


def read(path, a, d, SW, SH):
    p = subprocess.Popen(['ffmpeg', '-v', 'error', '-ss', '%.4f' % max(a, 0), '-i', path, '-t', '%.4f' % d, '-vf', 'fps=%d' % FPS,
                          '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-'], stdout=subprocess.PIPE)
    out = []
    while True:
        b = p.stdout.read(SW * SH * 3)
        if len(b) < SW * SH * 3: break
        out.append(np.frombuffer(b, np.uint8).reshape(SH, SW, 3))
    p.wait()
    return out


def track(fr, s, SW, SH):
    if (SW, SH) not in _det:
        _det[(SW, SH)] = cv2.FaceDetectorYN.create(FACE_MODEL, '', (SW, SH), 0.55, 0.3, 50)
    det = _det[(SW, SH)]; n = len(fr); xs = np.full(n, np.nan)
    for k in range(0, n, 3):
        _, faces = det.detect(fr[k])
        if faces is not None and len(faces):
            c = [(f[0] + f[2] / 2) / SW for f in faces]
            j = int(np.argmin([abs(v - s['x']) for v in c]))
            if abs(c[j] - s['x']) < 0.16: xs[k] = c[j]
    good = ~np.isnan(xs)
    if s.get('track', True) and good.sum() >= 4:
        idx = np.arange(n); x = np.interp(idx, idx[good], xs[good])
        if n > 30: x = cv2.GaussianBlur(x.reshape(-1, 1), (1, 0), 0, sigmaY=14).ravel()
        x = 0.7 * x + 0.3 * s['x']            # stay close to the planned composition
    else:
        x = np.full(n, s['x'])
    return x, int(good.sum())


def one(i):
    cv2.setNumThreads(1)
    from fx import grade_u8                   # imported here so workers don't need clips/meta.json
    s = SHOTS[i]; path = s.get('pack', SCENEPACK); SW, SH = probe(path)
    fr = read(path, s['src'] - PRE, (s['end'] - s['start']) + PRE + POST, SW, SH); n = len(fr)
    if n == 0: raise RuntimeError('shot %d (%s): no frames at src=%.2f in %s' % (i, s['name'], s['src'], path))
    x, nfaces = track(fr, s, SW, SH)
    ch = SH / s['z']; cw = ch * W / H         # full-height 9:16 window, tightened by z
    cx = np.clip(x * SW, cw / 2, SW - cw / 2); cy = np.clip(s.get('cy', .5) * SH, ch / 2, SH - ch / 2)
    d = os.path.join('clips', '%02d' % i); os.makedirs(d, exist_ok=True)
    for f in os.listdir(d): os.remove(os.path.join(d, f))
    small = []
    for k in range(n):
        M = np.float32([[W / cw, 0, -(cx[k] - cw / 2) * W / cw], [0, H / ch, -(cy - ch / 2) * H / ch]])
        im = cv2.warpAffine(fr[k], M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        bl = cv2.GaussianBlur(im, (0, 0), 2.2); im = cv2.addWeighted(im, 1.45, bl, -0.45, 0)
        small.append(cv2.resize(im, (54, 96), interpolation=cv2.INTER_AREA).astype(np.float32))
        g = grade_u8(im, s)
        cv2.imwrite(os.path.join(d, '%04d.jpg' % k), g, JPEG)
    # usable range: stop at hard cuts inside the source clip
    dif = np.array([np.abs(small[k + 1] - small[k]).mean() for k in range(n - 1)]) if n > 1 else np.zeros(0)
    k0 = int(round(PRE * FPS)); k1 = int(round((PRE + s['end'] - s['start']) * FPS))
    med = np.median(dif) if len(dif) else 0
    cuts = [k + 1 for k in range(n - 1) if dif[k] > 28 and dif[k] > 6 * med]
    lo = max([c for c in cuts if c <= k0] + [0]); hi = min([c for c in cuts if c > k0] + [n]) - 1
    if 'valid' in s: lo, hi = max(0, s['valid'][0]), min(n - 1, s['valid'][1])
    warn = ''
    if hi < k1: warn += ' WARNING: clip ends %.2f s before its slot does (frozen frame)' % ((k1 - hi) / FPS)
    if lo > k0: warn += ' WARNING: clip starts %.2f s late (frozen frame)' % ((lo - k0) / FPS)
    if nfaces < 4 and s.get('track', True): warn += ' note: no face tracked, crop fixed at x=%.2f' % s['x']
    return i, dict(n=n, lo=lo, hi=hi, k0=k0, k1=k1, graded=True, faces=nfaces, cuts=cuts), warn


def sheet(meta):
    rows = []
    for i, s in enumerate(SHOTS):
        m = meta.get(str(i)) or meta.get(i)
        if not m: continue
        row = []
        for k in (m['k0'], (m['k0'] + m['k1']) // 2, m['k1']):
            im = cv2.imread('clips/%02d/%04d.jpg' % (i, min(max(k, m['lo']), m['hi'])))
            row.append(cv2.resize(im, (180, 320), interpolation=cv2.INTER_AREA))
        cv2.putText(row[0], '%d %s' % (i, s['name']), (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        rows.append(np.hstack(row))
    blank = np.zeros_like(rows[0])
    out = np.vstack([np.hstack(rows[r:r + 4] + [blank] * (4 - len(rows[r:r + 4]))) for r in range(0, len(rows), 4)])
    os.makedirs('out/sheets', exist_ok=True); cv2.imwrite('out/sheets/clips.jpg', out, [cv2.IMWRITE_JPEG_QUALITY, 85])


if __name__ == '__main__':
    if not os.path.exists(FACE_MODEL):
        sys.exit('face model missing: %s (see the voidfx-edit skill for where to get it)' % FACE_MODEL)
    only = [int(x) for x in sys.argv[1:]] or list(range(len(SHOTS)))
    mp = 'clips/meta.json'
    meta = json.load(open(mp)) if os.path.exists(mp) and sys.argv[1:] else {}
    os.makedirs('clips', exist_ok=True)
    with Pool(min(len(only), os.cpu_count() or 4)) as pool:
        for i, m, warn in pool.imap_unordered(one, sorted(only, key=lambda i: -(SHOTS[i]['end'] - SHOTS[i]['start']))):
            meta[str(i)] = m
            print('%2d %-11s %3d frames, usable %3d..%3d, slot %3d..%3d%s' % (i, SHOTS[i]['name'], m['n'], m['lo'], m['hi'], m['k0'], m['k1'], warn), flush=True)
    json.dump(meta, open(mp, 'w'))
    sheet(meta); print('framing check: out/sheets/clips.jpg')
