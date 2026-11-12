"""Cut every planned shot out of its scene pack as a face-tracked, graded 9:16 clip.

  python extract.py            # all shots (parallel, one process per core)
  python extract.py 3 7        # only these shots, e.g. after changing their src, framing or grade
  python extract.py --sr [ids] # ML-upscale (Real-ESRGAN, see sr.py) the frames that reach the screen; slow, sequential

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
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))

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
    det = _det[(SW, SH)]; n = len(fr); xs = np.full(n, np.nan); ys = np.full(n, np.nan); hs = np.full(n, np.nan)
    for k in range(0, n, 3):
        _, faces = det.detect(fr[k])
        if faces is not None and len(faces):
            c = [(f[0] + f[2] / 2) / SW for f in faces]
            j = int(np.argmin([abs(v - s['x']) for v in c]))
            if abs(c[j] - s['x']) < 0.16 + (0.14 if s.get('frame') == 'auto' else 0):
                xs[k] = c[j]; ys[k] = (faces[j][1] + faces[j][3] / 2) / SH; hs[k] = faces[j][3] / SH
    good = ~np.isnan(xs)
    track.last = (ys, hs, good)
    if s.get('track', True) and good.sum() >= 4:
        idx = np.arange(n); x = np.interp(idx, idx[good], xs[good])
        if n > 30: x = cv2.GaussianBlur(x.reshape(-1, 1), (1, 0), 0, sigmaY=14).ravel()
        if s.get('frame') != 'auto': x = 0.7 * x + 0.3 * s['x']            # stay close to the planned composition
    else:
        x = np.full(n, s['x'])
    return x, int(good.sum())


SR_PRE, SR_POST = 27, 30                     # frames around the slot that transitions can show


def _write(path, img):
    tmp = path + '.tmp.jpg'; cv2.imwrite(tmp, img, JPEG); os.replace(tmp, path)   # atomic: previews never read half a file


def one(i, use_sr=False):
    cv2.setNumThreads(1 if not use_sr else 2)
    from fx import grade_u8                   # imported here so workers don't need clips/meta.json
    s = SHOTS[i]; path = s.get('pack', SCENEPACK); SW, SH = probe(path)
    fr = read(path, s['src'] - PRE, (s['end'] - s['start']) + PRE + POST, SW, SH); n = len(fr)
    if n == 0: raise RuntimeError('shot %d (%s): no frames at src=%.2f in %s' % (i, s['name'], s['src'], path))
    x, nfaces = track(fr, s, SW, SH)
    ch = SH / s['z']; cw = ch * W / H         # full-height 9:16 window, tightened by z
    cyv = np.full(n, s.get('cy', .5) * SH)
    ys, hs, good = track.last
    if s.get('frame') == 'auto' and good.sum() >= 4:
        # auto framing: face about 30% of the crop height, eyes in the upper third, one steady zoom per shot
        fh = float(np.median(hs[good])) * SH
        ch = float(np.clip(fh / s.get('face_frac', 0.30), SH / s.get('zmax', 1.5), SH)); cw = ch * W / H
        idx = np.arange(n); yv = np.interp(idx, idx[good], ys[good]) * SH
        if n > 30: yv = cv2.GaussianBlur(yv.reshape(-1, 1), (1, 0), 0, sigmaY=18).ravel()
        cyv = yv + 0.10 * ch
    cx = np.clip(x * SW, cw / 2, SW - cw / 2); cyv = np.clip(cyv, ch / 2, SH - ch / 2)
    d = os.path.join('clips', '%02d' % i); os.makedirs(d, exist_ok=True)
    for f in os.listdir(d):
        if not f.endswith('.jpg') or not f[:4].isdigit() or int(f[:4]) >= n: os.remove(os.path.join(d, f))
    k0 = int(round(PRE * FPS)); k1 = int(round((PRE + s['end'] - s['start']) * FPS))
    sr_lo, sr_hi = (max(0, k0 - SR_PRE), min(n - 1, k1 + SR_POST)) if use_sr else (1, 0)
    if use_sr:                                # one source region covering every crop position in the range
        rx0 = int(max(0, np.floor(cx[sr_lo:sr_hi + 1].min() - cw / 2) - 2)); rx1 = int(min(SW, np.ceil(cx[sr_lo:sr_hi + 1].max() + cw / 2) + 2))
        ry0 = int(max(0, np.floor(cyv[sr_lo:sr_hi + 1].min() - ch / 2) - 2)); ry1 = int(min(SH, np.ceil(cyv[sr_lo:sr_hi + 1].max() + ch / 2) + 2))
        sc = W / cw / 4; prev = None; nsr = 0
    small = []
    for k in range(n):
        if sr_lo <= k <= sr_hi:
            reg = fr[k][ry0:ry1, rx0:rx1]; key = cv2.resize(reg, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32)
            # reuse only for true repeats (24 fps material in a 60 fps file): compare with the PREVIOUS source frame,
            # not the last upscaled one, or slow movement accumulates under the threshold and stutters
            if prev is None or np.abs(key - prev[0]).mean() > 0.10:
                up = sr.upscale4(reg); nsr += 1
                big = cv2.resize(up, (int(round(up.shape[1] * sc)), int(round(up.shape[0] * sc))), interpolation=cv2.INTER_AREA)
                prev = (key, big)
            else:
                prev = (key, prev[1])
            big = prev[1]; ax = big.shape[1] / (4 * (rx1 - rx0)); ay = big.shape[0] / (4 * (ry1 - ry0))
            f_ = (W / cw) / (4 * ax); g_ = (H / ch) / (4 * ay)
            M = np.float32([[f_, 0, (rx0 - (cx[k] - cw / 2)) * W / cw], [0, g_, (ry0 - (cyv[k] - ch / 2)) * H / ch]])
            im = cv2.warpAffine(big, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            bl = cv2.GaussianBlur(im, (0, 0), 1.6); im = cv2.addWeighted(im, 1.18, bl, -0.18, 0)
        else:
            M = np.float32([[W / cw, 0, -(cx[k] - cw / 2) * W / cw], [0, H / ch, -(cyv[k] - ch / 2) * H / ch]])
            im = cv2.warpAffine(fr[k], M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            bl = cv2.GaussianBlur(im, (0, 0), 2.2); im = cv2.addWeighted(im, 1.45, bl, -0.45, 0)
        small.append(cv2.resize(im, (54, 96), interpolation=cv2.INTER_AREA).astype(np.float32))
        g = grade_u8(im, s)
        _write(os.path.join(d, '%04d.jpg' % k), g)
    # usable range: stop at hard cuts inside the source clip
    dif = np.array([np.abs(small[k + 1] - small[k]).mean() for k in range(n - 1)]) if n > 1 else np.zeros(0)
    med = np.median(dif) if len(dif) else 0
    cuts = [k + 1 for k in range(n - 1) if dif[k] > 28 and dif[k] > 6 * med]
    lo = max([c for c in cuts if c <= k0] + [0]); hi = min([c for c in cuts if c > k0] + [n]) - 1
    if 'valid' in s: lo, hi = max(0, s['valid'][0]), min(n - 1, s['valid'][1])
    warn = ''
    if hi < k1: warn += ' WARNING: clip ends %.2f s before its slot does (frozen frame)' % ((k1 - hi) / FPS)
    if lo > k0: warn += ' WARNING: clip starts %.2f s late (frozen frame)' % ((lo - k0) / FPS)
    if nfaces < 4 and s.get('track', True): warn += ' note: no face tracked, crop fixed at x=%.2f' % s['x']
    if use_sr: warn += ' (ML-upscaled %d unique frames in %d..%d)' % (nsr, sr_lo, sr_hi)
    return i, dict(n=n, lo=lo, hi=hi, k0=k0, k1=k1, graded=True, faces=nfaces, cuts=cuts, sr=bool(use_sr)), warn


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
    args = [a for a in sys.argv[1:] if a != '--sr']; use_sr = '--sr' in sys.argv[1:]
    only = [int(x) for x in args] or list(range(len(SHOTS)))
    mp = 'clips/meta.json'
    meta = json.load(open(mp)) if os.path.exists(mp) and (args or use_sr) else {}
    os.makedirs('clips', exist_ok=True)
    report = lambda i, m, warn: print('%2d %-11s %3d frames, usable %3d..%3d, slot %3d..%3d%s' % (i, SHOTS[i]['name'], m['n'], m['lo'], m['hi'], m['k0'], m['k1'], warn), flush=True)
    if use_sr:
        import sr
        path = globals().get('SR_MODEL')
        if not sr.available(path): sys.exit('--sr needs torch and the weights at SR_MODEL (%s); see edit/sr.py' % path)
        sr.load(path)
        for i in only:                        # sequential: torch already uses every core
            i, m, warn = one(i, True); meta[str(i)] = m; report(i, m, warn)
            json.dump(meta, open(mp + '.tmp', 'w')); os.replace(mp + '.tmp', mp)
    else:
        with Pool(min(len(only), os.cpu_count() or 4)) as pool:
            for i, m, warn in pool.imap_unordered(one, sorted(only, key=lambda i: -(SHOTS[i]['end'] - SHOTS[i]['start']))):
                meta[str(i)] = m; report(i, m, warn)
    json.dump(meta, open(mp + '.tmp', 'w')); os.replace(mp + '.tmp', mp)
    sheet(meta); print('framing check: out/sheets/clips.jpg')
