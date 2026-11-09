import numpy as np, cv2, subprocess, os, sys, json
from config import *
SRC = '../media/scenepack.mp4'; SW, SH = 1280, 720
det = cv2.FaceDetectorYN.create(os.path.expanduser('~/.cache/rea-video/models/face_detection_yunet_2023mar.onnx'), '', (SW, SH), 0.55, 0.3, 50)
def read(a, d):
    p = subprocess.Popen(['ffmpeg', '-v', 'error', '-ss', '%.4f' % a, '-i', SRC, '-t', '%.4f' % d, '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-'], stdout=subprocess.PIPE)
    out = []
    while True:
        b = p.stdout.read(SW * SH * 3)
        if len(b) < SW * SH * 3: break
        out.append(np.frombuffer(b, np.uint8).reshape(SH, SW, 3))
    return out
only = [int(x) for x in sys.argv[1:]] or range(len(SHOTS))
meta = {}
for i in only:
    s = SHOTS[i]; a = s['src'] - PRE; d = (s['end'] - s['start']) + PRE + POST
    fr = read(a, d); n = len(fr)
    cw = 405 / s['z']; ch = 720 / s['z']
    xs = np.full(n, np.nan)
    for k in range(0, n, 3):
        _, faces = det.detect(fr[k])
        if faces is not None and len(faces):
            c = [(f[0] + f[2] / 2) / SW for f in faces]
            j = int(np.argmin([abs(v - s['x']) for v in c]))
            if abs(c[j] - s['x']) < 0.16: xs[k] = c[j]
    good = ~np.isnan(xs)
    if s.get('track', True) and good.sum() >= 4:
        idx = np.arange(n); x = np.interp(idx, idx[good], xs[good])
        x = cv2.GaussianBlur(x.reshape(-1, 1), (1, 0), 0, sigmaY=14).ravel() if n > 30 else x
        # blend toward the hint so framing stays close to the planned composition
        x = 0.7 * x + 0.3 * s['x']
    else:
        x = np.full(n, s['x'])
    cx = np.clip(x * SW, cw / 2, SW - cw / 2); cy = np.clip(s.get('cy', .5) * SH, ch / 2, SH - ch / 2)
    os.makedirs('clips/%02d' % i, exist_ok=True)
    for k in range(n):
        M = np.float32([[W / cw, 0, -(cx[k] - cw / 2) * W / cw], [0, H / ch, -(cy - ch / 2) * H / ch]])
        im = cv2.warpAffine(fr[k], M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        bl = cv2.GaussianBlur(im, (0, 0), 2.2); im = cv2.addWeighted(im, 1.45, bl, -0.45, 0)
        cv2.imwrite('clips/%02d/%04d.jpg' % (i, k), im, [cv2.IMWRITE_JPEG_QUALITY, 93])
    meta[i] = dict(n=n, faces=int(good.sum()), x0=float(x[0]), x1=float(x[-1]))
    print(i, s['name'], n, 'frames; faces', int(good.sum()), 'x %.2f->%.2f' % (x[0], x[-1]), flush=True)
