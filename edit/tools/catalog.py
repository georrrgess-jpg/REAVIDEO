"""Catalog a scene pack: detect shots and write contact sheets to pick clips from.

  python catalog.py SCENEPACK.mp4 [-o catalog]

Writes catalog/shots.json ([start, end] per shot) and catalog/sheet_N.jpg (40 shots per
sheet, two frames each, labelled #index and time). Then look closer at candidates with
  python catalog.py SCENEPACK.mp4 --strip 12 40 57
which shows 8 timestamped frames across each listed shot, for choosing in-points.
"""
import argparse, json, os, subprocess
import numpy as np, cv2


def detect(path, fps=30, W=160, H=90):
    p = subprocess.Popen(['ffmpeg', '-v', 'error', '-i', path, '-vf', f'fps={fps},scale={W}:{H}', '-f', 'rawvideo',
                          '-pix_fmt', 'bgr24', '-'], stdout=subprocess.PIPE)
    prev = None; cuts = [0]; n = 0
    while True:
        b = p.stdout.read(W * H * 3)
        if len(b) < W * H * 3: break
        f = np.frombuffer(b, np.uint8).reshape(H, W, 3)
        h = cv2.calcHist([cv2.cvtColor(f, cv2.COLOR_BGR2HSV)], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256]).flatten(); h /= h.sum()
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if prev is not None:
            dh = 0.5 * np.abs(h - prev[0]).sum(); dg = np.abs(g - prev[1]).mean()
            if (dh > 0.45 or dg > 40) and n - cuts[-1] >= 6: cuts.append(n)
        prev = (h, g); n += 1
    cuts.append(n)
    return [(cuts[i] / fps, cuts[i + 1] / fps) for i in range(len(cuts) - 1)]


def sheets(path, shots, out, ids=None, frames=2, per=40, cols=4):
    cap = cv2.VideoCapture(path); tw, th = 240, 135
    ids = list(range(len(shots))) if ids is None else ids

    def grab(t):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000); ok, f = cap.read()
        return cv2.resize(f, (tw, th)) if ok else np.zeros((th, tw, 3), np.uint8)
    names = []
    for s0 in range(0, len(ids), per):
        sub = ids[s0:s0 + per]; rows = (len(sub) + cols - 1) // cols
        sheet = np.zeros((rows * th, cols * frames * tw, 3), np.uint8)
        for k, i in enumerate(sub):
            a, b = shots[i]; r, c = divmod(k, cols)
            for j in range(frames):
                t = a + 0.08 + (b - a - 0.16) * (j + 0.5) / frames; im = grab(t)
                label = ('#%d ' % i if j == 0 else '') + '%.2f' % t
                cv2.rectangle(im, (0, 0), (130, 20), (0, 0, 0), -1)
                cv2.putText(im, label, (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
                sheet[r * th:(r + 1) * th, (c * frames + j) * tw:(c * frames + j + 1) * tw] = im
        name = os.path.join(out, '%s_%d.jpg' % ('sheet' if frames == 2 else 'strip', s0 // per)); names.append(name)
        cv2.imwrite(name, sheet, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return names


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('video'); ap.add_argument('-o', default='catalog')
    ap.add_argument('--strip', type=int, nargs='+', help='shot indexes to show as 8-frame strips')
    a = ap.parse_args(); os.makedirs(a.o, exist_ok=True); jp = os.path.join(a.o, 'shots.json')
    if a.strip:
        shots = json.load(open(jp))
        print(sheets(a.video, shots, a.o, ids=a.strip, frames=8, per=13, cols=1))
    else:
        shots = detect(a.video); json.dump(shots, open(jp, 'w'))
        print(len(shots), 'shots ->', jp); print(sheets(a.video, shots, a.o))
