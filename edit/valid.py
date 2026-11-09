import numpy as np, cv2, json, glob
from config import *
meta = {}
rows = []
for i, s in enumerate(SHOTS):
    fs = sorted(glob.glob('clips/%02d/*.jpg' % i)); n = len(fs)
    sm = [cv2.resize(cv2.imread(f), (54, 96)).astype(np.float32) for f in fs]
    d = np.array([np.abs(sm[k + 1] - sm[k]).mean() for k in range(n - 1)])
    k0 = int(round(PRE * FPS)); k1 = int(round((PRE + s['end'] - s['start']) * FPS))
    cuts = [k + 1 for k in range(n - 1) if d[k] > 28 and d[k] > 6 * np.median(d)]
    lo = max([c for c in cuts if c <= k0] + [0]); hi = min([c for c in cuts if c > k0] + [n]) - 1
    if 'valid' in s: lo, hi = max(0, s['valid'][0]), min(n - 1, s['valid'][1])
    meta[i] = dict(n=n, lo=lo, hi=hi, k0=k0, k1=k1)
    print('%2d %-11s n=%3d valid %3d..%3d  slot %3d..%3d  %s' % (i, s['name'], n, lo, hi, k0, k1, 'CUTS ' + str(cuts) if cuts else ''))
    row = [cv2.resize(cv2.imread(fs[min(max(k, lo), hi)]), (180, 320)) for k in (k0, (k0 + k1) // 2, k1)]
    cv2.putText(row[0], '%d %s' % (i, s['name']), (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    rows.append(np.hstack(row))
json.dump(meta, open('clips/meta.json', 'w'))
sheet = np.vstack([np.hstack(rows[r:r + 4] + [np.zeros_like(rows[0])] * (4 - len(rows[r:r + 4]))) for r in range(0, len(rows), 4)])
import os; os.makedirs('out/sheets', exist_ok=True); cv2.imwrite('out/sheets/clips.jpg', sheet, [cv2.IMWRITE_JPEG_QUALITY, 82]); print(sheet.shape)
