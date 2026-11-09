"""render.py frames N0 N1 [workers]   -> out/frames/%05d.jpg
   render.py sheet NAME t0 t1 step     -> out/sheets/NAME.jpg contact sheet for checking"""
import sys, os, time
import numpy as np, cv2
from multiprocessing import Pool


def render_range(args):
    n0, n1 = args
    from config import FPS
    from timeline import frame
    from fx import finish
    for n in range(n0, n1):
        cv2.imwrite('out/frames/%05d.jpg' % n, finish(frame(n / FPS), n), [cv2.IMWRITE_JPEG_QUALITY, 96])
    return n1 - n0


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'frames':
        n0, n1 = int(sys.argv[2]), int(sys.argv[3]); workers = int(sys.argv[4]) if len(sys.argv) > 4 else 4
        os.makedirs('out/frames', exist_ok=True)
        step = 24
        chunks = [(a, min(a + step, n1)) for a in range(n0, n1, step)]
        t0 = time.time(); done = 0
        with Pool(workers) as p:
            for k in p.imap_unordered(render_range, chunks):
                done += k
                print('\r%d/%d  %.1fs' % (done, n1 - n0, time.time() - t0), end='', flush=True)
        print()
    else:
        from config import FPS
        from timeline import frame
        from fx import finish
        name = sys.argv[2]; a, b, st = float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
        cols = int(sys.argv[6]) if len(sys.argv) > 6 else 10
        ts = np.arange(a, b + 1e-6, st); ims = []
        for t in ts:
            n = int(round(t * FPS)); im = cv2.resize(finish(frame(n / FPS), n), (216, 384), interpolation=cv2.INTER_AREA)
            cv2.rectangle(im, (0, 0), (74, 20), (0, 0, 0), -1)
            cv2.putText(im, '%.3f' % (n / FPS), (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            ims.append(im)
        while len(ims) % cols: ims.append(np.zeros_like(ims[0]))
        os.makedirs('out/sheets', exist_ok=True); cv2.imwrite('out/sheets/%s.jpg' % name, np.vstack([np.hstack(ims[r:r + cols]) for r in range(0, len(ims), cols)]), [cv2.IMWRITE_JPEG_QUALITY, 85])
