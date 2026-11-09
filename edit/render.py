"""Render the edit.

  python render.py video                 # final: out/edit.mp4, 1080x1920 60 fps
  python render.py video --draft         # quick review: every other frame (30 fps), fast encode
  python render.py video --out NAME.mp4  # choose the file name (written in out/)
  python render.py sheet NAME t0 t1 step [cols]   # out/sheets/NAME.jpg contact sheet to check a span

Frames go straight from the render workers into ffmpeg (no intermediate image files), in order,
with a bounded queue so memory stays flat. The soundtrack is rebuilt first when the plan changed.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.getcwd())   # run from an edit folder: its config.py wins, engine modules come from edit/
import collections, os, subprocess, sys, time
from multiprocessing import Pool
import numpy as np, cv2

FINAL = ['-c:v', 'libx264', '-preset', 'medium', '-tune', 'film', '-crf', '18', '-maxrate', '30M', '-bufsize', '60M']
DRAFT = ['-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23']


def render_chunk(frames):
    cv2.setNumThreads(1)
    from config import FPS
    from timeline import frame
    from fx import finish
    return b''.join(finish(frame(n / FPS), n).tobytes() for n in frames)


def ensure_audio():
    eng = os.path.dirname(os.path.abspath(__file__))
    wav, srcs = 'out/audio.wav', ['config.py', os.path.join(eng, 'audio.py'), os.path.join(eng, 'timing.py')]
    if not os.path.exists(wav) or os.path.getmtime(wav) < max(os.path.getmtime(f) for f in srcs):
        subprocess.run([sys.executable, os.path.join(eng, 'audio.py')], check=True)
    return wav


def video(out, draft=False, workers=None):
    from config import W, H, FPS, DUR
    os.makedirs('out', exist_ok=True)
    step = 2 if draft else 1; fps = FPS // step
    frames = list(range(0, int(round(DUR * FPS)), step))
    wav = ensure_audio()
    cmd = ['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', '%dx%d' % (W, H), '-r', str(fps), '-i', '-',
           '-i', wav] + (DRAFT if draft else FINAL) + ['-pix_fmt', 'yuv420p', '-color_primaries', 'bt709', '-color_trc', 'bt709',
           '-colorspace', 'bt709', '-c:a', 'aac', '-b:a', '256k', '-shortest', '-movflags', '+faststart', out]
    enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    workers = workers or os.cpu_count() or 4
    chunks = [frames[i:i + 12] for i in range(0, len(frames), 12)]
    t0 = time.time(); done = 0; pending = collections.deque()
    with Pool(workers) as pool:
        it = iter(chunks)
        for _ in range(workers * 2):
            c = next(it, None)
            if c: pending.append((len(c), pool.apply_async(render_chunk, (c,))))
        while pending:
            n, r = pending.popleft(); enc.stdin.write(r.get()); done += n
            c = next(it, None)
            if c: pending.append((len(c), pool.apply_async(render_chunk, (c,))))
            print('\r%d/%d frames  %.0fs' % (done, len(frames), time.time() - t0), end='', flush=True)
    enc.stdin.close()
    if enc.wait() != 0: sys.exit('ffmpeg failed')
    print('\n%s  (%.0f s, %.1f MB)' % (out, time.time() - t0, os.path.getsize(out) / 1e6))


def sheet(name, a, b, st, cols=10):
    from config import FPS
    from timeline import frame
    from fx import finish
    ims = []
    for t in np.arange(a, b + 1e-6, st):
        n = int(round(t * FPS)); im = cv2.resize(finish(frame(n / FPS), n), (216, 384), interpolation=cv2.INTER_AREA)
        cv2.rectangle(im, (0, 0), (74, 20), (0, 0, 0), -1)
        cv2.putText(im, '%.3f' % (n / FPS), (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        ims.append(im)
    while len(ims) % cols: ims.append(np.zeros_like(ims[0]))
    os.makedirs('out/sheets', exist_ok=True)
    cv2.imwrite('out/sheets/%s.jpg' % name, np.vstack([np.hstack(ims[r:r + cols]) for r in range(0, len(ims), cols)]), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print('out/sheets/%s.jpg' % name)


if __name__ == '__main__':
    args = sys.argv[1:]
    if args and args[0] == 'video':
        draft = '--draft' in args
        name = args[args.index('--out') + 1] if '--out' in args else ('edit_draft.mp4' if draft else 'edit.mp4')
        video(os.path.join('out', os.path.basename(name)), draft)
    elif args and args[0] == 'sheet':
        sheet(args[1], float(args[2]), float(args[3]), float(args[4]), int(args[5]) if len(args) > 5 else 10)
    else:
        sys.exit(__doc__)
