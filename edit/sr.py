"""Optional ML upscaling for extract.py: Real-ESRGAN realesr-general-x4v3 (SRVGGNetCompact) on CPU.

Needs `pip install torch` (CPU wheel) and the weights file, e.g.
  curl -L -o media/models/realesr-general-x4v3.pth \
    https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth
Weights are loaded with torch.load(weights_only=True): tensors only, no pickled code runs.
About 3.5 s per 405x720 crop on 4 CPU cores, so extract.py only upscales unique frames inside each
shot's slot plus its transition margins.
"""
import numpy as np

_model = None


def available(path):
    import os
    if not path or not os.path.exists(path): return False
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def _build():
    import torch.nn as nn, torch.nn.functional as F

    class SRVGG(nn.Module):
        def __init__(s, nf=64, nc=32, up=4):
            super().__init__(); s.up = up
            b = [nn.Conv2d(3, nf, 3, 1, 1), nn.PReLU(num_parameters=nf)]
            for _ in range(nc): b += [nn.Conv2d(nf, nf, 3, 1, 1), nn.PReLU(num_parameters=nf)]
            b += [nn.Conv2d(nf, 3 * up * up, 3, 1, 1)]
            s.body = nn.ModuleList(b); s.ps = nn.PixelShuffle(up)

        def forward(s, x):
            o = x
            for m in s.body: o = m(o)
            return s.ps(o) + F.interpolate(x, scale_factor=s.up, mode='nearest')
    return SRVGG()


def load(path, threads=4):
    global _model
    if _model is None:
        import torch
        torch.set_num_threads(threads)
        sd = torch.load(path, map_location='cpu', weights_only=True)
        sd = sd.get('params', sd.get('params_ema', sd))
        m = _build(); m.load_state_dict(sd, strict=True); _model = m.eval()
    return _model


def upscale4(bgr):
    """uint8 BGR -> uint8 BGR at 4x."""
    import torch
    x = torch.from_numpy(np.ascontiguousarray(bgr[..., ::-1])).permute(2, 0, 1)[None].float() / 255
    with torch.inference_mode():
        y = _model(x)
    return (y[0].permute(1, 2, 0).clamp(0, 1).numpy()[..., ::-1] * 255 + .5).astype(np.uint8)
