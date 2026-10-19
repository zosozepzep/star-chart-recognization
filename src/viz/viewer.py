"""Browser artifacts in detector coordinates; display stretch never changes FITS."""
import json
import numpy as np
from matplotlib import image as mpl_image


def stretch(image):
    a = np.asarray(image, dtype=float)
    good = np.isfinite(a) & (a >= 0) & (a < 30000)
    values = a[good]
    if not len(values):
        return np.zeros(a.shape)
    low, high = np.percentile(values, [30, 99.7])
    return np.arcsinh(10 * np.clip((np.where(good, a, low) - low) / max(high-low, 1), 0, 1)) / np.arcsinh(10)


def export_viewer(sequence, run, output):
    """Store max-pooled overview and first/middle/last observed target cutouts."""
    reference = sequence.image(run.reference_frame)
    height, width = reference.shape
    stride = max(1, int(np.ceil(max(height, width) / 1024)))
    display = stretch(reference)
    pad_y, pad_x = (-height) % stride, (-width) % stride
    display = np.pad(display, ((0, pad_y), (0, pad_x)), constant_values=0)
    preview = display.reshape(display.shape[0]//stride, stride, display.shape[1]//stride, stride).max(axis=(1, 3))
    mpl_image.imsave(output / 'reference-preview.png', preview, cmap='gray', vmin=0, vmax=1, origin='upper')
    selections = {}
    for k, v in enumerate(run.targets, 1):
        for i in sorted({0, len(v.track.frames)//2, len(v.track.frames)-1}):
            selections.setdefault(v.track.frames[i], []).append((k, i, v))
    crops = {}
    for f, targets in selections.items():
        a = sequence.image(f)
        for k, i, verdict in targets:
            x, y = np.rint(verdict.track.xy_det[i]).astype(int)
            half = 60
            x0, x1 = max(0, x-half), min(width, x+half+1)
            y0, y1 = max(0, y-half), min(height, y+half+1)
            name = f'target-{k}-f{f}.png'
            mpl_image.imsave(output / name, stretch(a[y0:y1, x0:x1]), cmap='gray', vmin=0, vmax=1, origin='upper')
            crops.setdefault(k, []).append(dict(file=name, frame=f, x0=int(x0), y0=int(y0)))
    return dict(width=width, height=height, preview_width=preview.shape[1]*stride,
                preview_height=preview.shape[0]*stride, reference_frame=run.reference_frame,
                origin='upper', coordinates='detector pixels, 0-based x=column y=row',
                preview='reference-preview.png', target_cutouts=crops,
                note='显示采用非线性拉伸与最大值降采样；测量使用原始像素。轨迹为逐帧探测器坐标。')
