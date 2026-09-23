"""Competition-facing reference image and faintest confirmed-star annotation."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from src.viz.figures import setup_matplotlib, glyphs_missing, FontUnavailable


def plot_star_catalog(image, catalog, path):
    font = setup_matplotlib()
    title = "指定图像：跨帧核验的恒星候选"
    subtitle = "最暗可复现恒星候选"
    if glyphs_missing(font, title + subtitle + "仪器星等未定标没有满足条件的候选像素"):
        raise FontUnavailable("中文字体不覆盖比赛结果图的标签")
    stars = [r for r in catalog["sources"] if r["confirmed"]]
    faintest = catalog["faintest_confirmed_star"]
    fig, (ax, zoom) = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [1.3, 1]})
    values = np.asarray(image)[np.isfinite(image)]
    lo, hi = np.percentile(values, [1, 99.5])
    ax.imshow(image, cmap="gray", origin="upper", vmin=lo, vmax=max(hi, lo + 1))
    if stars:
        marker_size = 45 if len(stars) < 500 else 1.5
        ax.scatter([s["x"] for s in stars], [s["y"] for s in stars], s=marker_size,
                   facecolors="none", edgecolors="#52dfaa", linewidths=0.8 if len(stars) < 500 else 0.25,
                   alpha=1.0 if len(stars) < 500 else 0.6)
    ax.set_title(f"{title} {len(stars)} 个 / f{catalog['reference_frame']}")
    ax.set_xlabel("x（列，0 起算）")
    ax.set_ylabel("y（行，0 起算）")
    if faintest is not None:
        x, y = faintest["x"], faintest["y"]
        ax.scatter([x], [y], s=200, facecolors="none", edgecolors="#ffc857", linewidths=1.8)
        radius = 25
        x0, x1 = max(0, int(x) - radius), min(image.shape[1], int(x) + radius + 1)
        y0, y1 = max(0, int(y) - radius), min(image.shape[0], int(y) + radius + 1)
        patch = image[y0:y1, x0:x1]
        zoom.imshow(patch, cmap="gray", origin="upper", vmin=lo, vmax=max(hi, lo + 1),
                    extent=(x0 - .5, x1 - .5, y1 - .5, y0 - .5))
        zoom.scatter([x], [y], s=300, facecolors="none", edgecolors="#ffc857", linewidths=1.8)
        calibration_label = 'Gaia G 参照估计' if catalog.get('catalog_fit') else '真值定标'
        magnitude = (f"m ≈ {faintest['mag']:.2f}（{calibration_label}）" if faintest["mag"] is not None
                     else f"仪器星等 {faintest['instrumental_mag_rate']:.2f}（未定标）")
        zoom.set_title(f"{subtitle}\n{magnitude}")
        zoom.set_xlabel(f"x={x:.2f}, y={y:.2f}; {faintest['hits']}/{len(catalog['verification_frames'])} 帧")
    else:
        zoom.text(.5, .5, "没有满足条件的候选", ha="center", va="center", transform=zoom.transAxes)
        zoom.set_axis_off()
    fig.tight_layout()
    path = Path(path)
    try:
        fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.15)
    finally:
        plt.close(fig)
    return path


def plot_motion_tracks(sequence, run, path):
    """Overview and independent image cutouts at three epochs per candidate."""
    setup_matplotlib()
    count = len(run.targets)
    # Show a long track and the fastest distinct candidates; all tracks remain
    # present in the overview and CSV, without an excessively tall contact sheet.
    selected = list(dict.fromkeys(([0] if count else []) +
                                  sorted(range(count), key=lambda i: -np.linalg.norm(run.targets[i].velocity))))[:3]
    shown = len(selected)
    fig = plt.figure(figsize=(15, max(7, 2.8 * shown)))
    grid = fig.add_gridspec(max(shown, 1), 4, width_ratios=[2.1, 1, 1, 1])
    ax = fig.add_subplot(grid[:, 0])
    image = sequence.image(run.reference_frame)
    lo, hi = np.percentile(image, [1, 99.5])
    ax.imshow(image, origin='upper', cmap='gray', vmin=lo, vmax=hi)
    ax.set_title(f'官方天基序列：{count} 条运动候选轨迹')
    ax.set_xlabel('x（列，0 起算）'); ax.set_ylabel('y（行，0 起算）')
    colors = plt.get_cmap('tab10')
    for k, verdict in enumerate(run.targets):
        track = verdict.track
        color = colors(k % 10)
        ax.plot(*track.xy_det.T, 'o-', ms=3, lw=1.4, color=color)
        ax.annotate(f'T{k+1}', track.xy_det[len(track.frames)//2], color=color,
                    xytext=(8, 8), textcoords='offset points', weight='bold')
        if k not in selected:
            continue
        for col, index in enumerate([0, len(track.frames)//2, len(track.frames)-1], 1):
            f = track.frames[index]
            x, y = track.xy_det[index]
            radius = max(24, min(80, int(np.sqrt(track.elongation[index]) * 10)))
            x0, x1 = max(0, int(x)-radius), min(image.shape[1], int(x)+radius+1)
            y0, y1 = max(0, int(y)-radius), min(image.shape[0], int(y)+radius+1)
            patch = sequence.image(f)[y0:y1, x0:x1]
            zoom = fig.add_subplot(grid[selected.index(k), col])
            zoom.imshow(patch, origin='upper', cmap='gray', vmin=lo, vmax=max(hi, lo+1),
                        extent=(x0-.5, x1-.5, y1-.5, y0-.5))
            zoom.scatter([x], [y], s=150, facecolors='none', edgecolors=[color])
            zoom.set_title(f'T{k+1} / f{f}\n{x:.1f}, {y:.1f}', fontsize=10)
            zoom.tick_params(labelsize=7)
    if not count:
        empty = fig.add_subplot(grid[:, 1:]); empty.set_axis_off()
        empty.text(.5, .5, '没有满足多帧判据的运动候选', ha='center')
    fig.tight_layout()
    try:
        fig.savefig(path, dpi=150, bbox_inches='tight')
    finally:
        plt.close(fig)
