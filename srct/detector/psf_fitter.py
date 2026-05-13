from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit

from detector.feature_extractor import SourceDetection


def gaussian_2d(coords, amplitude, x0, y0, sigma_x, sigma_y, background):
    x, y = coords
    model = amplitude * np.exp(
        -(((x - x0) ** 2) / (2 * sigma_x ** 2)
          + ((y - y0) ** 2) / (2 * sigma_y ** 2))
    ) + background
    return model.ravel()


class PSFFitter:
    def __init__(self, patch_size: int = 11):
        self.patch_size = patch_size

    def fit_sources(self, data, detections):
        total = len(detections)
        for i, det in enumerate(detections):
            # 每处理 500 个点打印一次进度
            if (i + 1) % 500 == 0 or i == total - 1:
                print(f"   -> [PSF 拟合进度] {i + 1} / {total} ...")
            
            self.fit_single(data, det)
        return detections

    def fit_single(self, data, det: SourceDetection):
        patch, x0_global, y0_global = self._extract_patch(data, det.x, det.y)
        if patch is None:
            det.fit_success = False
            det.flags["psf_patch_failed"] = True
            return

        ny, nx = patch.shape
        y, x = np.mgrid[:ny, :nx]

        amplitude0 = patch.max() - np.median(patch)
        background0 = np.median(patch)
        x0 = nx / 2
        y0 = ny / 2

        p0 = [amplitude0, x0, y0, 1.5, 1.5, background0]

        bounds = (
            [0, 0, 0, 0.3, 0.3, -np.inf],
            [np.inf, nx, ny, 10.0, 10.0, np.inf],
        )

        try:
            popt, _ = curve_fit(
                gaussian_2d,
                (x, y),
                patch.ravel(),
                p0=p0,
                bounds=bounds,
                maxfev=200,
            )

            amp, xf, yf, sx, sy, bg = popt

            det.x = x0_global + xf
            det.y = y0_global + yf

            det.fwhm_x = 2.3548 * sx
            det.fwhm_y = 2.3548 * sy
            det.fwhm = np.sqrt(det.fwhm_x * det.fwhm_y)

            a = max(det.fwhm_x, det.fwhm_y)
            b = min(det.fwhm_x, det.fwhm_y)

            det.elongation = a / b if b > 0 else np.inf
            det.ellipticity = 1 - b / a if a > 0 else np.nan

            model = gaussian_2d((x, y), *popt).reshape(patch.shape)
            residual = patch - model

            det.chi2 = float(np.mean(residual ** 2))
            denom = np.sum(np.abs(patch)) + 1e-8
            det.residual_ratio = float(np.sum(np.abs(residual)) / denom)

            det.fit_success = True

        except Exception as e:
            det.fit_success = False
            det.flags["psf_fit_failed"] = True
            det.flags["psf_error"] = str(e)

    def _extract_patch(self, data, x, y):
        half_size = self.patch_size // 2
        ix, iy = int(round(x)), int(round(y))
        
        y0, y1 = iy - half_size, iy + half_size + 1
        x0, x1 = ix - half_size, ix + half_size + 1
        
        if y0 < 0 or y1 > data.shape[0] or x0 < 0 or x1 > data.shape[1]:
            return None, 0, 0
            
        patch = data[y0:y1, x0:x1].astype(np.float64)
        return patch, x0, y0