"""FITS 序列读取。

本数据集的头部有三处非标准写法，必须专门处理：
  AZIMUTH = '0270.28456 0000.00000'   # 字符串，两个字段，取第一个
  ELEVATIO= '0015.24484 0000.00000'   # 同上
  EXPOSURE= '0000          00080'     # 字符串，毫秒在最后一个字段
因为这些卡片不合规，fits.open() 之后必须 verify('silentfix')，否则读头即抛异常。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.time import Time

_NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def _numbers(value) -> list[float]:
    if isinstance(value, (int, float)):
        return [float(value)]
    found = _NUMBER.findall(str(value))
    if not found:
        raise ValueError(f"无法从卡片值中解析出数字: {value!r}")
    return [float(x) for x in found]


def parse_pointing(value) -> float:
    """取双字段指向卡片的第一个数（度）。"""
    return _numbers(value)[0]


def parse_exposure_ms(value) -> float:
    """取曝光卡片的最后一个数（毫秒）。"""
    return _numbers(value)[-1]


@dataclass(frozen=True)
class FrameHeader:
    index: int
    path: Path
    date_obs: Time
    azimuth_deg: float
    elevation_deg: float
    exposure_ms: float
    site_lon_deg: float
    site_lat_deg: float
    site_alt_m: float
    space_aux: dict | None = None

    @property
    def exposure_s(self) -> float:
        return self.exposure_ms / 1000.0


def read_header(path: str | Path, index: int) -> FrameHeader:
    path = Path(path)
    with fits.open(path) as hdul:
        hdul.verify("silentfix")
        hdr = hdul[0].header
        auxiliary = None
        # Zero site coordinates alone are insufficient: require a physically
        # valid telemetry payload and matching platform pointing as well.
        if all(float(hdr.get(k, float('nan'))) == 0 for k in ('SITELONG', 'SITELATI', 'SITEALTI')):
            from src.dataio.space_metadata import decode_auxiliary
            with path.open('rb') as fh:
                fh.seek(hdul[0].fileinfo()['datLoc'])
                raw = fh.read(208)
            try:
                auxiliary = decode_auxiliary(raw)
            except ValueError:
                pass
            if auxiliary is not None and not np.allclose(
                    [auxiliary['p_az'], auxiliary['p_el']],
                    [parse_pointing(hdr['AZIMUTH']), parse_pointing(hdr['ELEVATIO'])], atol=1e-4):
                raise ValueError('天基首行指向与 FITS 头不一致')
        return FrameHeader(
            index=index,
            path=path,
            date_obs=Time(str(hdr["DATE-OBS"]).strip(), format="isot", scale="utc"),
            azimuth_deg=parse_pointing(hdr["AZIMUTH"]),
            elevation_deg=parse_pointing(hdr["ELEVATIO"]),
            exposure_ms=parse_exposure_ms(hdr["EXPOSURE"]),
            site_lon_deg=float(hdr["SITELONG"]),
            site_lat_deg=float(hdr["SITELATI"]),
            site_alt_m=float(hdr["SITEALTI"]),
            space_aux=auxiliary,
        )


def load_image(path: str | Path) -> np.ndarray:
    """读取像素数据为 float64。

    BITPIX=16 + BZERO=0 → astropy 返回有符号 >i2（上限 32767）；
    历史地基数据实测最大 28190；官方天基数据确实包含负值，保留原始
    有符号含义。天基分析分支单独屏蔽异常像素，绝不无依据地转无符号。
    """
    with fits.open(Path(path)) as hdul:
        hdul.verify("silentfix")
        return np.asarray(hdul[0].data, dtype=np.float64)


@dataclass
class FrameSequence:
    dataset_id: str
    directory: Path
    headers: list[FrameHeader]
    truth_path: Path | None

    @classmethod
    def from_directory(cls, directory: str | Path) -> "FrameSequence":
        directory = Path(directory)
        files = sorted(directory.glob("*.fits")) + sorted(directory.glob("*.FITS"))
        files = sorted(set(files))
        if not files:
            raise FileNotFoundError(f"目录中没有 FITS 文件: {directory}")
        headers = [read_header(p, i) for i, p in enumerate(files)]
        # stable 排序：DATE-OBS 相同时保留上面按文件名建立的次序
        order = np.argsort([h.date_obs.unix for h in headers], kind="stable")
        headers = [
            replace(headers[old], index=new_index)
            for new_index, old in enumerate(order)
        ]
        dats = sorted(directory.glob("*.DAT")) + sorted(directory.glob("*.dat"))
        dats = sorted(set(dats))
        if len(dats) > 1:
            raise ValueError(
                f"目录中存在多个真值文件，无法判定使用哪一个: {[p.name for p in dats]}"
            )
        return cls(
            dataset_id=directory.name,
            directory=directory,
            headers=headers,
            truth_path=dats[0] if dats else None,
        )

    def __len__(self) -> int:
        return len(self.headers)

    @property
    def observation_mode(self) -> str:
        flags = [h.space_aux is not None for h in self.headers]
        if any(flags) and not all(flags):
            raise ValueError('序列混合天基与地基格式，或部分帧的辅助数据损坏')
        return 'space' if all(flags) and flags else 'ground'

    def image(self, index: int) -> np.ndarray:
        return load_image(self.headers[index].path)

    def times(self) -> Time:
        return Time([h.date_obs for h in self.headers])

    def cadence_s(self) -> float:
        """帧间隔中位数（秒）。单帧序列无间隔可言，显式报错而非返回 nan。"""
        if len(self.headers) < 2:
            raise ValueError(
                f"序列少于 2 帧，无法计算帧间隔: n={len(self.headers)}"
            )
        return float(np.median(np.diff(self.times().unix)))
