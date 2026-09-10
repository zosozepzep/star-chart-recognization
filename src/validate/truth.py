"""`.DAT` 真值读取与定量比对。

严格约束（真值隔离）：本模块只被验证与报告环节导入。detect / register / target
三个包不得引用它——否则整条判决链就变成"用真值算出真值"，这在答辩中是致命的
原理性质疑。该约束由两处守卫：本任务 `tests/validate/test_truth.py` 的
`test_judgement_chain_does_not_import_truth`（AST 走这三个包，只查这一个方向），
以及 Task 13 `tests/test_architecture.py` 的完整导入图检查。

.DAT 格式（见数据源 `说明文件.txt`，逐字引用）::

    时间系统：UTC 世界时
    坐标系统：J2000
    内容格式：年 月 日 时 分 秒 赤经 赤纬 星等

赤经 = right ascension，赤纬 = declination，J2000——所以字段名是 `ra_deg` /
`dec_deg`。实测取值范围与之相符：第 6 列 ∈ [201.724, 209.236]（合法 RA），
第 7 列 ∈ [12.355, 21.527]（合法 Dec）。星等字段为 000 表示缺测。

时间戳格式化用 `%09.6f` 而非 `%.6f`，因为文件跨了整分边界：分列从 26 走到 27
（0 基行 15 = `17 26 59.4010`，行 16 = `17 27 00.4100`），秒列取值 0.4100 –
59.4010。`%09.6f` 把 0.410 渲染成 `"00.410000"`，整数位补零，与判决方自己的
零填充秒字段（`00.4100`）同形——Task 25 的 `.DAT` 写出必须逐字节复现这个形状。
注意这是**字节保真**要求而不是可解析性要求：`Time("2026-07-21T17:26:0.410000",
scale="utc")` 实测被 astropy 正常接受并解析正确，所以裸 `%.6f` 不会炸，只会写
出与判决方不同形的文本。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.time import Time

N_COLUMNS = 9


@dataclass
class TruthTable:
    path: Path
    time: Time
    ra_deg: np.ndarray
    dec_deg: np.ndarray
    mag: np.ndarray

    def __len__(self) -> int:
        return int(self.ra_deg.size)


def load_truth(path: str | Path) -> TruthTable:
    path = Path(path)
    raw = np.loadtxt(path, ndmin=2)
    if raw.shape[1] != N_COLUMNS:
        raise ValueError(
            f"{path.name} 应有 {N_COLUMNS} 列（年 月 日 时 分 秒 赤经 赤纬 星等），"
            f"实际 {raw.shape[1]} 列"
        )

    # %09.6f: 秒的整数位补零，见模块 docstring 的跨分边界说明。
    stamps = [
        "%04d-%02d-%02dT%02d:%02d:%09.6f" % (r[0], r[1], r[2], r[3], r[4], r[5]) for r in raw
    ]
    mag = raw[:, 8].astype(np.float64)
    mag[mag <= 0.0] = np.nan  # 000 表示缺测

    return TruthTable(
        path=path,
        time=Time(stamps, scale="utc"),
        ra_deg=raw[:, 6].astype(np.float64),
        dec_deg=raw[:, 7].astype(np.float64),
        mag=mag,
    )


def match_frames(truth: TruthTable, sequence, frames, *, max_dt_s: float = 0.5):
    """按时间戳把真值行匹配到帧号。返回 (帧号数组, 真值行号数组)。

    数据集 B 上这两个时钟**作为时刻逐行相等**：把 55 行真值对全部 80 帧匹配，
    得到帧 16–70 连续、行 0–54 顺序、55 个互不重复的帧号，且 `dt` 的 max / mean
    / min 实测**恰为 0**——不是"很小"。float64 的 JD 差为 0，astropy 高精度的
    `(truth_time - frame_time).sec` 也为 0。

    但它们**不是文本相等**：实测 0 / 55 条时间戳字符串逐字节相同，因为 `.DAT`
    的秒字段带四位小数（`44.2670`），这里的 `%09.6f` 渲染六位（`44.267000`），
    两者只因多出的位全是尾随零才等值。Task 25 要写出逐字节一致的 `.DAT`，只有
    解析后的数值能过这道往返，文本不能直接抄。

    直接后果：`max_dt_s` 在数据集 B 上完全不起作用（任何取值都留下全部 55 行），
    所以它的行为只能由合成序列来测，见
    `tests/validate/test_truth.py::test_match_frames_respects_tolerance_synthetic`。

    每一行独立取最近帧，因此两行真值可能落到同一帧上（时钟有偏置的真值文件就会
    这样）。本函数不去重——`compare` 负责拒收，见那里的守卫。
    """
    frames = np.asarray(list(frames), dtype=int)
    frame_unix = sequence.times().unix[frames]
    truth_unix = truth.time.unix

    matched_frames: list[int] = []
    matched_rows: list[int] = []
    for row, t in enumerate(truth_unix):
        k = int(np.argmin(np.abs(frame_unix - t)))
        if abs(frame_unix[k] - t) <= max_dt_s:
            matched_frames.append(int(frames[k]))
            matched_rows.append(row)
    return np.array(matched_frames, dtype=int), np.array(matched_rows, dtype=int)


@dataclass
class TruthReport:
    dataset_id: str
    available: bool
    n_matched: int = 0
    dt_max_s: float | None = None
    plate_scale_arcsec_px: float | None = None
    plate_scale_std: float | None = None
    angular_step_arcsec: float | None = None
    pixel_step_px: float | None = None
    mag_correlation: float | None = None
    zero_point: float | None = None
    zero_point_std: float | None = None
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "dataset_id": self.dataset_id,
            "available": self.available,
            "n_matched": self.n_matched,
            "dt_max_s": self.dt_max_s,
            "plate_scale_arcsec_px": self.plate_scale_arcsec_px,
            "plate_scale_std": self.plate_scale_std,
            "angular_step_arcsec": self.angular_step_arcsec,
            "pixel_step_px": self.pixel_step_px,
            "mag_correlation": self.mag_correlation,
            "zero_point": self.zero_point,
            "zero_point_std": self.zero_point_std,
            "note": self.note,
        }


def compare(
    track, truth: TruthTable, sequence, registration, *, max_dt_s: float = 0.5
) -> TruthReport:
    """把目标轨迹与真值定量比对，导出比例尺、星等相关性与零点。

    **为什么数据集 B 上 `n_matched` 是 54 而不是 55。** 真值覆盖帧 16–70（55 行），
    而目标轨迹覆盖帧 17–70（54 点）——目标在 f16 还探不到。本函数把
    `track.frames` 当候选帧列表传给 `match_frames`，于是真值第 0 行（f16）匹配不
    到任何轨迹点而被丢弃，剩下行 1–54 对帧 17–70。知道真值有 55 行的读者若不知道
    这一层，会把 `n_matched=54` 读成 bug；它不是。

    **角步长的定义。** 用余弦改正的平面式 `hypot(Δra·cos(dec), Δdec)`——cos 因子
    属于赤经差，因为赤经线向极点收敛。三种合理定义在本数据上差别可忽略，实测于
    `compare` 真正看到的 **53 步**上：余弦改正平面式 **776.9149″/步**、大圆
    **776.7793**、不作改正的平面式 **791.2781**。大圆与本式相差 0.017%，写成大圆
    量到 776.78 的实现不要以为任务书错了。报告里引用的口径是「53 个匹配步上的
    776.91″/步」，与"全部 55 行 = 54 步"的统计量（均值 776.6289）不是同一个数。

    含蓄的比例尺：`776.9149 / 125.7334 = 6.17907`（两个均值之比），而
    `plate_scale_arcsec_px` 是**逐步比值的均值**，实测 **6.179047**。两者都约
    6.179，但不是同一个统计量，别把其中一个当另一个复算。

    `zero_point_std` 用 `diff.std()`，即 `ddof=0` 的总体标准差。实测在 54 个匹配
    点上 `ddof=0` → 0.217907、`ddof=1` → 0.219953，比 1.009390（0.94%）。想复算
    这个数的读者必须用 `ddof=0`。
    """
    frames, rows = match_frames(truth, sequence, track.frames, max_dt_s=max_dt_s)

    # 重复帧守卫：`match_frames` 逐行独立取最近帧，两行真值可以落到同一帧上。
    # 数据集 B 上不会（实测 54 / 54 互不重复），但时钟有偏置的真值文件、数据集 A
    # 或任何后续数据集都会，而且是静默的：下面 `good = pixel > 1e-6` 只保护
    # `plate_scale_arcsec_px`，`angular_step_arcsec` 与 `pixel_step_px` 两个均值
    # 仍把那个零步长算进去。实测后果是三个数字不再满足
    # `plate_scale ≈ angular_step / pixel_step`（相差 33%），却带着
    # available=True 和空 note 发出去——判决方一除就发现自相矛盾。这三个数没有
    # 任何可辩护的报法，所以只能抛错而不是告警。
    if len(set(frames.tolist())) != len(frames):
        unique, counts = np.unique(frames, return_counts=True)
        repeated = unique[counts > 1]
        raise ValueError(
            f"真值有 {len(frames) - len(unique)} 行与其他行匹配到重复帧号"
            f"（首个重复帧 f{int(repeated[0])}），比例尺与角/像素步长将互不自洽，"
            f"拒绝比对；请检查真值时钟与帧时钟是否存在偏置"
        )

    if len(frames) < 3:
        return TruthReport(
            dataset_id=sequence.dataset_id,
            available=False,
            note=f"真值与轨迹时间匹配仅 {len(frames)} 点，不足以定量比对",
        )

    order = {f: i for i, f in enumerate(track.frames)}
    sel = np.array([order[f] for f in frames], dtype=int)
    xy_sky = track.xy_sky[sel]
    flux = track.flux[sel]

    ra = truth.ra_deg[rows]
    dec = truth.dec_deg[rows]
    dra = np.diff(ra) * np.cos(np.deg2rad(dec[:-1]))
    ddec = np.diff(dec)
    angular = np.hypot(dra, ddec) * 3600.0            # ″/步
    pixel = np.hypot(*np.diff(xy_sky, axis=0).T)       # px/步
    good = pixel > 1e-6
    scale = angular[good] / pixel[good]

    with np.errstate(divide="ignore", invalid="ignore"):
        m_inst = -2.5 * np.log10(np.where(flux > 0, flux, np.nan))
    ok = np.isfinite(m_inst) & np.isfinite(truth.mag[rows])
    if ok.sum() >= 3:
        corr = float(np.corrcoef(truth.mag[rows][ok], m_inst[ok])[0, 1])
        diff = truth.mag[rows][ok] - m_inst[ok]
        zp, zp_std = float(diff.mean()), float(diff.std())   # ddof=0
    else:
        # 星等全缺测（真值 000 列）时降级为纯几何比对：几何仍然有效，所以
        # available 保持 True，三个星等量报 None 而不是 nan。np.corrcoef 在全 nan
        # 输入上实测返回 nan，而 nan 进 json.dumps(allow_nan=False) 会抛错。
        corr = zp = zp_std = None

    dt = np.abs(sequence.times().unix[frames] - truth.time.unix[rows])
    return TruthReport(
        dataset_id=sequence.dataset_id,
        available=True,
        n_matched=int(len(frames)),
        dt_max_s=float(dt.max()),
        plate_scale_arcsec_px=float(np.mean(scale)),
        plate_scale_std=float(np.std(scale)),
        angular_step_arcsec=float(np.mean(angular)),
        pixel_step_px=float(np.mean(pixel)),
        mag_correlation=corr,
        zero_point=zp,
        zero_point_std=zp_std,
        note="真值仅用于验证与定标，未参与目标判决",
    )


def write_report(report: TruthReport, output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "truth_report.json"
    path.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return path
