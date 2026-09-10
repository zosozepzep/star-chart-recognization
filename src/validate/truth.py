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
    # encoding="utf-8" is not cosmetic: np.loadtxt otherwise decodes with the
    # platform default, so a .DAT carrying a Chinese header comment parses in the
    # container (UTF-8 locale) and raises UnicodeDecodeError on a GBK host
    # (measured: "'gbk' codec can't decode byte 0xb4"). Ruling 352.
    raw = np.loadtxt(path, ndmin=2, encoding="utf-8")
    # An empty (or data-free) file lands on shape (0, 1), so the column guard
    # below would report "实际 1 列" for a file that has no columns at all.
    # Diagnose the real problem first. Ruling 352.
    if raw.size == 0:
        raise ValueError(f"{path.name} 为空或不含数据行")
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
    track, truth: TruthTable, sequence, registration=None, *, max_dt_s: float = 0.5
) -> TruthReport:
    """把目标轨迹与真值定量比对，导出比例尺、星等相关性与零点。

    `registration` 只为与流水线上下文的调用形状对齐而接受（Task 30 的调用点
    `truth_mod.compare(track, truth, seq, ctx.registration)` 按位置传第四个参数），
    **函数体从不读取它**——`track.xy_sky` 已经是配准坐标系里的坐标。默认 `None`
    使直接调用者不必构造一个被忽略的值。见 Ruling 350。

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

    把 cos 因子挪到赤纬差上（`hypot(Δra, Δdec·cos(dec))`）在数据集 B 上**测不出
    来**：dec ∈ [12.526, 21.527] → cos(dec) ∈ [0.9302, 0.9762]，离 1 太近，
    `angular_step` 只从 776.9149 挪到 770.6778，仍在 `rel=0.02` 界内，`plate_scale`
    只用掉 99.11% 的 `abs=0.05` 预算——余量 0.000443。区分这个物理错误的是纯合成
    单测 `test_angular_step_places_cos_factor_on_ra`（dec ≡ 60° 时两式相差 2 倍）。
    见 Ruling 347。

    **三个统计量都只在像素步长非退化的步上取均值。** `angular_step_arcsec`、
    `pixel_step_px`、`plate_scale_arcsec_px` 共用同一个 `good = pixel > 1e-6`
    掩码，于是三者由构造互相自洽，只剩「比值之均值 vs 均值之比」那 3.1e-6 的差。
    若只给 `plate_scale` 加掩码，一个前半段静止的目标（帧号可以互不重复，因此
    R344 的重复帧守卫盖不住）会让报告吐出 `plate_scale ≠ angular_step /
    pixel_step` 的三个数字，却带着 `available=True` 和正常 note 发出去——判决方
    一除就发现自相矛盾。见 Ruling 348。数据集 B 上这个掩码是恒等映射（实测 53/53
    步、最小步长 123.7161 px），所以它对下面那些定稿数字逐位免费。

    含蓄的比例尺：`776.9149 / 125.7334 = 6.17907`（两个均值之比），而
    `plate_scale_arcsec_px` 是**逐步比值的均值**，实测 **6.179047**。两者都约
    6.179，但不是同一个统计量，别把其中一个当另一个复算。

    `zero_point_std` 用 `diff.std()`，即 `ddof=0` 的总体标准差。实测在 54 个匹配
    点上 `ddof=0` → 0.217907、`ddof=1` → 0.219953，比 1.009390（0.94%）。想复算
    这个数的读者必须用 `ddof=0`。

    **零点的两条口径（Ruling 349，引用时必须写清，否则就是错的）：**

    1. `zero_point_std` 是**单点残差的样本散布**，**不是均值不确定度**。均值标准误
       是 `σ/√n = 0.219953/√54 = 0.029932`（用 `ddof=1` 的 σ 算）。所以报告只能写
       「零点 15.335，单点残差散布 0.218（均值标准误 0.030）」，不得写成
       `15.335 ± 0.218` 而不说 `±` 是什么。
    2. 零点的**光度基准是总 ADU，未作曝光归一化**：实测匹配帧的 exposure 恒为
       0.03 s（该目录另有 0.08 s 的帧，但都在跟踪段之外），所以 15.335125 是总
       ADU 基准；除以曝光后为 **19.142322**（`2.5·log10(0.03) = -3.807197`）。两者
       相差 3.807，相对 0.218 是 17.5σ——拿它与任何星表零点比较前必须先说清基准。
    """
    frames, rows = match_frames(truth, sequence, track.frames, max_dt_s=max_dt_s)

    # 重复帧守卫：`match_frames` 逐行独立取最近帧，两行真值可以落到同一帧上。
    # 数据集 B 上不会（实测 54 / 54 互不重复），但时钟有偏置的真值文件、数据集 A
    # 或任何后续数据集都会。
    #
    # 注意危害的措辞已随 R348 的掩码修法改变。R344 记的是"三个数字互不自洽"，
    # 那在掩码只保护 `plate_scale` 的旧实现上成立（在 R344 的 fixture 上偏
    # 33%，同一个 fixture 换个分母就是 50%——R344 的 33% 与 R348 的 50% 是同一次
    # 测量的两种归一化，不是两个 fixture 的性质）。三个均值共用 `good` 之后，
    # 重复帧带来的零像素步长会被三者一起剔除，实测 `plate_scale` 与
    # `angular_step / pixel_step` 重新逐位相等（0.0%），**旧的自洽性判据再也抓不
    # 到重复帧**。
    #
    # 仍然必须抛错的理由是另一个：重复帧意味着真值时钟与帧时钟有偏置，于是每一
    # 步的角位移与像素位移量的**不是同一段时间间隔**，比例尺被系统性地按
    # Δt_真值 / Δt_帧 偏置，而且是静默的。这个偏置没有任何可辩护的报法，所以只
    # 能拒收而不是告警。
    if len(set(frames.tolist())) != len(frames):
        unique, counts = np.unique(frames, return_counts=True)
        repeated = unique[counts > 1]
        raise ValueError(
            f"真值有 {len(frames) - len(unique)} 行与其他行匹配到重复帧号"
            f"（首个重复帧 f{int(repeated[0])}），说明真值时钟与帧时钟存在偏置，"
            f"角位移与像素位移量的不是同一段时间间隔，比例尺会被静默偏置，"
            f"拒绝比对；请检查两个时钟"
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
    if not good.any():
        # 目标在配准天球坐标里逐帧不动：每一步的比例尺都是 0/0。旧实现让
        # `np.mean([])` 产出 nan 却仍报 available=True，写出的 JSON 里是裸 `NaN`
        # token（`jq` / `JSON.parse` / Go 全部拒收）。见 Ruling 346 / 348。
        return TruthReport(
            dataset_id=sequence.dataset_id,
            available=False,
            n_matched=int(len(frames)),
            note="目标在配准坐标系里逐帧不动（像素步长全为零），无法定标比例尺",
        )
    angular_good = angular[good]
    pixel_good = pixel[good]
    scale = angular_good / pixel_good

    with np.errstate(divide="ignore", invalid="ignore"):
        m_inst = -2.5 * np.log10(np.where(flux > 0, flux, np.nan))
    ok = np.isfinite(m_inst) & np.isfinite(truth.mag[rows])
    if ok.sum() >= 3:
        mag_ok = truth.mag[rows][ok]
        inst_ok = m_inst[ok]
        # 零方差守卫：真值星等恒定（数据集 A 的合成真值、或任何单星等目标）或
        # flux 恒定时 `np.corrcoef` 除以零标准差，实测返回 nan。相关系数在这种
        # 输入上根本没有定义，报 None 而不是把 nan 写进 JSON。零点与散布不需要
        # 方差非零，照常计算。见 Ruling 346。
        if mag_ok.std() == 0.0 or inst_ok.std() == 0.0:
            corr = None
        else:
            corr = float(np.corrcoef(mag_ok, inst_ok)[0, 1])
        diff = mag_ok - inst_ok
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
        angular_step_arcsec=float(np.mean(angular_good)),
        pixel_step_px=float(np.mean(pixel_good)),
        mag_correlation=corr,
        zero_point=zp,
        zero_point_std=zp_std,
        note="真值仅用于验证与定标，未参与目标判决",
    )


def write_report(report: TruthReport, output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "truth_report.json"
    # allow_nan=False 是刻意的：默认 True 会把 nan 渲染成裸 `NaN` token，那不是
    # JSON（`jq` / `JSON.parse` / Go `encoding/json` 全部拒收，只有 Python 自己的
    # json.loads 接受它）。真值报告是判决方唯一的定量交付物，宁可在这里响亮抛
    # ValueError，也不要静默写出一个读不了的文件。见 Ruling 346。
    payload = json.dumps(report.to_dict(), indent=2, ensure_ascii=False, allow_nan=False)
    path.write_text(payload, encoding="utf-8")
    return path
