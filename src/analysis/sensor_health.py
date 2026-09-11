"""传感器健康报告与跨历元坏点持久性——派生数据集 4。

意义：同一批热像素簇在相隔 134.158 天的两次观测中位置几乎不动，这既
  (a) 证明它们是探测器固有缺陷而非天体，
  (b) 给出本项目热像素识别的一个不依赖真值的自查手段，
  (c) 构成一份对观测方有实用价值的设备状态副产品。

实测支撑（probe/t22-recheck.py..t22-recheck4.py）：
  剔除读出伪影后两历元各 4 个簇，A->B 最近邻距离
  [0.0000, 0.0000, 0.1310, 0.3353] px，**4 对中 2 对逐位相同**，
  最大分离 0.3353 px，均值 0.1166 px。
  注意口径：含伪影的 5 簇会给 [0,0,0,0.1310,0.3353]、均值 0.0933、
  「3 对逐位相同」——**那一组不得作为持久性证据**，因为多出来的那个
  d=0.0000 正是本模块剔除的 (0, 0)。

一个反面结论同样重要：位于 (0, 0) 的候选**不是坏点**。
  判据是量级，不是曝光响应：它在两个历元**恰好都是全帧最大值**
  （26975 / 26974 ADU），是同帧中部行中位数的 5395 / 2452 倍，
  而最亮的真缺陷 (2192,3223) 只有 520 / 236 倍；这个值在两历元 40 帧上
  只在 26966-26991 之间动（相对起伏 7e-4）。
  **不要用「不随曝光变化」当依据**：历元 A 实测 (0,0) 的 r(exp) = -0.3723，
  而真缺陷 (2192,3223) 是 -0.3932、(327,3210) 只有 -0.0667——同一把尺子
  会把真缺陷判成伪影。且 20 帧预算内曝光与帧序共线
  （corr = -0.6937 / -0.8496），r(exp) 本身读不出纯曝光效应。
  第 0 行 4096 列里只有 3 列异常，行中位数与第 1 行相等，所以剔的是
  「一行里的三个写入位」，代价是第 1、2 行实测 0 个异常像素——为零。
  剔除数量由 `n_metadata_artifacts` 如实报出，不隐藏任何东西。

坏点的空间分布：三簇集中在 (244-327, 3174-3210) 的 ~85 px 范围内，
  是一处局部损伤区；另有 (2192, 3223) 一个孤立单像素高值缺陷
  （median_value ~ 2608）。四簇均**不随曝光正向增长**：
  若 (312,3206) 是暗电流热点，30 ms 的 134 ADU 到 80 ms 应达 ~357 ADU，
  实测最大值仅 139 ADU、斜率 -0.1217 ADU/ms（反向）——暗电流被排除，
  这些是固定偏置/增益缺陷。

坏点簇的 npix 在两历元并不相同：实测 4 对里有 2 对是 9 vs 7（-22%）。
  位置复现与簇范围复现是两件事，所以 `common` 同时带 `npix_a`/`npix_b`。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from src.astrometry.photometry import instrumental_mag
from src.config import load_config
from src.detect.psf import FWHM_PER_SIGMA

_CFG_CACHE: dict | None = None


def _sensor_cfg() -> dict:
    """``sensor_health`` 段，取自 ``src/config/default.yaml``，缓存一次。"""
    global _CFG_CACHE
    if _CFG_CACHE is None:
        _CFG_CACHE = dict(load_config()["sensor_health"])
    return _CFG_CACHE


def _finite_or_none(value) -> float | None:
    """非有限值转 None——json.dumps 会写出裸 NaN，而 RFC 8259 与 JS JSON.parse 都拒绝它。"""
    if value is None:
        return None
    v = float(value)
    return v if np.isfinite(v) else None


@dataclass
class SensorReport:
    dataset_id: str
    epoch_isot: str
    n_clusters: int
    clusters: list[dict] = field(default_factory=list)
    background_median: float | None = None
    background_rms: float | None = None
    n_saturated: int = 0
    fwhm_median_px: float | None = None
    n_metadata_artifacts: int = 0
    metadata_rows: int = 0
    background_frame_index: int | None = None
    background_exposure_ms: float | None = None

    def to_dict(self) -> dict:
        return {
            "dataset_id": self.dataset_id,
            "epoch_utc": self.epoch_isot,
            "n_hot_clusters": int(self.n_clusters),
            "clusters": self.clusters,
            "background_median_adu": _finite_or_none(self.background_median),
            "background_rms_adu": _finite_or_none(self.background_rms),
            "n_saturated_px": int(self.n_saturated),
            "fwhm_median_px": _finite_or_none(self.fwhm_median_px),
            # 读出伪影的剔除是可审计的：剔了几个、按什么规则剔的，都写出来
            "n_metadata_artifacts": int(self.n_metadata_artifacts),
            "metadata_rows_excluded": int(self.metadata_rows),
            # 背景中位数强烈依赖取哪一帧（实测 B 帧0=11.0 ADU vs 帧19=6.0 ADU，差 83.33%），
            # 所以必须连帧号与曝光一起报，否则这个数字无法解释
            "background_frame_index": self.background_frame_index,
            "background_exposure_ms": _finite_or_none(self.background_exposure_ms),
        }

    @property
    def xy(self) -> np.ndarray:
        if not self.clusters:
            return np.zeros((0, 2), dtype=np.float64)
        return np.array([[c["cx"], c["cy"]] for c in self.clusters], dtype=np.float64)


def build_report(
    dataset_id: str,
    hotpixel_map,
    bkg_stats: dict,
    *,
    epoch_isot: str,
    fwhm_median_px: float | None = None,
    metadata_rows: int | None = None,
    background_frame_index: int | None = None,
    background_exposure_ms: float | None = None,
) -> SensorReport:
    """把一次观测的坏点图与背景统计整理成一份设备状态报告。

    `metadata_rows`：图像顶部若干行由相机写入状态字，不是感光区缺陷，
    从坏点普查中剔除。缺省从 `sensor_health.metadata_rows` 读（实测依据见
    模块 docstring）；显式传 0 可保留全部簇。
    """
    if metadata_rows is None:
        metadata_rows = int(_sensor_cfg()["metadata_rows"])
    if metadata_rows < 0:
        raise ValueError(f"metadata_rows 不能为负: {metadata_rows}")

    kept, n_artifacts = [], 0
    for c in hotpixel_map.clusters:
        # 严格小于：cy 是行**下标**而 metadata_rows 是行**数**，所以
        # metadata_rows=1 只剔第 0 行；写成 <= 会连第 1 行一起吃掉。
        if float(c.cy) < float(metadata_rows):
            n_artifacts += 1
            continue
        kept.append(
            {
                "cx": float(c.cx),
                "cy": float(c.cy),
                "npix": int(c.npix),
                "median_value": float(c.median_value),
                "flat_ratio": float(c.flat_ratio),
            }
        )
    return SensorReport(
        dataset_id=dataset_id,
        epoch_isot=str(epoch_isot),
        n_clusters=len(kept),
        clusters=kept,
        background_median=float(bkg_stats["median"]),
        background_rms=float(bkg_stats["std"]),
        n_saturated=int(bkg_stats["n_saturated"]),
        fwhm_median_px=None if fwhm_median_px is None else float(fwhm_median_px),
        n_metadata_artifacts=n_artifacts,
        metadata_rows=int(metadata_rows),
        background_frame_index=(
            None if background_frame_index is None else int(background_frame_index)
        ),
        background_exposure_ms=(
            None if background_exposure_ms is None else float(background_exposure_ms)
        ),
    )


@dataclass
class CrossEpochResult:
    epochs: list[str]
    match_radius_px: float
    common: list[dict] = field(default_factory=list)
    only_in_a: list[dict] = field(default_factory=list)
    only_in_b: list[dict] = field(default_factory=list)

    @property
    def n_common(self) -> int:
        return len(self.common)

    @property
    def n_a(self) -> int:
        return len(self.common) + len(self.only_in_a)

    @property
    def n_b(self) -> int:
        return len(self.common) + len(self.only_in_b)

    @property
    def persistence(self) -> float:
        """common / min(n_a, n_b)。

        注意口径：分母取两历元中较少的那个，所以 3 对 100 且 3 个全配上时
        它读 1.0000——"100% 的坏点持久"，而 97 个 B 簇没有解释。
        因此 `to_dict()` 同时输出 `n_a`、`n_b` 与 `jaccard`，
        任何引用 persistence 的地方都能被反查。
        """
        denom = min(self.n_a, self.n_b)
        return float(len(self.common) / denom) if denom else 0.0

    @property
    def jaccard(self) -> float:
        """common / (n_a + n_b - common)——并集口径，不会掩盖任一侧的盈余。"""
        denom = self.n_a + self.n_b - len(self.common)
        return float(len(self.common) / denom) if denom else 0.0

    @property
    def max_separation_px(self) -> float | None:
        if not self.common:
            return None
        return float(max(c["separation_px"] for c in self.common))

    def to_dict(self) -> dict:
        return {
            "epochs": list(self.epochs),
            "match_radius_px": float(self.match_radius_px),
            "n_a": self.n_a,
            "n_b": self.n_b,
            "n_common": self.n_common,
            "persistence": self.persistence,
            "jaccard": self.jaccard,
            "max_separation_px": self.max_separation_px,
            "common": self.common,
            "only_in_a": self.only_in_a,
            "only_in_b": self.only_in_b,
        }


def _match_nearest_first(
    xy_a: np.ndarray, xy_b: np.ndarray, radius: float
) -> list[tuple[int, int, float]]:
    """按距离升序做一对一匹配，返回 (i, j, d) 列表。

    不能用"先到先得"的贪心：那样会把一个 B 簇配给先遍历到的 A 簇，
    即使另一个 A 簇更近。实测反例 A=[(100,100), (101,100)]、B=[(101.2,100)]
    的距离是 [1.2, 0.2]，贪心配出 1.2，把 0.2 这一对更好的匹配挤进 only_in_a。
    先收集半径内的**全部**候选对，按距离排序后再分配，结果与输入顺序无关。
    """
    if len(xy_a) == 0 or len(xy_b) == 0:
        return []
    tree = cKDTree(xy_b)
    candidates: list[tuple[float, int, int]] = []
    for i, neighbours in enumerate(tree.query_ball_point(xy_a, radius)):
        for j in neighbours:
            j = int(j)
            d = float(np.hypot(xy_a[i][0] - xy_b[j][0], xy_a[i][1] - xy_b[j][1]))
            if d < radius:  # 严格小于，与 match_radius_px 的语义一致
                candidates.append((d, i, j))
    candidates.sort()
    used_a: set[int] = set()
    used_b: set[int] = set()
    pairs: list[tuple[int, int, float]] = []
    for d, i, j in candidates:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        pairs.append((i, j, d))
    return pairs


def compare_epochs(
    report_a: SensorReport,
    report_b: SensorReport,
    *,
    match_radius_px: float | None = None,
) -> CrossEpochResult:
    """对比两个历元的坏点普查，给出复现的簇、各自独有的簇与持久性指标。

    `match_radius_px` 缺省从 `sensor_health.match_radius_px` 读；实测在
    0.5-16.0 px（32 倍跨度）上 n_common 恒定，所以缺省值不在悬崖上。
    """
    if match_radius_px is None:
        match_radius_px = float(_sensor_cfg()["match_radius_px"])
    if not (match_radius_px > 0.0):
        raise ValueError(f"match_radius_px 必须为正: {match_radius_px}")

    xy_a, xy_b = report_a.xy, report_b.xy
    res = CrossEpochResult(
        epochs=[report_a.epoch_isot, report_b.epoch_isot],
        match_radius_px=float(match_radius_px),
    )
    if len(xy_a) == 0 or len(xy_b) == 0:
        res.only_in_a = list(report_a.clusters)
        res.only_in_b = list(report_b.clusters)
        return res

    pairs = _match_nearest_first(xy_a, xy_b, float(match_radius_px))
    matched_a = {i for i, _, _ in pairs}
    matched_b = {j for _, j, _ in pairs}
    for i, j, d in pairs:
        res.common.append(
            {
                "xy_a": xy_a[i].tolist(),
                "xy_b": xy_b[j].tolist(),
                "separation_px": float(d),
                "npix_a": int(report_a.clusters[i]["npix"]),
                "npix_b": int(report_b.clusters[j]["npix"]),
            }
        )
    res.only_in_a = [c for i, c in enumerate(report_a.clusters) if i not in matched_a]
    res.only_in_b = [c for j, c in enumerate(report_b.clusters) if j not in matched_b]
    return res


@dataclass(frozen=True)
class SeeingEstimate:
    """一帧的合成星像宽度。

    ``fwhm_px`` 与 ``elongation_median`` 是**纯像素量，不含真值**；
    ``fwhm_arcsec`` 乘了真值定出的板比例（6.179 ″/px，
    ``TruthReport.plate_scale_arcsec_px``），**是真值定标量**。
    报告引用后者时不得称为独立观测结论（R426）。
    """

    n_stars: int
    n_rejected: int = 0
    fwhm_px: float | None = None
    fwhm_arcsec: float | None = None
    elongation_median: float | None = None
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "n_stars": int(self.n_stars),
            "n_rejected": int(self.n_rejected),
            "fwhm_px": _finite_or_none(self.fwhm_px),
            "fwhm_arcsec": _finite_or_none(self.fwhm_arcsec),
            "elongation_median": _finite_or_none(self.elongation_median),
            "note": self.note,
        }


def estimate_seeing(fits, scale_arcsec_px: float) -> SeeingEstimate:
    """由收敛的高斯拟合给出合成星像宽度。

    只取 ``success=True`` 且两轴 sigma 都落在配置区间内的拟合。四种非物理
    sigma 都能穿过单纯的 ``isfinite`` 检查：``sigma=0`` 给 FWHM 0.0；两轴同为
    负时乘积为正、负号被 ``sqrt`` 吞掉，得到与正常值无法区分的 3.767712；
    一正一负则 ``sqrt`` 得 ``nan``，``np.median`` 把整个结果传染成 ``nan``
    而 ``n_stars`` 照旧计数；``sigma=1e6`` 给 2354820.0450。所以要区间而非
    ``isfinite``。真实数据上这个区间实测剔 0 个（sigma 跨度 0.883-2.063），
    它防的是穿透，不是常态筛选。

    sigma 用两轴几何平均，对轻微拖长比算术平均稳（实测 2:1 拖长下几何 σ
    2.2627 vs 算术 2.4000）。用中位数而非均值：实测
    ``sigma=[1.2,1.4,1.6,1.8,9.9]`` 时中位数给 FWHM 3.767712、均值给 7.488328，
    差 3.720616 px，一个失控的拟合不该拖动结论。

    **返回的是「探测器 + 大气 + 跟踪拖长」的合成宽度，不是大气视宁度**，
    因此 ``elongation_median`` 必须与数字一起返回——拖长比是这个数字可读性的
    **前提条件**。合成 10:1 拖长时几何平均给 FWHM 11.914552 px
    （73.6200 arcsec），那显然不是视宁度；而本项目两个历元实测拖长比中位数
    只有 1.1959 / 1.2027（最长的单个源 2.65:1），所以前提成立，
    3.40 px 这个数可以当星像宽度读。
    """
    cfg = _sensor_cfg()
    lo = float(cfg["sigma_min_px"])
    hi = float(cfg["sigma_max_px"])

    sig: list[float] = []
    elong: list[float] = []
    n_rejected = 0
    for f in fits:
        sx, sy = float(f.sigma_x), float(f.sigma_y)
        if not (f.success and np.isfinite(sx) and np.isfinite(sy)):
            n_rejected += 1
            continue
        if not (lo <= sx <= hi and lo <= sy <= hi):
            n_rejected += 1
            continue
        sig.append(float(np.sqrt(sx * sy)))
        elong.append(float(max(sx, sy) / min(sx, sy)))

    if not sig:
        return SeeingEstimate(
            n_stars=0, n_rejected=n_rejected, note="无可用的 PSF 拟合，星像宽度不可估"
        )
    fwhm = float(np.median(sig)) * FWHM_PER_SIGMA
    return SeeingEstimate(
        n_stars=len(sig),
        n_rejected=n_rejected,
        fwhm_px=fwhm,
        fwhm_arcsec=fwhm * float(scale_arcsec_px),
        elongation_median=float(np.median(elong)),
        note="含跟踪拖长的合成星像宽度，非纯大气视宁度；可读性前提见 elongation_median",
    )


def sky_brightness(
    bkg_median_adu,
    *,
    zero_point,
    scale_arcsec_px,
    exposure_s: float | None = None,
) -> float | None:
    """天光背景面亮度（mag/arcsec²）。

        mu = ZP - 2.5 log10( B / t ) + 2.5 log10( s^2 )   当 zp.exposure_s 给出
        mu = ZP - 2.5 log10( B )     + 2.5 log10( s^2 )   当 zp.exposure_s 是 None

    **曝光时间取自 ``zero_point.exposure_s``，不由调用方决定**（与
    ``calibrate_table`` 同一先例）。``exposure_s`` 只是允许调用方显式复述的
    覆盖参数，``None``（默认）表示「不复述、沿用零点」；给出而与
    ``zp.exposure_s`` 不一致时抛 ``ValueError``。理由：那个不一致**就是**
    ``-2.5 log10(t)`` 的整体平移（t=0.030 s 时实测 3.807197 mag），
    静默采用其中一边会让 mu 从 19.178117 变成 15.370920 而没有任何痕迹，
    而它是**可检测的**，所以不该挑一边。

    面积项是**加**的：per-arcsec² 的流量比 per-px 小 ``s²`` 倍
    （6.179 ″/px 时 38.1800 倍），所以星等更暗，实测 +3.954591 mag。
    丢掉这一项在同一算例上给 15.223526。

    **这个数的全部信息量在零点里**：平移零点 1 mag，mu 平移 1 mag。
    本项目的零点由真值定出（15.335125，散度 0.217907），板比例也由真值定出，
    所以这是条件命题「给定该零点则 mu = 17.34」，**不是独立观测量**（R426）。
    实测数据集 B 第 30 帧 6.0 ADU -> 17.344338，A 第 30 帧 5.0 ADU -> 17.542291，
    落在城市微光夜空边缘。背景中位是整数 ADU，1 ADU 在 6.0 上折 0.18 mag，
    所以报告不要写小数位。

    零点不可用、或背景/板比例非正时返回 ``None``——绝不用未定标的仪器值
    冒充面亮度。
    """
    if zero_point is None:
        return None

    zp_exposure = zero_point.exposure_s
    if exposure_s is not None:
        if zp_exposure is None or not np.isclose(
            float(exposure_s), float(zp_exposure), rtol=0.0, atol=1e-12
        ):
            raise ValueError(
                f"曝光时间与零点记录不一致: 零点 {zp_exposure!r}，调用方 {exposure_s!r}"
            )

    b = float(bkg_median_adu)
    s = float(scale_arcsec_px)
    if not (b > 0.0 and s > 0.0):
        return None
    rate = b if zp_exposure is None else b / float(zp_exposure)
    if not rate > 0.0:
        return None
    return float(
        zero_point.value
        + float(instrumental_mag(np.array([rate]))[0])
        + 2.5 * np.log10(s * s)
    )
