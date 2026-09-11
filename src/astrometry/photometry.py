"""仪器星等、真值零点定标与极限星等。

零点来源优先级：
  1. truth —— 用 .DAT 中目标自身的星等序列求零点。离线、无需星表与网络，
     且与本次观测的曝光、增益、大气状态完全同源，是最可靠的一路。
  2. catalog —— 盲板求解成功后用星表比对（Task 19 可选路径，需网络）。
  3. none —— 两者都不可用时只报仪器星等，明确标注未定标，绝不假装有绝对星等。

零点定标属于验证/定标环节，允许读真值；目标判决链不得引用本模块的
``zero_point_from_truth``（判决只用像素运动学）。``zero_point_from_truth``
只收裸数组，真值文件的读取由调用方完成，因此 ``src/detect/``、
``src/register/``、``src/target/`` 经由本模块也拿不到通往真值的路径。

**``limiting_magnitude`` 报的是什么**：它返回**定标星等分布的第 N 个百分位**，
是一个经验暗端指标。它**不是**可复现性极限（没有任何跨帧比对），也**不是**探测
完备性极限（没有注入-回收实验）。报告里引用的就是这个百分位数，不要在任何地方
把它说成「最暗可复现源」。要做真正的可复现性极限，需要「哪些源复现了」这个逐源
掩码，而 Task 14 的 ``cross_frame_repeatability`` 只返回三个聚合计数
（``(n_reproducible, reproducibility, purity)``），掩码是其函数内的局部变量、
不外露；所以那条路要么改动一个已关闭模块的签名、要么在此处复制它的 KD-tree
匹配逻辑，两者都不做，本模块因此不提供该变体。

曝光时间绑定在 ``ZeroPoint`` 上而不是逐次调用传入。理由是实测的一条静默系统
误差：用 ``exposure_s=0.08`` 拟合零点却不带曝光去定标，每一个星等平移
``2.5*log10(0.08) = -2.7423`` mag，而调用方无从察觉。现在 ``calibrate_table``
只认 ``zp.exposure_s``，显式覆盖值不一致时抛错。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from src.config import load_config

logger = logging.getLogger(__name__)

# 两个常量是**回退默认值**，权威取值在 src/config/default.yaml 的 photometry 段。
# 「回退」的含义是精确的：只有配置里缺这个键时才用到它们（``_photometry_cfg`` 的
# ``.get``）。键存在时模块读的是配置——``test_min_zp_points_override_changes_behaviour``
# 与 ``test_max_clip_rounds_override_changes_behaviour`` 用改过的配置证实了这一点。
#
# 原先这两个数**只是**模块常量，靠一条「配置值 == 模块常量」的测试维持同步，而模块
# 从不读配置。那条测试的牙齿只到「两处数字相等」为止：把 default.yaml 的值改成 9
# 并同步改常量，它照样绿，而调用方以为自己调了配置、实际什么也没变。
MIN_ZP_POINTS = 3
MAX_CLIP_ROUNDS = 5

_CFG_CACHE: dict | None = None


def _photometry_cfg() -> dict:
    """``photometry`` 段，取自 ``src/config/default.yaml``，缓存一次。

    读取时机与失败行为照 ``src.analysis.sensor_health._sensor_cfg`` 的既有做法：
    **首次用到时懒加载并缓存**，不在 import 期读（import 期读会让任何 import 本模块
    的动作都依赖配置文件存在，测试收集阶段就会炸）。缺键时回退到模块常量而不抛错，
    因为一个残缺的配置不该让整条测光链无法运行——回退值与出厂值相同，
    行为因此与配置完整时一致。
    """
    global _CFG_CACHE
    if _CFG_CACHE is None:
        _CFG_CACHE = dict(load_config()["photometry"])
    return _CFG_CACHE


def _min_zp_points() -> int:
    return int(_photometry_cfg().get("min_zp_points", MIN_ZP_POINTS))


def _max_clip_rounds() -> int:
    return int(_photometry_cfg().get("max_clip_rounds", MAX_CLIP_ROUNDS))


def _validate_exposure(exposure_s: float | None) -> float | None:
    """校验曝光时间，非正值抛中文 ``ValueError``。

    必须在任何 ``log10`` 之前调用。写成 ``exposure_s is None`` 的显式判断而不是
    ``if exposure_s:`` 真值测试：后者把 ``0.0`` 和 ``0`` 当成「未请求归一化」而
    静默放过，而负值会走进 ``log10`` 产出一个与「流量 <= 0」无法区分的 nan。
    另外，曝光那一行的 ``log10`` 不在 ``np.errstate`` 块内，负值会漏出
    ``RuntimeWarning``，于是行为取决于 ``pytest.ini`` 的 ``filterwarnings``
    配置——把校验提到最前面，nan 与警告两条路都不再存在。
    """
    if exposure_s is None:
        return None
    value = float(exposure_s)
    if not value > 0.0:
        raise ValueError(f"曝光时间必须为正，实际 {value!r}")
    return value


def instrumental_mag(flux, exposure_s: float | None = None) -> np.ndarray:
    """仪器星等 ``-2.5*log10(flux)``，可选按曝光时间归一化为流量率。

    ``flux <= 0`` 给 ``nan``（无定义星等，且一帧里出现几个是常态），该 ``log10``
    被 ``np.errstate`` 包住所以不留警告。``exposure_s`` 给出时加
    ``2.5*log10(exposure_s)``，等价于 ``-2.5*log10(flux/exposure_s)``；
    ``exposure_s <= 0`` 抛 ``ValueError``，校验在两个 ``log10`` 之上。
    """
    exposure = _validate_exposure(exposure_s)
    flux = np.asarray(flux, dtype=np.float64).ravel()
    with np.errstate(divide="ignore", invalid="ignore"):
        m = -2.5 * np.log10(np.where(flux > 0.0, flux, np.nan))
    if exposure is not None:
        m = m + 2.5 * np.log10(exposure)
    return m


@dataclass(frozen=True)
class ZeroPoint:
    """一次零点定标的结果。

    ``std`` 是**剪裁后存活点**的散度，不是输入点的散度——判据问
    「0.218 是哪些点的散度」时答案就是这个。它对真实高斯散度**系统性偏低约
    2~3%**，成因有两个、量级相当（n=54 实测，见 ``zero_point_from_truth``）：
    ``np.std`` 默认 ``ddof=0`` 加上样本标准差的有限 n 偏差占约 **1.4%**
    （理论值 ``sqrt((n-1)/n) * c4(n) = 0.9860``），3σ 剪裁的**单侧性**占约
    **1.2%**——它只会剪掉尾部、只会让 ``std`` 变小，所以哪怕多数次拟合一个不剪
    （实测剪裁数中位 0、有剪裁的比例 0.147），均值仍被往下拽。

    ``exposure_s`` 记录拟合时用的曝光时间（``None`` 表示未归一化），定标必须沿用
    同一个值，否则会有 ``2.5*log10(exposure_s)`` 量级的整体平移。
    """

    value: float
    std: float
    n_points: int
    source: str = "truth"
    exposure_s: float | None = None

    def apply(self, m_inst) -> np.ndarray:
        """把零点加到仪器星等上，返回定标星等。"""
        return np.asarray(m_inst, dtype=np.float64) + self.value

    def to_dict(self) -> dict:
        """返回可直接 ``json.dumps`` 的字典。"""
        return {
            "value": float(self.value),
            "std": float(self.std),
            "n_points": int(self.n_points),
            "source": self.source,
            "exposure_s": None if self.exposure_s is None else float(self.exposure_s),
        }


def zero_point_from_truth(
    flux,
    truth_mag,
    *,
    exposure_s: float | None = None,
    sigma_clip: float = 3.0,
) -> ZeroPoint:
    """用真值星等序列拟合零点，返回 ``ZeroPoint``。

    零点取 ``truth_mag - instrumental_mag(flux)`` 的均值，先剔除非有限配对，再做
    最多 ``photometry.max_clip_rounds`` 轮 3σ 剪裁。剪裁基于 ``np.std`` 而非 MAD：实测在带真实
    散度的算例（40 点、σ=0.218、一个 +5 mag 错配）上两者结果**逐位一致**，不必换。

    有效配对少于 ``photometry.min_zp_points`` 时抛 ``ValueError``。注意剪裁循环的进入条件是
    ``len(d) > min_zp_points``（严格大于），所以恰好等于下限的输入只走未剪裁路径。

    **两个阈值都从配置读**（``_photometry_cfg``，懒加载并缓存），模块常量只是缺键
    时的回退值。这一点是可验证的而不是约定：把 ``min_zp_points`` 调到 5 后
    4 个有效配对实测抛错，而模块常量仍是 3。

    **轮数上限咬住时会记一条 WARNING**（终审第 2 项）。上限原先是硬编码的 5、
    且耗尽后直接落到 ``return``，没有任何痕迹。本项目算例用不到它：
    干净 12 点 1 轮（``sd == 0`` 即停）、带真实散度 12 点 1 轮、
    40 点含一个 +5 mag 错配 2 轮。但 5 轮**是可以被咬住的**，
    构造输入实测：40 个 0 加 10 个等差离群点要 6 轮，
    30 点几何尾 ``3**k`` 要 24 轮，后者 5 轮时剩 25 点、均值 1.694577e+10，
    放开上限后剩 7 点、均值 156.142857——**差 1.69e10 mag 而无警告**。
    这不是本项目会遇到的输入，所以上限不必提高；要紧的是它不再静默。
    """
    min_points = _min_zp_points()
    max_rounds = _max_clip_rounds()
    m_inst = instrumental_mag(flux, exposure_s=exposure_s)
    truth_mag = np.asarray(truth_mag, dtype=np.float64).ravel()
    if len(truth_mag) != len(m_inst):
        raise ValueError("流量与真值星等数量不一致")

    diff = truth_mag - m_inst
    ok = np.isfinite(diff)
    if ok.sum() < min_points:
        raise ValueError(f"零点定标至少需要 {min_points} 个有效配对，实际 {int(ok.sum())}")

    d = diff[ok]
    if sigma_clip and len(d) > min_points:
        converged = False
        for _ in range(max_rounds):
            med, sd = np.median(d), np.std(d)
            if sd <= 0.0:
                converged = True
                break
            keep = np.abs(d - med) <= sigma_clip * sd
            if keep.all():
                converged = True
                break
            d = d[keep]
            if len(d) <= min_points:
                converged = True
                break
        if not converged:
            # 静默耗尽上限就是「剪裁没做完但结果照发」，量级不可预估，必须留痕。
            logger.warning(
                "sigma clipping hit the %d-round cap without converging; "
                "%d of %d pairs kept, zero point may still be biased",
                max_rounds,
                len(d),
                int(ok.sum()),
            )

    return ZeroPoint(
        value=float(np.mean(d)),
        std=float(np.std(d)),
        n_points=int(len(d)),
        source="truth",
        exposure_s=_validate_exposure(exposure_s),
    )


def calibrate_table(table, zp: ZeroPoint, *, exposure_s: float | None = None) -> np.ndarray:
    """把源表的流量转成定标星等，曝光时间**取自 ``zp.exposure_s``**。

    ``exposure_s`` 只是一个允许调用方显式复述的覆盖参数，``None``（默认）表示
    「不复述、沿用零点」。给出而与 ``zp.exposure_s`` 不一致时抛 ``ValueError``，
    因为静默采用其中一个会造成 ``2.5*log10(exposure_s)`` 量级的整体平移
    （0.08 s 时是 2.7423 mag）而没有任何痕迹。零点未记曝光而调用方给出一个值，
    同样算不一致。
    """
    if exposure_s is not None:
        if zp.exposure_s is None or not np.isclose(
            float(exposure_s), float(zp.exposure_s), rtol=0.0, atol=1e-12
        ):
            raise ValueError(
                f"曝光时间与零点记录不一致: 零点 {zp.exposure_s!r}，调用方 {exposure_s!r}"
            )
    return zp.apply(instrumental_mag(table.flux, exposure_s=zp.exposure_s))


def limiting_magnitude(
    table, zp: ZeroPoint, *, percentile: float = 95.0
) -> float:
    """返回定标星等分布的第 ``percentile`` 个百分位，一个经验暗端指标。

    这**不是**可复现性极限，也**不是**探测完备性极限：它只对**单帧**的源表取
    百分位，全程没有任何跨帧比对或注入-回收。报告里引用的正是这个百分位数。
    命名上不要叫它「最暗可复现源」。

    空表与全 ``nan``（整帧流量都 <= 0，扣背景失败时的形态）都返回 ``nan``，且不
    留 ``RuntimeWarning``。曝光时间沿用 ``zp.exposure_s``。
    """
    if len(table) == 0:
        return float("nan")
    mags = calibrate_table(table, zp)
    if not np.isfinite(mags).any():
        return float("nan")
    return float(np.nanpercentile(mags, percentile))


@dataclass
class PhotometryReport:
    """一次测光的对外报告。

    三个星等字段在 ``to_dict()`` 里显式转 ``float``：numpy 标量能过 ``json.dumps``
    是偶然的——``np.float64`` 是 ``float`` 的子类所以侥幸通过，而 ``np.float32``
    与 ``np.int64`` 都抛 ``TypeError``。不转换就是一个潜伏 bug，只要有一条代码
    路径产出 float32 流量列就立刻发作。
    """

    zero_point: ZeroPoint | None
    n_calibrated: int = 0
    mag_min: float | None = None
    mag_max: float | None = None
    limiting_mag: float | None = None
    note: str = ""

    def to_dict(self) -> dict:
        """返回可直接 ``json.dumps`` 的字典；``None`` 原样保留。"""
        return {
            "zero_point": self.zero_point.to_dict() if self.zero_point else None,
            "n_calibrated": int(self.n_calibrated),
            "mag_min": None if self.mag_min is None else float(self.mag_min),
            "mag_max": None if self.mag_max is None else float(self.mag_max),
            "limiting_mag": None if self.limiting_mag is None else float(self.limiting_mag),
            "note": self.note,
        }
