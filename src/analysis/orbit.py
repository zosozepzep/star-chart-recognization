"""圆轨道高度区间反演（派生数据集 2）。

**本模块给出的是自相容高度区间，不是定轨结果。** 单站、单弧段、无距离测量，
信息量不足以定出六个轨道要素；能定量给出的只有「在圆轨道假设下，哪些高度能产生
实测到的角速度」这一区间。

两个界的几何含义
-----------------
圆轨道速度 ``v`` 垂直于地心矢 ``r_sat``。视线单位矢与 ``r_sat`` 的夹角
``theta_sat`` 由正弦定理给出 ``sin(theta_sat) = R*cos(E) / (R + h)``。轨道面相对
测站铅垂面的取向**未知**，所以真正产生角运动的横向速度分量只能界定在区间内：

    v*cos(theta_sat)  <=  |v_垂直视线|  <=  v
    ^                                      ^
    轨道面含测站铅垂面                     轨道面垂直于铅垂面

测站自转速度向东、量值 ``omega_E*R*cos(纬度)``，它垂直视线的分量**符号随过境
方位而变**（dataset B 实测 ``AZIMUTH`` 从 233.04 度扫到 283.97 度，且 f08/f79
回跳），所以**不做带符号相减**，而是并入区间：下界再减、上界再加
``omega_E*R*cos(纬度)/rho``，下界夹 0。h=1080、E=15.784004 处实测测站项
27.8973 ''/s，占裸**下**界 342.3724 的 **8.15%**、占裸**上**界 602.6412 的
**4.63%**（两个分母都写出来，只写一个百分数会被追问）。

dataset B 的成品带
------------------
角速度 770.150746 ''/s（53 对独立步）、仰角 15.784004 度（轨迹 f17-f70 的
``ELEVATIO`` 均值）、测站纬度 43.312 度：

| 量 | 实测 |
|---|---|
| 成品宽带 | **[213.4411, 848.3592] km**，宽 **634.9181 km**，带比值 **3.9747** |
| 斜距 | 668.8702 / 2079.7271 km |
| 周期 | 88.7654 / 101.8947 min |
| 裸带（无测站项，**只是中间量**） | [258.3995, 804.1031] km，宽 545.7037 |
| 测站项贡献的宽度 | +89.2145 km |

误差预算四项**全宽口径**（全书统一）：几何 **634.9181** km，噪声 **15.7048** km
（``rate_std/mean = 5.908433/770.150746`` 折算——**用四舍五入的 0.007672 会得
15.7053**），选帧 **14.9325** km（窗内仰角**两端** 15.43476/15.95376 度，跨度
0.519 度），投影 **0.1637** km（R397(c) 实测均匀比例尺偏置 +0.016%）。几何项比其余
三项之**和** 30.8011 km 大 **20.61 倍**，所以 ``dominant`` 恒为 ``geometry``。

**投影项曾按 +1.4% 估作 14.11 km，那个上限（R303）在本模块内作废。** 修正一个
被高估的误差项让「几何主导」这个结论更强，而不是更弱——这正是不该拿旧数凑安全
边界的理由。

界与高度的对应关系（容易弄反）
------------------------------
两个界都随高度**单调递减**；高度 ``h`` 可行 <=> ``lo(h) <= omega <= hi(h)``。
所以 ``lo(h) = omega`` 的根是 **height_min**（比它更低，连最慢的取向都太快），
``hi(h) = omega`` 的根是 **height_max**。**弄反会算出负带宽**：若算出的带宽是
负数，是对应关系反了，不是符号笔误，回去重推，不要取绝对值。

单调性实测（6000 点，[200, 40000] km，E=15.7775）四个函数全部 ``n_increase = 0``：
``topocentric_rate_max`` 2540.7378 -> 13.6698、``topocentric_rate_min`` 913.9129 ->
13.5496、宽带上界 2651.1964 -> 15.2478、宽带下界 803.4542 -> 11.9716。后两个才是
``estimate_height`` 真正反演的函数。唯一性前提因此成立，**不要**加二分回退或多根
搜索。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

from src.config import load_config

# --------------------------------------------------------------------------
# 物理常数（自然常数，不是可调阈值，故留在代码里）
# --------------------------------------------------------------------------

MU_EARTH_KM3_S2 = 398600.4418        # 地球引力常数 GM，km^3/s^2
R_EARTH_KM = 6378.137                # WGS-84 赤道半径，km
OMEGA_EARTH_RAD_S = 7.2921159e-5     # 地球自转角速度，rad/s
ARCSEC_PER_RAD = 180.0 * 3600.0 / np.pi
SECONDS_PER_DAY = 86400.0

# --------------------------------------------------------------------------
# 实测常数（都注明来源；只进误差预算与 note，不进反演本身）
# --------------------------------------------------------------------------

#: 均匀比例尺在 dataset B 轨迹跨度上的固有偏高，R397(c) 在 Task 19 交付轮实测。
#: 早先按 R303 的 +1.4% 上限估作 14.11 km，那个数在本模块内作废：实测小 87 倍。
PROJECTION_SCALE_BIAS = 1.6e-4

#: 角速度的相对散度 ``rate_std/mean``，dataset B 交付实测（``ddof=0``，53 对：
#: 5.908433 / 770.150746）。**写成商而不是四舍五入的 0.007672**：后者让噪声项从
#: 15.7048 km 变成 15.7053 km，误差预算就对不上台账了。只作为 ``estimate_height``
#: 的缺省值；``estimate_from_trajectory`` 会用轨迹自己的两个量覆盖它。
RATE_FRAC_STD_DEFAULT = 5.908433 / 770.150746

#: 跟踪窗内 ``ELEVATIO`` 的两个端点（度），dataset B 轨迹 f17-f70 实测 min 15.43476 /
#: max 15.95376，跨度 0.519000。
#:
#: **必须是两个端点，不能只留跨度。** 窗内仰角分布是偏的：均值 15.784004 而
#: ``(min+max)/2 = 15.694260``，差 0.089744 度。按「均值 ± 半跨度」算选帧项实测
#: 得 14.9165 km，按真实两端算得 **14.9325 km**——后者才是台账的数。
ELEVATION_WINDOW_DEG_DEFAULT = (15.43476, 15.95376)

MODEL_NOTE = "圆轨道假设 + 轨道面取向未知，故给出高度区间；非定轨结果"

_BRACKET_CACHE: tuple[float, float] | None = None


def default_bracket_km() -> tuple[float, float]:
    """``orbit.bracket_km``：``brentq`` 的搜索区间，取自 ``src/config/default.yaml``。

    结果缓存一次，避免每次反演都读一遍 YAML。
    """
    global _BRACKET_CACHE
    if _BRACKET_CACHE is None:
        lo, hi = load_config()["orbit"]["bracket_km"]
        _BRACKET_CACHE = (float(lo), float(hi))
    return _BRACKET_CACHE


# --------------------------------------------------------------------------
# 几何
# --------------------------------------------------------------------------


def slant_range_km(height_km: float, elevation_deg: float) -> float:
    """测站到圆轨道目标的斜距（km），球形地球、测站在海平面。

    ``rho = -R*sin(E) + sqrt((R+h)^2 - R^2*cos(E)^2)``。**曲率项不可丢**：丢掉它
    （即令 ``rho = h/sin(E)`` 之类）实测让 ``height_max`` 从 848.3592 掉到无根。
    仰角 90 度时退化为高度本身，实测 ``slant_range_km(1080, 90) = 1080.000``；
    仰角 15.784004 度时同一高度的斜距是 2762.0894 km，是高度的 2.56 倍——
    **把斜距当成高度用**实测把带整体推到 [1002.4935, 1944.1552]。
    """
    height_km = float(height_km)
    if height_km <= 0.0:
        raise ValueError("高度必须为正（km），收到 %.6f" % height_km)
    a = R_EARTH_KM + height_km
    cos_e = np.cos(np.deg2rad(float(elevation_deg)))
    sin_e = np.sin(np.deg2rad(float(elevation_deg)))
    return float(-R_EARTH_KM * sin_e + np.sqrt(a * a - R_EARTH_KM * R_EARTH_KM * cos_e * cos_e))


def orbital_speed_km_s(height_km: float) -> float:
    """圆轨道环绕速度 ``sqrt(mu/(R+h))``，km/s。

    **半径是 R+h，不是 R。** 在地表算（即 ``sqrt(mu/R)``）实测把带推到
    [219.8279, 916.8212]——它的 ``height_min`` 落在 (200, 230) 内，只靠
    ``height_max`` 才拒得掉，这就是数据集断言收到 ±3 km 的理由。
    """
    height_km = float(height_km)
    if height_km <= 0.0:
        raise ValueError("高度必须为正（km），收到 %.6f" % height_km)
    return float(np.sqrt(MU_EARTH_KM3_S2 / (R_EARTH_KM + height_km)))


def cos_theta_sat(height_km: float, elevation_deg: float) -> float:
    """视线与地心矢夹角 ``theta_sat`` 的余弦。

    ``sin(theta_sat) = R*cos(E) / (R + h)``，**分母是 R+h，不是 R**：用 R 实测让
    ``height_min`` 无根。这个量**同时依赖高度与仰角**，0.568 只是 h=1080 那一行；
    E=15.7775 实测 h=400 -> 0.42427、800 -> 0.51851、1080 -> 0.56808、
    2000 -> 0.68066。
    """
    height_km = float(height_km)
    if height_km <= 0.0:
        raise ValueError("高度必须为正（km），收到 %.6f" % height_km)
    sin_theta = R_EARTH_KM * np.cos(np.deg2rad(float(elevation_deg))) / (R_EARTH_KM + height_km)
    return float(np.sqrt(max(0.0, 1.0 - sin_theta * sin_theta)))


def topocentric_rate_max(height_km: float, elevation_deg: float) -> float:
    """裸上界 ``v/rho``（''/s）：轨道面垂直于测站铅垂面，速度整根横穿视线。

    **这是横穿视线的极端，不是唯一值；报告须用 ``rate_bounds``。** 单用它会得出
    错误结论：实测同步轨道 35786 km @ 45 度的裸上界 16.9515 ''/s 比恒星速率
    15.0411 高 12.70%，于是「同步目标」会被排除掉——因为同步目标的视运动几乎全部
    来自测站自转（实测 ``omega_E*R*cos(lat)`` 0.338421 km/s 占 v_GEO 3.074661 的
    11.01%），必须并入测站项才判得对。保留本函数是为了单调性检验与量级对照。
    """
    return orbital_speed_km_s(height_km) / slant_range_km(height_km, elevation_deg) * ARCSEC_PER_RAD


def topocentric_rate_min(height_km: float, elevation_deg: float) -> float:
    """裸下界 ``v*cos(theta_sat)/rho``（''/s）：轨道面含测站铅垂面。

    **``cos(theta_sat)`` 不可丢**：丢掉它下界就等于上界，实测 ``height_min`` 从
    213.4411 跳到 760.5519，而 ``height_max`` 逐位不变——只有 ``height_min`` 那条
    断言拒得掉它。
    """
    return (
        orbital_speed_km_s(height_km)
        * cos_theta_sat(height_km, elevation_deg)
        / slant_range_km(height_km, elevation_deg)
        * ARCSEC_PER_RAD
    )


def site_rotation_rate_arcsec_s(
    height_km: float, elevation_deg: float, *, site_lat_deg: float
) -> float:
    """测站自转在视线横向的**量值**上界 ``omega_E*R*cos(纬度)/rho``（''/s）。

    ``cos`` 里是**测站纬度**，不是仰角：用仰角实测让 ``height_min`` 无根
    （E=15.7775 处 ``omega_E*R*cos(elev)`` = 0.447578 km/s 而
    ``omega_E*R*cos(lat)`` = 0.338421，大 1.32 倍）。
    """
    return (
        OMEGA_EARTH_RAD_S
        * R_EARTH_KM
        * np.cos(np.deg2rad(float(site_lat_deg)))
        / slant_range_km(height_km, elevation_deg)
        * ARCSEC_PER_RAD
    )


def rate_bounds(
    height_km: float, elevation_deg: float, *, site_lat_deg: float
) -> tuple[float, float]:
    """给定高度与仰角，实测角速度可能落入的区间（''/s），已并入测站自转项。

    下界再**减**、上界再**加** ``site_rotation_rate_arcsec_s``，下界夹 0：
    测站项的符号随过境方位而变，不能带符号相减。三种写错法的实测后果：
    加到**两个**界上 -> ``height_min`` 304.7285（``height_max`` 逐位不变）；
    从上界**减** -> ``height_max`` 760.5519（``height_min`` 逐位不变）；
    整项丢掉 -> 退化成裸带 [258.3995, 804.1031]。所以两条端点断言互补、缺一不可。
    """
    site = site_rotation_rate_arcsec_s(height_km, elevation_deg, site_lat_deg=site_lat_deg)
    lo = topocentric_rate_min(height_km, elevation_deg) - site
    hi = topocentric_rate_max(height_km, elevation_deg) + site
    return (max(0.0, float(lo)), float(hi))


def orbital_period_min(height_km: float) -> float:
    """圆轨道周期（分钟）。"""
    a = R_EARTH_KM + float(height_km)
    return float(2.0 * np.pi * np.sqrt(a ** 3 / MU_EARTH_KM3_S2) / 60.0)


def height_from_mean_motion(mean_motion_rev_day: float) -> float:
    """由平均运动（圈/天）反算圆轨道高度（km）。

    实测参考量：n=13.479 -> 1079.99、n=15.00 -> 566.90、n=14.50 -> 725.65、
    n=2.0 -> 20232.09 km；反向 n(200 km)=16.2723、n(2000 km)=11.3209、
    n(1800 km)=11.7387。**本函数不做 SGP4**，只用开普勒第三定律取量级。
    """
    n = float(mean_motion_rev_day)
    if not np.isfinite(n) or n <= 0.0:
        raise ValueError("平均运动必须是正的有限值（圈/天），收到 %r" % (mean_motion_rev_day,))
    period_s = SECONDS_PER_DAY / n
    a = (MU_EARTH_KM3_S2 * (period_s / (2.0 * np.pi)) ** 2) ** (1.0 / 3.0)
    return float(a - R_EARTH_KM)


# --------------------------------------------------------------------------
# 反演
# --------------------------------------------------------------------------


def _invert(rate_arcsec_s: float, bound, bracket_km: tuple[float, float]) -> float | None:
    """在 ``bracket_km`` 内解 ``bound(h) = rate``，无根返回 ``None``。

    ``bound`` 在区间内严格单调递减（见模块 docstring 的 6000 点实测），所以变号
    即唯一根，**不加**二分回退或多根搜索。
    """
    lo, hi = bracket_km

    def residual(h: float) -> float:
        return bound(h) - rate_arcsec_s

    if residual(lo) * residual(hi) > 0.0:
        return None
    return float(brentq(residual, lo, hi, xtol=1e-9, rtol=1e-12))


@dataclass(frozen=True)
class OrbitEstimate:
    """一次高度区间反演的产出。

    ``height_km`` 是区间**算术中点**，只供单值展示；报告与 PPT 必须报区间。
    ``error_budget`` 在收敛时由 ``estimate_height`` 填好（不收敛时为 ``None``），
    作为字段而不是 ``to_dict()`` 内的现算量：重算它需要测站纬度与两个散度输入，
    而那三个都不是本数据类的字段，现算会把它们再传一遍。
    """

    rate_arcsec_s: float
    elevation_deg: float
    bracket_km: tuple[float, float]
    converged: bool
    height_min_km: float | None = None
    height_max_km: float | None = None
    height_km: float | None = None          # 区间算术中点，仅供单值展示
    range_min_km: float | None = None
    range_max_km: float | None = None
    period_min_min: float | None = None
    period_max_min: float | None = None
    velocity_km_s: float | None = None      # 取 height_km 处的环绕速度
    residual_arcsec_s: float | None = None
    note: str = ""
    error_budget: dict | None = field(default=None)

    @property
    def band_width_km(self) -> float | None:
        """区间宽度（km）。**为负说明两个界的对应关系反了**，见模块 docstring。"""
        if self.height_min_km is None or self.height_max_km is None:
            return None
        return float(self.height_max_km - self.height_min_km)

    def to_dict(self) -> dict:
        """可直接 ``json.dumps(..., allow_nan=False)`` 的记录。

        所有数值显式转成**内建** ``float``：``np.float64`` 能过
        ``json.dumps`` 是偶然的（它是 ``float`` 的子类），而 ``np.bool_`` 与
        ``np.int64`` 实测都抛 ``TypeError``。
        """
        def clean(value) -> float | None:
            if value is None:
                return None
            value = float(value)
            return value if np.isfinite(value) else None

        return {
            "model": MODEL_NOTE,
            "converged": bool(self.converged),
            "rate_arcsec_s": clean(self.rate_arcsec_s),
            "elevation_deg": clean(self.elevation_deg),
            "bracket_km": [float(self.bracket_km[0]), float(self.bracket_km[1])],
            "height_min_km": clean(self.height_min_km),
            "height_max_km": clean(self.height_max_km),
            "band_width_km": clean(self.band_width_km),
            "height_km": clean(self.height_km),
            "range_min_km": clean(self.range_min_km),
            "range_max_km": clean(self.range_max_km),
            "period_min_min": clean(self.period_min_min),
            "period_max_min": clean(self.period_max_min),
            "velocity_km_s": clean(self.velocity_km_s),
            "residual_arcsec_s": clean(self.residual_arcsec_s),
            "error_budget": None if self.error_budget is None else dict(self.error_budget),
            "note": str(self.note),
        }


def _error_budget(
    rate_arcsec_s: float,
    elevation_deg: float,
    height_min_km: float,
    height_max_km: float,
    *,
    site_lat_deg: float,
    rate_frac_std: float,
    elevation_window_deg: tuple[float, float],
    bracket_km: tuple[float, float],
) -> dict:
    """四项误差预算，**全宽口径**（全书统一），按实际输入现算。

    - ``geometry_km``：轨道面取向未知 + 测站自转符号未定，就是带宽本身，主导项。
    - ``noise_km``：把角速度上下各推 ``rate_frac_std``，看 ``height_max`` 移动多少。
      dataset B 实测 ``5.908433/770.150746`` -> **15.7048 km**（用四舍五入的
      0.007672 会得 15.7053 km）。
    - ``elevation_km``：``height_max`` 在窗内仰角**两个端点**处之差。dataset B 实测
      15.43476 -> 838.3006 km、15.95376 -> 853.2331 km，差 **14.9325 km**（宽带
      口径；同一算法在裸带上是 14.4208 km）。**不能用「均值 ± 半跨度」**：窗内
      分布是偏的（均值 15.784004 vs 中点 15.694260），那样算得 14.9165 km。
    - ``projection_km``：``PROJECTION_SCALE_BIAS`` 折算，实测 **0.1637 km**。

    上界比下界敏感得多（噪声在 ``height_min`` 上只移 5.0104 km），所以三项都取
    ``height_max`` 的位移作为全宽，口径统一、可比。
    """
    def h_max(rate: float, elev: float) -> float | None:
        return _invert(rate, lambda h: rate_bounds(h, elev, site_lat_deg=site_lat_deg)[1],
                       bracket_km)

    def spread(a: float | None, b: float | None) -> float:
        if a is None or b is None:
            return float("nan")
        return float(abs(a - b))

    elev_lo, elev_hi = float(elevation_window_deg[0]), float(elevation_window_deg[1])
    geometry_km = float(height_max_km - height_min_km)
    noise_km = spread(h_max(rate_arcsec_s * (1.0 - rate_frac_std), elevation_deg),
                      h_max(rate_arcsec_s * (1.0 + rate_frac_std), elevation_deg))
    elevation_km = spread(h_max(rate_arcsec_s, elev_lo), h_max(rate_arcsec_s, elev_hi))
    projection_km = spread(h_max(rate_arcsec_s * (1.0 + PROJECTION_SCALE_BIAS), elevation_deg),
                           height_max_km)
    rest = [noise_km, elevation_km, projection_km]
    dominant = "geometry" if all(geometry_km > x or not np.isfinite(x) for x in rest) else "other"
    return {
        "geometry_km": geometry_km,
        "noise_km": noise_km,
        "elevation_km": elevation_km,
        "projection_km": projection_km,
        "dominant": dominant,
    }


def estimate_height(
    rate_arcsec_s: float,
    elevation_deg: float,
    *,
    site_lat_deg: float,
    rate_frac_std: float = RATE_FRAC_STD_DEFAULT,
    elevation_window_deg: tuple[float, float] = ELEVATION_WINDOW_DEG_DEFAULT,
    bracket_km: tuple[float, float] | None = None,
) -> OrbitEstimate:
    """由角速度与仰角反演圆轨道的**自相容高度区间**。

    ``site_lat_deg`` 是必需关键字参数：测站项 ``omega_E*R*cos(纬度)`` 需要真实
    纬度，**不得从仰角凑**。``rate_frac_std`` 与 ``elevation_window_deg`` 只进误差
    预算与 ``note``，不影响区间本身；缺省值是 dataset B 交付实测量，
    ``estimate_from_trajectory`` 会用真实值覆盖。

    ``height_min_km`` 是 ``rate_bounds`` **下界** = 角速度的根，``height_max_km``
    是**上界**的根（见模块 docstring：弄反会算出负带宽）。

    不收敛的四条分支各给一句不同的中文 ``note``，便于诊断：角速度非有限、非正、
    低于/高于搜索区间内可达的范围；另有两条「界落在搜索区间外」的分支——测站纬度
    过低时会走到（实测纬度 10 度处宽带下界在 200 km 上仍低于实测角速度）。
    """
    bracket = default_bracket_km() if bracket_km is None else (
        float(bracket_km[0]), float(bracket_km[1])
    )
    rate = float(rate_arcsec_s)
    elevation_deg = float(elevation_deg)

    def unresolved(note: str) -> OrbitEstimate:
        return OrbitEstimate(
            rate_arcsec_s=rate,
            elevation_deg=elevation_deg,
            bracket_km=bracket,
            converged=False,
            note=note,
        )

    if not np.isfinite(rate):
        return unresolved("角速度不是有限值（%r），无法反演高度" % (rate_arcsec_s,))
    if rate <= 0.0:
        return unresolved("角速度非正（%.6f ''/s），无法反演高度" % rate)

    reach_lo = rate_bounds(bracket[1], elevation_deg, site_lat_deg=site_lat_deg)[0]
    reach_hi = rate_bounds(bracket[0], elevation_deg, site_lat_deg=site_lat_deg)[1]
    if rate < reach_lo:
        return unresolved(
            "角速度 %.4f ''/s 低于搜索区间 [%.1f, %.1f] km 内可达的最小值 %.4f ''/s，"
            "目标可能比 %.1f km 更高或不是圆轨道"
            % (rate, bracket[0], bracket[1], reach_lo, bracket[1])
        )
    if rate > reach_hi:
        return unresolved(
            "角速度 %.4f ''/s 高于搜索区间 [%.1f, %.1f] km 内可达的最大值 %.4f ''/s，"
            "目标可能比 %.1f km 更低或不是轨道目标"
            % (rate, bracket[0], bracket[1], reach_hi, bracket[0])
        )

    height_min_km = _invert(
        rate, lambda h: rate_bounds(h, elevation_deg, site_lat_deg=site_lat_deg)[0], bracket
    )
    height_max_km = _invert(
        rate, lambda h: rate_bounds(h, elevation_deg, site_lat_deg=site_lat_deg)[1], bracket
    )
    if height_min_km is None:
        return unresolved(
            "区间下界不在搜索区间 [%.1f, %.1f] km 内：低至 %.1f km 都仍自相容，"
            "下界受搜索区间截断，不报区间"
            % (bracket[0], bracket[1], bracket[0])
        )
    if height_max_km is None:
        return unresolved(
            "区间上界不在搜索区间 [%.1f, %.1f] km 内：高至 %.1f km 都仍自相容，"
            "上界受搜索区间截断，不报区间"
            % (bracket[0], bracket[1], bracket[1])
        )

    mid_km = 0.5 * (height_min_km + height_max_km)
    lo_at_min = rate_bounds(height_min_km, elevation_deg, site_lat_deg=site_lat_deg)[0]
    hi_at_max = rate_bounds(height_max_km, elevation_deg, site_lat_deg=site_lat_deg)[1]
    budget = _error_budget(
        rate,
        elevation_deg,
        height_min_km,
        height_max_km,
        site_lat_deg=site_lat_deg,
        rate_frac_std=float(rate_frac_std),
        elevation_window_deg=(float(elevation_window_deg[0]), float(elevation_window_deg[1])),
        bracket_km=bracket,
    )
    note = (
        "圆轨道自相容高度区间 [%.4f, %.4f] km（宽 %.4f km，带比值 %.4f）；"
        "仰角均值 %.6f 度、窗内跨度 %.6f 度（%.5f–%.5f）；测站纬度 %.5f 度；"
        "轨道面取向未知且测站自转符号未定，故报区间不报单值"
        % (
            height_min_km,
            height_max_km,
            height_max_km - height_min_km,
            height_max_km / height_min_km,
            elevation_deg,
            float(elevation_window_deg[1]) - float(elevation_window_deg[0]),
            float(elevation_window_deg[0]),
            float(elevation_window_deg[1]),
            float(site_lat_deg),
        )
    )
    return OrbitEstimate(
        rate_arcsec_s=rate,
        elevation_deg=elevation_deg,
        bracket_km=bracket,
        converged=True,
        height_min_km=height_min_km,
        height_max_km=height_max_km,
        height_km=float(mid_km),
        range_min_km=slant_range_km(height_min_km, elevation_deg),
        range_max_km=slant_range_km(height_max_km, elevation_deg),
        period_min_min=orbital_period_min(height_min_km),
        period_max_min=orbital_period_min(height_max_km),
        velocity_km_s=orbital_speed_km_s(mid_km),
        residual_arcsec_s=float(max(abs(lo_at_min - rate), abs(hi_at_max - rate))),
        note=note,
        error_budget=budget,
    )


def estimate_from_trajectory(
    trajectory,
    headers,
    *,
    site_lat_deg: float | None = None,
) -> OrbitEstimate:
    """从一条轨迹与整个 ``headers`` 列表反演高度区间。

    **仰角按 ``point.frame`` 索引，不对传进来的整个 ``headers`` 取均值。**
    「传哪一批 header」是个真实自由度，实测三种合理传法给出三个不同的上界：

    | 传进去的 header 集合 | 仰角均值 | ``height_max`` | 与全 80 帧之差 |
    |---|---|---|---|
    | ``seq.headers``（全 80 帧） | 15.700203 | 845.9494 | — |
    | 跟踪段 f16-f70（55） | 15.777502 | 848.1723 | +2.22 km |
    | **目标轨迹 f17-f70（54）** | **15.784004** | **848.3592** | **+2.41 km** |

    轨迹自己带着帧号（实测 ``[pt.frame for pt in trj.points] ==
    list(range(trk.start, trk.end + 1))`` 为 True），所以这个自由度应当被**消掉**，
    而不是记录下来：调用方传全 80 帧还是只传 54 帧，结果都必须一样。

    ``site_lat_deg`` 缺省取轨迹首帧 header 的 ``site_lat_deg``（两数据集同站，
    实测 43.312 度）。角速度的相对散度与仰角跨度都取轨迹的**真实**值，只在轨迹
    不带 ``rate_std_arcsec_s`` 时退回模块缺省。
    """
    frames = [int(pt.frame) for pt in trajectory.points]
    if not frames:
        raise ValueError("轨迹为空，无法反演高度区间")
    n_headers = len(headers)
    if max(frames) >= n_headers:
        raise ValueError(
            "帧号超出 headers 长度：最大帧号 %d，headers 仅 %d 个" % (max(frames), n_headers)
        )
    picked = [headers[f] for f in frames]        # 由轨迹决定，不由调用方决定
    elevations = np.array([float(h.elevation_deg) for h in picked], dtype=np.float64)
    elevation_deg = float(np.mean(elevations))
    # 窗内两个端点，不是「均值 ± 半跨度」：窗内分布是偏的，见 _error_budget。
    elevation_window_deg = (float(elevations.min()), float(elevations.max()))
    if site_lat_deg is None:
        site_lat_deg = float(picked[0].site_lat_deg)

    rate = float(trajectory.mean_rate_arcsec_s)
    std = getattr(trajectory, "rate_std_arcsec_s", None)
    if std is not None and np.isfinite(std) and np.isfinite(rate) and rate > 0.0:
        rate_frac_std = float(std) / rate
    else:
        rate_frac_std = RATE_FRAC_STD_DEFAULT

    return estimate_height(
        rate,
        elevation_deg,
        site_lat_deg=float(site_lat_deg),
        rate_frac_std=rate_frac_std,
        elevation_window_deg=elevation_window_deg,
    )


# --------------------------------------------------------------------------
# 离线 TLE 自相容校验
# --------------------------------------------------------------------------

_TLE_LINE_WIDTH = 69
_TLE_NORAD_SLICE = slice(2, 7)
_TLE_MEAN_MOTION_SLICE = slice(52, 63)     # 第 2 行第 53-63 列，0-based 切片


@dataclass(frozen=True)
class TLEReference:
    """一条公开两行根数里本模块用得到的部分。

    只读平均运动，**不做 SGP4 定轨**：本校验回答的只是「反演出的高度区间是否与
    公开根数自相容」。
    """

    name: str
    norad_id: int
    mean_motion_rev_day: float

    @property
    def height_km(self) -> float:
        """由平均运动反算的圆轨道高度（km）。"""
        return height_from_mean_motion(self.mean_motion_rev_day)


def load_tle(path, norad_id: int) -> TLEReference:
    """从两行根数文件里取出一颗目标的平均运动。

    只读第 2 行第 53-63 列（0-based 切片 ``[52:63]``，实测真实格式行
    ``ln[2:7] -> '60385'``、``ln[52:63] -> '13.5432109 '``）。``#`` 开头的行与空行
    跳过；紧邻 ``1 ...`` 之前的非编号行当作目标名。

    找不到该 NORAD 号抛 ``KeyError``；行格式不合法抛**中文** ``ValueError``，
    不让 ``float()`` 的英文错误冒上来。
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    name = ""
    for raw in text.splitlines():
        line = raw.rstrip("\n")
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if line.startswith("1 "):
            continue
        if line.startswith("2 "):
            if line[_TLE_NORAD_SLICE].strip() != str(int(norad_id)):
                continue
            field_text = line[_TLE_MEAN_MOTION_SLICE].strip()
            try:
                mean_motion = float(field_text)
            except ValueError:
                raise ValueError(
                    "TLE 第 2 行的平均运动字段不合法：NORAD %d 的第 53-63 列取到 %r，"
                    "原行长 %d（合法 TLE 行宽 %d，抄录时不要重排空格）"
                    % (int(norad_id), field_text, len(line), _TLE_LINE_WIDTH)
                ) from None
            if not np.isfinite(mean_motion) or mean_motion <= 0.0:
                raise ValueError(
                    "TLE 第 2 行的平均运动必须为正：NORAD %d 取到 %r"
                    % (int(norad_id), field_text)
                )
            return TLEReference(
                name=name or "NORAD %d" % int(norad_id),
                norad_id=int(norad_id),
                mean_motion_rev_day=mean_motion,
            )
        name = line.strip()
    raise KeyError(
        "未在 %s 中找到 NORAD %d 的两行根数；容器无网络，需人工从 celestrak 抄入"
        % (path.name, int(norad_id))
    )


def cross_check_tle(estimate: OrbitEstimate, reference) -> dict:
    """把反演出的**区间**与公开根数的高度比一比。

    判的是「参考高度是否落在 ``[height_min_km, height_max_km]`` 内」，**不对点值
    判**，也**不设百分比容差**：15% 既不是噪声预算（实测 15.7048 km = 中点的
    ±1.48%）也不是模型预算（实测 634.9181 km = 中点的 ±59.80%），它没有来源。

    ``margin_km`` 带内为 0、带外为到最近端的距离（实测 n=2.0 的 20232.09 km 离
    带上界 848.3592 有 19383.73 km）。``bool(...)`` 那层包裹**必须保留**：实测
    ``json.dumps(np.bool_(True))`` 抛 ``TypeError``（numpy 2.2.6 的已知不对称，
    ``np.float64`` 可序列化，``np.bool_`` 与 ``np.int64`` 不行）。

    **不收敛时判定只能是「不一致」**，且不抛异常：没有区间就没有「落在区间内」。
    """
    reference_height_km = float(reference.height_km)
    lo = estimate.height_min_km
    hi = estimate.height_max_km
    if not estimate.converged or lo is None or hi is None:
        return {
            "agrees": False,
            "estimated_height_min_km": None,
            "estimated_height_max_km": None,
            "reference": str(reference.name),
            "reference_height_km": reference_height_km,
            "inside_band": False,
            "margin_km": None,
            "note": "反演未收敛（%s），没有区间可比：参考高度 %.2f km 无法判定自相容"
                    % (estimate.note, reference_height_km),
        }

    lo, hi = float(lo), float(hi)
    inside = bool(lo <= reference_height_km <= hi)
    margin_km = 0.0 if inside else float(min(abs(reference_height_km - lo),
                                            abs(reference_height_km - hi)))
    note = (
        "参考高度 %.2f km（%s，平均运动 %.6f 圈/天）%s反演区间 [%.4f, %.4f] km"
        % (
            reference_height_km,
            reference.name,
            float(reference.mean_motion_rev_day),
            "落在" if inside else "在",
            lo,
            hi,
        )
    )
    if inside:
        note += "内，自相容"
    else:
        note += "之外 %.2f km；三种可能原因：目标确实更低、ELEVATIO 记录的是机架" \
                "指向而非目标真实仰角、或圆轨道假设在这一弧段不成立" % margin_km
    return {
        "agrees": bool(inside),
        "estimated_height_min_km": lo,
        "estimated_height_max_km": hi,
        "reference": str(reference.name),
        "reference_height_km": reference_height_km,
        "inside_band": inside,
        "margin_km": margin_km,
        "note": note,
    }
