"""目标轨迹时序表与角速度。

本模块把一条 `Track` 与它的帧时标合成一张按帧排列的时序表，并给出角速度、
总弧长与时长三个聚合量。它是派生数据集 1（目标轨迹）的产出端。

**真值隔离（硬约束）**：本模块不得 import `src.validate.truth`。像素→角秒的
比例尺只以一个 `float` 形参传入，由调用方从 `TruthReport.plate_scale_arcsec_px`
取（数据集 B 实测 6.179047，逐步比值的均值）。判决链一旦引用真值就变成
「用真值算出真值」，在答辩中是致命的原理性质疑。

**本轮不报的三个量**：`ra_deg` / `dec_deg` / `pa_deg` 恒为 `None`，理由见
`TrajectoryPoint` 的 docstring。位置轴报的是**天球系像素坐标**，角速度与弧长
报角秒，赤经赤纬不报。

口径（PPT 与答辩照此说）：arcsec 量是**真值定标量**——比例尺由 `.DAT` 真值定出
（FITS 头 28 张卡片里没有焦距、没有像元尺寸、没有 WCS）。可以说「像素系
125.73 px/帧由识别链独立测得，换算所用比例尺来自真值定标」，**不得**声称角速度
独立验证了真值。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.astrometry.photometry import ZeroPoint, instrumental_mag


def angular_rates(xy, times, scale_arcsec_px: float) -> np.ndarray:
    """逐点角速度（角秒/秒），长度与 ``xy`` 相同。

    ``xy`` 是 ``(N, 2)`` ``float64`` 的 ``[x, y]`` 点集（0-based，x=列、y=行），
    ``times`` 是与之同长的**相对秒**序列，``scale_arcsec_px`` 是像素→角秒的
    标量比例尺。

    ``times`` 必须是**相对秒**（``(t - t[0]).sec``）：传绝对 Unix 秒会让 1e-9
    量级的一致性丢失。绝对 Unix 秒量级 1.7e9，float64 在那里的分辨率约
    2.4e-07 s，于是 1.008 s 的节拍在 1.007999897..1.008000135 之间抖动——实测
    20 点匀速夹具上均值的相对误差 **2.59e-09**、速率散度 **9.008e-05**，而相对秒
    走 ``Time`` 内部的 jd1/jd2 双精度对、不做绝对历元的大数相减，同一夹具实测
    4.02e-14 与 3.412e-09。真实数据（dataset B 54 帧）上两条时基的均值逐位相同，
    所以只有那个 1e-9 夹具能分开它们。

    **只需要一个标量比例尺，不需要任何定向信息。** 旋转加均匀缩放是共形映射、
    翻转是正交变换，两者都严格保模长，所以角速度与弧长上定向完全约掉。实测
    （20 点夹具，步长 ``(125.73, -5.0)``）：

    | ``rotation_deg`` | ``flip`` | 经 ``pixel_to_offset`` 与标量直乘的最大差 | ``total_arc`` |
    |---|---|---|---|
    | 0.0 | False | **1.137e-13** | 14772.495055 |
    | 0.0 | True | 1.137e-13 | 14772.495055 |
    | +34.0 | False | 1.137e-13 | 14772.495055 |
    | +34.0 | True | 1.137e-13 | 14772.495055 |
    | -20.0 | False | 1.137e-13 | 14772.495055 |
    | -20.0 | True | 1.137e-13 | 14772.495055 |

    这张表留在这里是因为：将来有人看到「角速度不带定向信息」可能会"补回"一个
    定向形参，以为修了什么。它什么也不修，而改动比例尺的取得路径会重新打开
    「构造后赋值」那个洞（见 ``Trajectory.scale_arcsec_px``）。

    **角速度不可为负。** 时间步守卫写 ``dt > 0.0`` 而非 ``dt != 0.0``：实测
    ``times=[0, 5, 2]``、步长 100 px 时，前者给 ``[123.5800, 123.5800, nan]``，
    后者给 ``[nan, 123.5800, -205.9667]``——最后一项是个负角速度，它会静默拉低
    均值而没有任何报错。

    ``rate[0]`` 由 ``rate[1]`` 前向填充，只为让速率列与点列同长便于制表；
    ``Trajectory`` 的统计量取的是 ``rate[1:]`` 这 n-1 个独立步速率，不含它。
    退化输入：``n=0`` 给空数组，``n=1`` 给 ``[nan]``，全零时标给全 ``nan``。
    """
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    times = np.asarray(times, dtype=np.float64).ravel()
    n = len(xy)
    if len(times) != n:
        raise ValueError(f"坐标与时标数量不一致：坐标 {n} 点，时标 {len(times)} 个")

    rate = np.full(n, np.nan, dtype=np.float64)
    if n < 2:
        return rate

    step_px = np.hypot(*np.diff(xy, axis=0).T)
    dt = np.diff(times)
    # dt > 0.0 而非 dt != 0.0：逆序时标必须给 nan，不能给负角速度。
    ok = dt > 0.0
    rate[1:][ok] = step_px[ok] * float(scale_arcsec_px) / dt[ok]
    rate[0] = rate[1]        # 前向填充，只为对齐点列长度
    return rate


@dataclass(frozen=True)
class TrajectoryPoint:
    """轨迹上的一个时刻。

    ``time_isot`` 只给人读与 ``to_records()`` 用，**不参与任何算术**：``isot``
    默认 3 位小数，实测节拍 1.0075 s 时经字符串往返损失 5.0e-04 s（1.008 /
    1.009 / 0.030 s 的节拍下恰为 0，所以这个损失只在非三位小数的节拍上现形）。
    ``duration_s`` 由 ``time_unix`` 这个绝对时刻相减，实测 19 s 跨度误差
    1.2e-08。**角速度用相对秒，时刻记录用绝对秒。**

    ``x_sky``/``y_sky`` 是**天球系像素坐标**（配准后），``x_det``/``y_det`` 是
    探测器系像素坐标；都是 0-based，x=列、y=行。

    **``ra_deg`` / ``dec_deg`` / ``pa_deg`` 在 MVP 里恒为 ``None``，这不是占位
    偷懒，是本轮范围的边界。** 实测 dataset A 首帧 FITS 头共 28 行卡片
    （``SITELONG/SITELATI/SITEALTI``、``AZIMUTH/ELEVATIO`` 双字段字符串、
    ``DATE-OBS``、``EXPOSURE``、``IMAGETYP``、``X_DEL/Y_DEL/WENDU/SHIDU/DAQIYA/
    HENGYAO/ZONGYAO/HX``），**没有赤经赤纬、没有焦距、没有像元尺寸、没有 WCS**：

    - ``ra_deg``/``dec_deg`` 需要盲板求解（国赛 Task 18）给出切点，本模块没有
      任何绝对定向来源。``offset_to_radec`` 已作为纯函数实现好等着它。
    - ``pa_deg`` 需要视场旋转与翻转标定（国赛 Task 16）。实测像素系轨迹方向
      ``atan2(off_x, off_y) % 360`` 均值 **95.6670°**（std 0.7767），而真值位置角
      （自北经东）均值 **321.9317°**（std 1.4255），两者相差 **226.2648°**——那
      226° 整个落在未解的翻转+旋转里。一个标着「位置角」而实际从 +x 起算的列是
      直接的原理性质疑靶子，所以**本模块不猜，报 ``None``**。

      国赛回来实现时照这张实测表，不要重新推（``atan2(off_x, off_y) % 360``，
      自北经东）：

      | 方向 | ``atan2(ox, oy) % 360``（**正确**） | ``atan2(oy, ox) % 360``（数学角，错） |
      |---|---|---|
      | +y（北） | **0°** | 90° |
      | +x（东） | **90°** | 0° |
      | -y（南） | **180°** | 270° |
      | -x（西） | **270°** | 180° |

    三个字段保留而不删除，是因为 Task 26 的绘图夹具与国赛的 Task 18 都按名字
    构造它们。
    """

    frame: int
    time_isot: str
    time_unix: float
    x_sky: float
    y_sky: float
    x_det: float
    y_det: float
    rate_arcsec_s: float
    flux: float
    mag: float | None = None
    ra_deg: float | None = None
    dec_deg: float | None = None
    pa_deg: float | None = None


def _finite_or_nan(values: np.ndarray, reducer) -> float:
    """在有有限值时归约，否则返回 ``nan``——且**不留 ``RuntimeWarning``**。

    实测 ``nanmean``/``nanstd`` 在全 ``nan`` 数组上返回 ``nan`` 并各打一条
    ``RuntimeWarning``（共两条）。全零时标或长度 1 的轨迹就会走到这里。
    """
    if values.size and np.isfinite(values).any():
        return float(reducer(values))
    return float("nan")


@dataclass
class Trajectory:
    """一条目标轨迹的时序表。

    ``scale_arcsec_px`` 是像素→角秒的比例尺，三条硬要求：

    (a) 名字**不带前导下划线**——带下划线的名字仍然是公开 dataclass 字段，会进
        ``__init__``/``__eq__``/``__repr__``，伪装成私有只会骗读者。
    (b) 声明在 ``source`` 之后、所有方法之前，读者先看到声明再看到用处。
    (c) 由 ``build_trajectory`` 作为**正常构造参数**传入，**不得**构造后
        ``trj.scale_arcsec_px = ...`` 赋值。实测在构造后赋值的写法下，任何直接
        ``Trajectory(points=..., source=...)`` 构造出来的对象 ``total_arc_arcsec``
        会静静地用 ``1.0`` 算，少乘 6.179 倍，无任何报错。默认值留 ``1.0`` 的
        含义是「未定标，弧长单位为像素」。
    """

    points: list[TrajectoryPoint] = field(default_factory=list)
    source: str = ""
    scale_arcsec_px: float = 1.0

    def __len__(self) -> int:
        return len(self.points)

    @property
    def _independent_rates(self) -> np.ndarray:
        """n-1 个独立步速率，**不含** ``points[0]`` 那个前向填充值。

        把填充值算进来等于给第一步双倍权重。数据集 B 实测两者相差
        770.150748（53 对）vs 769.890998（含填充的 54 项），0.26 ″/s。
        """
        return np.array([p.rate_arcsec_s for p in self.points[1:]], dtype=np.float64)

    @property
    def mean_rate_arcsec_s(self) -> float:
        """角速度均值（53 对口径，见 ``_independent_rates``）。"""
        return _finite_or_nan(self._independent_rates, np.nanmean)

    @property
    def rate_std_arcsec_s(self) -> float:
        """角速度散度，``ddof=0`` 的总体标准差（``np.nanstd`` 默认）。

        复算这个数必须用 ``ddof=0``。数据集 B 实测 5.908426（53 对）。
        """
        return _finite_or_nan(self._independent_rates, np.nanstd)

    @property
    def total_arc_arcsec(self) -> float:
        """天球系逐步弧长之和 × ``scale_arcsec_px``。

        用逐步累加而非首末距离：目标轨迹接近直线但并不严格是直线，两者在数据集 B
        上分别是 41176.0477″ 与 6663.2580 px × 6.179 = 41172.2707″。
        """
        if len(self.points) < 2:
            return 0.0
        xy = np.array([[p.x_sky, p.y_sky] for p in self.points], dtype=np.float64)
        steps = np.hypot(*np.diff(xy, axis=0).T)
        return float(steps.sum() * float(self.scale_arcsec_px))

    @property
    def duration_s(self) -> float:
        """首末**绝对** Unix 秒之差，不经 ``isot`` 字符串往返。"""
        if len(self.points) < 2:
            return 0.0
        return float(self.points[-1].time_unix - self.points[0].time_unix)

    def to_records(self) -> list[dict]:
        """返回可直接 ``json.dumps(..., allow_nan=False)`` 的记录列表。

        所有数值字段显式转成**内建** ``float``/``int``，不是 numpy 标量：
        ``np.float64`` 能过 ``json.dumps`` 是偶然的（它是 ``float`` 的子类），
        而 ``np.float32``/``np.int64`` 实测都抛 ``TypeError: Object of type
        float32 is not JSON serializable``。断言这一点必须用 ``type(x) is float``
        而不是 ``isinstance``。

        ``time_utc``（人读）与 ``time_unix``（算术用）两列同时输出。非有限的
        ``rate_arcsec_s`` / ``mag`` 降为 ``None``：``json.dumps`` 默认会把 ``nan``
        渲染成裸 ``NaN`` token，那不是合法 JSON（``jq``/``JSON.parse``/Go 全部
        拒收），``allow_nan=False`` 时直接抛 ``ValueError``。三个 ``None`` 字段
        原样保留。
        """
        def clean(value) -> float | None:
            if value is None:
                return None
            value = float(value)
            return value if np.isfinite(value) else None

        return [
            {
                "frame": int(p.frame),
                "time_utc": str(p.time_isot),
                "time_unix": float(p.time_unix),
                "x_sky": float(p.x_sky),
                "y_sky": float(p.y_sky),
                "x_det": float(p.x_det),
                "y_det": float(p.y_det),
                "rate_arcsec_s": clean(p.rate_arcsec_s),
                "flux": float(p.flux),
                "mag": clean(p.mag),
                "ra_deg": None if p.ra_deg is None else float(p.ra_deg),
                "dec_deg": None if p.dec_deg is None else float(p.dec_deg),
                "pa_deg": None if p.pa_deg is None else float(p.pa_deg),
            }
            for p in self.points
        ]


def build_trajectory(
    track,
    sequence,
    scale_arcsec_px: float,
    *,
    zero_point: ZeroPoint | None = None,
) -> Trajectory:
    """把一条 ``Track`` 与序列时标合成时序表。

    ``scale_arcsec_px`` 由调用方提供（MVP 里取自真值定标的
    ``TruthReport.plate_scale_arcsec_px``，实测 6.179047）；本模块不读真值。

    **``sequence.times()[frames]`` 是位置索引，而 ``track.frames`` 装的是
    ``FrameHeader.index``。** ``times()`` 返回按 ``date_obs`` 排序的 ``Time``
    数组，索引是**位置**。本项目两个数据集上实测 ``argsort(date_obs) ==
    arange(n)`` 均为 ``True``（B 的 80 帧、A 的 36 帧），所以位置恰好等于 index，
    此处直接下标是对的。**前提写在这里**：若某数据集的文件名排序与时标排序不一致，
    此处必须改为按 index 查表。这个错误很隐蔽——实测帧号错 1 帧只带来
    0.126–0.169 ″/s 的偏移，数据集断言抓不到。

    帧号上界守卫用 ``len(sequence.times())`` 而非 ``len(sequence)``：前者正是被
    下标的那个数组的长度，而且不要求 ``sequence`` 实现 ``__len__``（测试替身
    只有 ``dataset_id`` 与 ``times()``）。

    星等只在给出 ``zero_point`` 时产出，且**必须沿用零点自带的曝光**
    （``exposure_s=zero_point.exposure_s``）。裸调 ``instrumental_mag(flux)`` 取
    默认 ``exposure_s=None`` 即不做曝光归一，实测差量：0.030 s（跟踪窗实测值）
    → **-3.8072 mag**，0.080 s → -2.7423 mag，而调用方无从察觉。dataset B 的
    ``EXPOSURE`` 卡片实测帧 0–7 为 80 ms、帧 8–79 为 30 ms，跟踪窗 f17–f70
    全为 30 ms、均匀，所以本函数不必处理窗内变曝光。

    ``offset_to_radec`` 在本轮**不被本函数调用**（没有盲板求解提供切点），
    三个天球量报 ``None``。
    """
    frames = [int(f) for f in track.frames]
    if not frames:
        raise ValueError("轨迹为空，无法构建时序表")

    times = sequence.times()
    n_times = len(times)
    if max(frames) >= n_times:
        raise ValueError(
            "帧号超出序列长度：最大帧号 %d，序列仅 %d 帧" % (max(frames), n_times)
        )

    picked = times[frames]
    # 相对秒供角速度用；绝对 Unix 秒供 duration_s 与记录用。
    rel_s = (picked - picked[0]).sec
    unix_s = picked.unix
    isot = picked.isot

    xy_sky = np.asarray(track.xy_sky, dtype=np.float64).reshape(-1, 2)
    xy_det = np.asarray(track.xy_det, dtype=np.float64).reshape(-1, 2)
    flux = np.asarray(track.flux, dtype=np.float64).ravel()
    rates = angular_rates(xy_sky, rel_s, scale_arcsec_px)

    if zero_point is None:
        mags: list[float | None] = [None] * len(frames)
    else:
        calibrated = zero_point.apply(
            instrumental_mag(flux, exposure_s=zero_point.exposure_s)
        )
        mags = [float(m) for m in calibrated]

    points = [
        TrajectoryPoint(
            frame=frames[i],
            time_isot=str(isot[i]),
            time_unix=float(unix_s[i]),
            x_sky=float(xy_sky[i, 0]),
            y_sky=float(xy_sky[i, 1]),
            x_det=float(xy_det[i, 0]),
            y_det=float(xy_det[i, 1]),
            rate_arcsec_s=float(rates[i]),
            flux=float(flux[i]),
            mag=mags[i],
        )
        for i in range(len(frames))
    ]
    return Trajectory(
        points=points,
        source=str(getattr(sequence, "dataset_id", "")),
        scale_arcsec_px=float(scale_arcsec_px),
    )


def offset_to_radec(ra0_deg, dec0_deg, off_x_arcsec, off_y_arcsec):
    """切平面偏移（角秒）→ 赤经赤纬，严格 TAN（gnomonic）反投影。

    不用线性近似 ``ra0 + Δx/cos(dec0)``：实测在本项目真实轨迹跨度上
    （端点大圆 11.646240°，切点取真值第 0 行 (209.23627, 12.35454)，端点偏移
    (-25614.6558, +33930.7378)″）线性式误差 **1189.4823 角秒 = 192.50 px**，
    而严格 TAN 的往返误差是 **1.15e-11 角秒**量级。视场本身 4096 px ×
    6.179 ″/px = **7.0303 度 = 421.82 角分**——注意不是「7 角分」，那个说法错
    60 倍。

    本函数在 MVP 里**不被 ``build_trajectory`` 调用**（无盲板求解提供
    ``ra0/dec0``），作为纯函数实现并测试，供国赛的 Task 18 消费。届时注意两条
    实测陷阱：(a) 盲板求解返回的是**机架指向 = 画面中心**，而实测 dataset B
    目标在 f17 位于 (2271.6, 1958.8)、画面中心 (2047.5, 2047.5)，相差
    241.0 px = 1489.2″ = **0.4137°**，把中心指向当成目标位置会让整条轨迹平移
    这么多；(b) 以真值首行锚定则依赖 ``track.frames[0]`` 恰为真值第 0 行那一帧，
    实测真值一步中位数 **778.777″ = 0.2163°**（mean 776.494 / min 761.389 /
    max 786.980），少一帧就整体偏这么多。

    赤经取模到 ``[0, 360)``；赤纬不取模（TAN 反投影的 ``arctan`` 天然落在
    ``(-90, 90)``）。
    """
    xi = np.deg2rad(np.asarray(off_x_arcsec, dtype=np.float64) / 3600.0)
    eta = np.deg2rad(np.asarray(off_y_arcsec, dtype=np.float64) / 3600.0)
    d0 = np.deg2rad(float(dec0_deg))
    denom = np.cos(d0) - eta * np.sin(d0)
    ra = np.deg2rad(float(ra0_deg)) + np.arctan2(xi, denom)
    dec = np.arctan((np.sin(d0) + eta * np.cos(d0)) / np.hypot(denom, xi))
    return np.degrees(ra) % 360.0, np.degrees(dec)
