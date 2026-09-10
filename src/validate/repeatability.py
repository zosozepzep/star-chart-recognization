"""跨帧可复现性——区分真源与噪声的核心手段，也是星点计数唯一可辩护的依据。

原理：真实恒星在配准后的天球坐标系中位置固定，会在多帧的同一位置重复出现；
噪声峰值位置随机，几乎不可能在多帧同一位置复现。因此"能在 >= min_fraction
的帧中被重复探测到的源数"才是可报的星数，而不是单帧的原始探测数。

这条判据直接回应本项目历史上 263822 颗星、单帧 9021 个源的问题：那些数字
在 2σ 附近，复现率不足 0.1，绝大部分是噪声。

真值隔离（硬约束）：本模块**不得** import ``src.validate.truth``。可复现性是
对着数据自身量的——同一颗星是否在多帧里重现——引入真值会让报出的星数变成
答案文件的函数。正因如此这个指标在无 ``.DAT`` 的数据集 A 上依然成立，
它是唯一一个不需要答案的科学性论据。

两个必须分开的量
----------------

本模块报两个比值，它们**不同名不同义**，只在「参考帧探测数恰等于各帧均值」
时相等（裁决 66）：

- ``reproducibility = n_reproducible / n_reference``，其中
  ``n_reference = len(frames[0])``。回答「你怎么知道这些是真星」——参考帧列出的
  源里，有多大比例能在足够多的帧里复现。**恒在 [0, 1]。**
- ``purity = n_reproducible / n_detected_mean``，其中 ``n_detected_mean`` 是
  **非空帧**的平均探测数。回答「别人多报了」——一帧典型探测里有多大比例是真源。

两者不可共用一个名字，因为 purity 会给出反直觉的结果：

1. 它能超过 1.0。六帧各 100 颗真星给出 1.0000；把第 5 帧改成一无所获，含空帧的
   均值降到 83.3333，纯度报成 **1.2000**——一个彻底失败的帧把质量指标抬高 20%。
   本实现因此把**空帧从纯度分母里排除**（空帧在命中统计里本来就被跳过，让它
   只影响分母是不自洽的），排除后同一情形给出 purity = 1.0000。
2. 帧间探测数不等时它会误报失败。参考帧 100 源、其余五帧各 200 源、参考帧的每一
   个源都在全部 6 帧复现：真复现率 1.000，而 purity = 100/183.3333 = **0.545**，
   读起来像 45% 没复现。

规格 ``docs/superpowers/specs/2026-09-08-太空目标识别系统.md`` 第 78 行的
``| 5.0σ, npix=5 | 124 | 90.3% |`` 与第 374 行都称之为「复现率」，但原公式量的
是 purity；``0.903 x 124 = 112`` 这个反推只在参考帧探测数恰为均值时对得上，
而没有任何实测支持这一点。**所以 90.3% 究竟是哪一个量，规格没有交代。**
两条曲线的实测值见 ``docs/reports/measurements.md`` 的「阈值-复现率曲线」，
报告同时列出 reproducibility 与 purity，不做二者的取舍。

两个已知的、按设计保留的偏差
----------------------------

- **计数是下界。** 参考帧法的上限是 ``len(frames[0])``：100 颗真星而参考帧只探到
  80 时，``n_reproducible`` 就是 80，无论其余帧看到多少。在接近探测下限的 sigma
  上，这一项单靠参考帧的漏检率就把计数压低。方向对保守论证有利（宁少报不多报），
  所以如实记录而不"修正"。
- **匹配是最近邻而非互为最近邻**，所以可能多对一：相距 1 px 的两个参考源可被同一
  个探测同时匹配。按 ``chance_match_probability`` 给出的密度量级这可以忽略，
  故不引入双向匹配。
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from src.calib.background import model_background
from src.detect.segmentation import detect_sources_in_frame

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RepeatabilityPoint:
    """一个阈值档位上的双分子双分母记录。

    ``reproducibility`` 与 ``purity`` 分子相同、分母不同，见模块 docstring。
    """

    n_sigma: float
    n_detected_mean: float
    n_reproducible: int
    n_reference: int
    reproducibility: float
    purity: float


@dataclass
class RepeatabilityCurve:
    points: list[RepeatabilityPoint] = field(default_factory=list)
    frames: list[int] = field(default_factory=list)
    match_radius_px: float = 3.0

    def to_dict(self) -> dict:
        return {
            "frames": list(self.frames),
            "match_radius_px": self.match_radius_px,
            "points": [
                {
                    "n_sigma": p.n_sigma,
                    "n_detected_mean": p.n_detected_mean,
                    "n_reproducible": p.n_reproducible,
                    "n_reference": p.n_reference,
                    "reproducibility": p.reproducibility,
                    "purity": p.purity,
                }
                for p in self.points
            ],
        }

    def recommended(self, min_reproducibility: float = 0.85) -> RepeatabilityPoint:
        """取「自身及所有更高阈值都达标」的最低阈值——在可信的前提下尽量多探测。

        判据是**单调合格后缀**而不是合格集里的最小 sigma。理由：一个上邻居不达标
        的点不是操作点。反例（裁决 69）——曲线 ``[(2.0, 0.10), (3.0, 0.88),
        (4.0, 0.60), (5.0, 0.91)]``、门限 0.85，合格集是 ``[3.0, 5.0]``，取最小
        会选中 3.0，而紧邻其上的 4.0 只有 0.60。后缀判据给出 5.0。
        良态（复现率随 sigma 非降）曲线上两者结果相同。

        选的是 ``reproducibility`` 而不是 ``purity``：参数名就是
        ``min_reproducibility``，两者在帧探测数不等时会给出不同排序。
        """
        ordered = sorted(self.points, key=lambda p: p.n_sigma)
        best: RepeatabilityPoint | None = None
        # 从高 sigma 往低走，一旦遇到不达标的档就停：此前累积的即是合格后缀。
        for point in reversed(ordered):
            if point.reproducibility < min_reproducibility:
                break
            best = point
        if best is None:
            raise ValueError(
                f"没有阈值达到复现率 {min_reproducibility}；最高为 "
                f"{max((p.reproducibility for p in self.points), default=float('nan')):.3f}"
            )
        return best


def min_hits_for(min_fraction: float, n_frames: int) -> int:
    """把「至少多少比例的帧要看到它」换算成命中帧数下限。

    返回值**含参考帧本身**，与 ``cross_frame_repeatability`` 的计数约定一致
    （参考帧自身记 1 次命中）。这个 off-by-one 是裁决 284 整段分歧的来源：
    「>= 4 of 6 帧」对应 ``min_hits = 4``，即 5 个邻居帧里要有 3 个匹配上，
    而不是 4 个。6 帧下的换算是 0.5→3、**0.6→4**、0.9→6、1.0→6。

    下限 2：跨帧复现至少要有两帧看到，否则判据退化成单帧探测。
    """
    return max(2, int(math.ceil(min_fraction * n_frames)))


def expected_neighbours_per_source(
    n_detections: int, match_radius_px: float, image_shape
) -> float:
    """匹配半径内的期望邻居数（均匀密度近似）：``n * pi * r^2 / A``。

    ``image_shape`` 按 NumPy 约定是 ``(ny, nx)``；面积只用两者之积，所以行列
    次序在这里无影响。面积必须取自数据：同样 2284 个探测在 4096² 场里给出
    0.00385，在 2048² 场里是 0.01540（恰 4 倍）。而 4096x4136 只挪到 0.00381
    ——**面积的量级要紧，具体帧尺寸不要紧**。
    """
    ny, nx = int(image_shape[0]), int(image_shape[1])
    area = float(ny) * float(nx)
    if area <= 0.0:
        raise ValueError(f"图幅面积必须为正: image_shape={tuple(image_shape)}")
    if match_radius_px <= 0.0:
        raise ValueError(f"匹配半径必须为正: {match_radius_px}")
    return float(n_detections) * math.pi * match_radius_px ** 2 / area


def single_frame_match_probability(
    n_detections: int, match_radius_px: float, image_shape
) -> float:
    """某个位置在**一个**帧里至少有一个偶然邻居的概率 ``1 - exp(-lambda)``。"""
    return 1.0 - math.exp(
        -expected_neighbours_per_source(n_detections, match_radius_px, image_shape)
    )


def chance_match_probability(
    n_detections: int,
    match_radius_px: float,
    image_shape,
    *,
    n_frames: int,
    min_hits: int,
) -> float:
    """偶然满足复现判据的源数**期望值**——这就是「为什么只报 10² 颗星」的论证本身。

    ``min_hits`` **含参考帧本身**（与 ``cross_frame_repeatability`` 的
    ``hits >= min_hits`` 约定一致，其中参考帧自记 1 次），所以 ``n_frames - 1``
    个邻居帧里需要至少 ``min_hits - 1`` 次偶然匹配。调用方应当把
    ``min_hits_for(min_fraction, n_frames)`` 的结果传进来，不要写死数字，
    否则两处的判据会各自漂移。

    实测（4096² 场、3 px 半径、6 帧、``min_hits=4`` 即「>=4 of 6」）：

    ==================  ===================
    探测数/帧           E[偶然可复现源]
    ==================  ===================
    124（报数阈值）     1.13e-8
    2284（2σ）          1.29e-3
    ==================  ===================

    判据松一档，数字涨好几个量级：2284 源/帧时 ``min_hits`` 取 4/5/6 分别给出
    1.29e-3 / 2.48e-6 / 1.91e-9。**1.9e-9 是 min_hits=6 那一档**（每一个邻居帧
    都偶然匹配），不是本模块用的 >=4 of 6 判据下的值；后者诚实的数字是 1.29e-3
    （裁决 284）。2σ 上真实可复现源约 201 个，故余量是 ``201 / 1.29e-3`` ≈
    **5.2 个量级**，不是九个（裁决 285）。

    这是均匀密度 Poisson 估计，而真实源是成团的，所以它是**量级下界**而非精确
    速率。5.2 个量级的余量足以让这个近似不承重——「3 像素半径是不是太松」因此
    有一个算得出的答案，而不是一句保证。
    """
    if n_frames < 2:
        raise ValueError(f"偶然重合估计至少需要 2 帧: n_frames={n_frames}")
    if not 2 <= min_hits <= n_frames:
        raise ValueError(f"min_hits 必须落在 [2, {n_frames}]，实际 {min_hits}")

    p_one = single_frame_match_probability(n_detections, match_radius_px, image_shape)
    n_neighbours = n_frames - 1
    needed = min_hits - 1
    tail = sum(
        math.comb(n_neighbours, k) * p_one ** k * (1.0 - p_one) ** (n_neighbours - k)
        for k in range(needed, n_neighbours + 1)
    )
    return float(n_detections) * tail


def cross_frame_repeatability(
    sky_positions,
    *,
    match_radius: float = 3.0,
    min_fraction: float = 0.6,
) -> tuple[int, float, float]:
    """统计在天球坐标系中跨帧重复出现的源数，返回 ``(源数, 复现率, 纯度)``。

    以第一帧为参考清单，逐帧统计命中（参考帧自记 1 次），命中数达到
    ``min_hits_for(min_fraction, len(frames))`` 的源计入 ``n_reproducible``。
    两个比值的定义、它们为什么必须分开、以及为什么计数是下界，见模块 docstring。

    空帧在命中统计里被跳过（它无法证实也无法否证任何源），因此也**不进纯度的
    分母**——只让它压低分母会使一个彻底失败的帧把纯度抬高 20%（裁决 66）。
    """
    frames = [np.asarray(p, dtype=np.float64).reshape(-1, 2) for p in sky_positions]
    if len(frames) < 2:
        raise ValueError("可复现性统计至少需要 2 帧")

    reference = frames[0]
    if len(reference) == 0:
        # 参考帧为空时统计**未定义**而非为零：其余帧可能满是真源，只是没有
        # 参考清单可比。Step 5 因此不得挑一个探测数为零的帧做参考帧。
        logger.warning("reference frame is empty; repeatability is undefined")
        return 0, 0.0, 0.0

    hits = np.ones(len(reference), dtype=int)
    for other in frames[1:]:
        if len(other) == 0:
            continue
        dist, _ = cKDTree(other).query(reference)
        # 最近邻、非互为最近邻：可能多对一，按实测偶然密度可忽略（见模块 docstring）。
        hits += (dist < match_radius).astype(int)

    min_hits = min_hits_for(min_fraction, len(frames))
    n_reproducible = int(np.count_nonzero(hits >= min_hits))

    n_reference = len(reference)
    reproducibility = n_reproducible / n_reference

    non_empty = [len(f) for f in frames if len(f) > 0]
    mean_detected = float(np.mean(non_empty))
    purity = n_reproducible / mean_detected if mean_detected > 0 else 0.0

    logger.info(
        "repeatability: n=%d of %d reference sources (hits>=%d of %d frames), "
        "reproducibility=%.4f purity=%.4f",
        n_reproducible,
        n_reference,
        min_hits,
        len(frames),
        reproducibility,
        purity,
    )
    return n_reproducible, float(reproducibility), float(purity)


def scan_thresholds(
    sequence,
    frames,
    registration,
    *,
    sigmas,
    npixels: int = 5,
    hot_clusters=None,
    match_radius_px: float = 3.0,
    min_fraction: float = 0.6,
    background_kwargs: dict | None = None,
) -> RepeatabilityCurve:
    """对一串阈值分别做"探测数"与"可复现数"双曲线。

    背景模型逐帧建一次、跨全部 sigma 复用：阈值只改 ``n_sigma * rms`` 这个乘数，
    背景本身与阈值无关，而建模是整条流水线最贵的一步（实测 4096² 单帧约 1.48 s）。
    """
    frames = list(frames)
    background_kwargs = dict(background_kwargs or {})

    models = {}
    images = {}
    for idx in frames:
        images[idx] = sequence.image(idx)
        models[idx] = model_background(images[idx], **background_kwargs)

    points: list[RepeatabilityPoint] = []
    for sigma in sigmas:
        sky_lists = []
        counts = []
        for idx in frames:
            table = detect_sources_in_frame(
                images[idx],
                models[idx],
                n_sigma=float(sigma),
                npixels=npixels,
                hot_clusters=hot_clusters,
                frame=idx,
            )
            counts.append(len(table))
            sky_lists.append(registration.to_sky(idx, table.xy))
        n_rep, rep, purity = cross_frame_repeatability(
            sky_lists, match_radius=match_radius_px, min_fraction=min_fraction
        )
        non_empty = [c for c in counts if c > 0]
        points.append(
            RepeatabilityPoint(
                n_sigma=float(sigma),
                n_detected_mean=float(np.mean(non_empty)) if non_empty else 0.0,
                n_reproducible=n_rep,
                n_reference=int(counts[0]),
                reproducibility=rep,
                purity=purity,
            )
        )

    points.sort(key=lambda p: p.n_sigma)
    return RepeatabilityCurve(
        points=points, frames=frames, match_radius_px=match_radius_px
    )
