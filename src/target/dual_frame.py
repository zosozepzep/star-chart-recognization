"""双坐标系交叉确认。

判决要求三条独立证据同时成立：
  1. 探测器系近静止（v_det < v_det_max）——机架在跟踪它；
  2. 天球系高速运动（v_sky > v_sky_min）——它相对恒星背景在动；
  3. 非像素锁定——坐标确实在变，不是热像素/坏列那种整数不动的伪迹。

第 3 条是关键的防质疑设计。前两条对热像素同样成立（热像素在探测器系完全
不动，在天球系随配准反推出 126 px/帧的"运动"），若缺这一条，坏点会被当成
最高置信度的空间目标——它的 v_det 恰好是 0，`classify` 给出的静止性对比度
是 `inf`，即可能的最高分。

播种策略：从每一帧的每一个源都尝试起一条轨迹。只从首帧播种会漏掉中途进场
的目标——实测中目标在 f17 才出现，从 f16 播种加 6-12 px 半径搜索得到 0 个候选。
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np
from scipy.spatial import cKDTree

from src.detect.segmentation import SourceTable
from src.register.solver import RegistrationResult


@dataclass
class Track:
    frames: list[int]
    xy_det: np.ndarray
    xy_sky: np.ndarray
    flux: np.ndarray
    peak: np.ndarray
    elongation: np.ndarray
    lock_span_px: float = 1.0

    @property
    def length(self) -> int:
        return len(self.frames)

    @property
    def start(self) -> int:
        return self.frames[0]

    @property
    def end(self) -> int:
        return self.frames[-1]

    @property
    def drift_det_px(self) -> float:
        return float(np.hypot(*(self.xy_det[-1] - self.xy_det[0])))

    @property
    def drift_sky_px(self) -> float:
        return float(np.hypot(*(self.xy_sky[-1] - self.xy_sky[0])))

    @property
    def v_det_px(self) -> float:
        return self._step_median(self.xy_det)

    @property
    def v_sky_px(self) -> float:
        return self._step_median(self.xy_sky)

    @property
    def pixel_locked(self) -> bool:
        """坐标在探测器系几乎完全不变——热像素、坏列的特征。

        判据是探测器系 2-D 跨度的**绝对**范数对 `lock_span_px`（默认 1.0 px），
        而不是任何与 `drift_sky_px` 成比例的量。

        为什么不能按天球漂移缩放：热像素的定义性质是"探测器上不动"，这是传感器
        的绝对属性，而按 `0.05 * drift_sky` 缩放会让同一个坏点随序列长度改变判决
        （10/30/54/80 帧的门限分别是 56.7/182.7/333.9/497.7 px）。更糟的是它把真
        目标也一并锁死：数据集 B 的目标在 54 帧上自身漂移 95.8 px，而门限
        0.05 x 6495 = 324.8 px，于是目标被判为像素锁定，`find_targets` 返回 0 个
        目标。没有任何 `lock_ratio` 取值能修好它——把数据集 B 的目标（比值
        0.01450）和热像素（0.00000）分开要求 `lock_ratio < 0.0145`，而这个数没有
        物理含义，余量还照样随序列长度变。

        为什么用范数而不是 `np.all(span < thr)`：`np.hypot(*span) >= max(span)` 恒
        成立，所以 `norm < thr` **蕴含** `np.all(span < thr)`——范数形式是两者中
        **更紧**的一个，标记为锁定的源严格更少。实测两种形式只在"各轴分量都小于
        门限而范数不小于门限"的源上分歧，例如跨度 (0.90, 0.90)（范数 1.273）与
        (0.80, 0.80)（范数 1.131）在 `np.all` 下是锁定、在范数下不是。等幅抖动的
        热像素正是这个形状，而漏掉一个热像素意味着把传感器缺陷以无穷大对比度报
        为目标，所以更紧的方向才是安全方向。代价为零：本任务所有真目标的单轴跨度
        都远在 1.0 px 之外（最慢的一例 19.80 px）。

        为什么不用规范里字面的 `np.round` 整数位置相等：实测跨在 x = 244.50 的热
        像素，其取整位置在**每一个**抖动量级下都在 244 与 245 之间跳（低至
        sigma = 0.007 px 仍如此），字面判据会把它放过去。

        1.0 px 的来处（实测配方：30 帧 x 6 次试验，9x9 窗口内 sigma=1.6 px 高斯加
        泊松噪声，量的是跨度范数）：静止源的质心本身就随光子噪声抖动，amp=200 时
        范数 0.136-0.181（最差约 0.18 px），amp=900 时 0.069-0.091，amp=3000 时
        0.035-0.045。最差静止情形 0.18 px 与本任务最慢真目标跨度 19.80 px 之间差
        109 倍，1.0 px 位于噪声地板之上 5.5 倍、最慢真目标之下 19.8 倍；[0.5, 19]
        区间内任何取值都可用，1.0 只是这条宽实测间隙里的一个圆整数，不是调出来的。

        注意 amp 依赖的形状，它划定了这条论证的边界：跨度范数大致按 1/sqrt(amp)
        下降，所以**更暗**的静止源抖得更厉害——外推 amp~20 会到 ~0.5 px、amp~5 会
        接近 1.0 px。但这样的源在本项目里低于 4 sigma 探测下限，根本到不了
        `build_tracks`，这才是该门限安全的理由。
        """
        span = np.abs(self.xy_det.max(axis=0) - self.xy_det.min(axis=0))
        return bool(np.hypot(*span) < self.lock_span_px)

    @property
    def peak_flat_ratio(self) -> float:
        """峰值平坦度 `(max-min)/median`——报告用的旁证，**从不参与判决**。

        刻意不与入 `pixel_locked` 做 AND：AND 只会削弱这道防线。实测两个逃逸
        case：紧贴溢出亮星的热像素给出约 0.67（半数帧峰值翻倍，(1800-900)/900），
        被移动目标穿过一次的热像素给出约 0.56（900 基线上一帧抬到 1400）——两者
        都远高于规范引用的 0.05，AND 进去会把它们判成"非锁定"，也就是报成目标。
        空间判据必须独立成立。

        用 `peak` 而不是 `flux`：热像素的定义性质是恒定的电子学电平，而积分流量
        还随分割器纳入的像素数变化。实测恒定电平源的足迹波动一个像素就把流量平坦
        度推到 0.1176（8<->9 px），7<->9 给 0.2500，5<->9 给 0.5714——纯分割噪声就
        超过 0.05。`src/calib/hotpixel.py` 的 `flat_ratio` 用的同样是像素值这类峰
        值量。

        中位数为 0（全零 `peak` 数组）时返回 0.0，否则会产出 nan/inf 并流进
        `to_dict()`，正是 Ruling 50 要堵的 JSON 隐患。
        """
        peak = np.asarray(self.peak, dtype=np.float64)
        if peak.size == 0:
            return 0.0
        median = float(np.median(peak))
        if not (median > 0.0):
            return 0.0
        return float((peak.max() - peak.min()) / median)

    def _step_median(self, xy: np.ndarray) -> float:
        """逐帧位移的**中位数**。长度不足 2 时返回 0.0。

        取中位数而非均值或首尾距离/N，是因为中位数不受单帧坏点影响。实测数据集 B
        的目标逐帧位移散布在 0.68–3.21 px（σ = 0.52，53 步），中位数 1.8440、
        均值 1.8497、首尾/N 1.8077 三者相差 2.2%，天球系一侧同为 126.0959 /
        125.7334 / 125.7218。`v_det_max` 与 `v_sky_min` 两个门限都是按中位数这个
        统计量校准的；三种统计量下所有余量都在 4.19 倍以上，所以没有任何判决取决于
        这个选择——但门限的校准口径是中位数，换统计量必须重测。

        `len(xy) < 2` 的守卫不是死代码，尽管默认参数下到不了：`np.diff` 在单行输入
        上给出空数组，`np.median([])` 返回 `nan` 并发两条 `RuntimeWarning`，而 `nan`
        会让**两条**速度判据同时静默失效（`nan > 8.0` 为 False，`nan < 30.0` 也为
        False），于是 `classify` 一条速度理由都不追加，`to_dict` 的
        `v_det_px_per_frame` / `v_sky_px_per_frame` 也带着 `nan` 出去——这两个字段
        没有 `stationarity_contrast` 那样的 `isfinite` 强转，
        `json.dumps(..., allow_nan=False)` 会抛 `ValueError`。

        长度 1 的轨迹从公开接口就能造出来：`find_targets(..., config=
        {"min_track_frames": 1})` 在合成场景上产生 3984 条长度 1 的轨迹。

        需要说清这个隐患的**条件性**，不要夸大：长度 1 的轨迹只有一行 `xy_det`，
        `max` 与 `min` 同行，跨度恰为 (0, 0)、范数恰为 0，因此只要 `lock_span_px > 0`
        它**必然** `pixel_locked` → `is_target=False` → `find_targets` 只返回命中项，
        永远不会把它吐出去（实测 `min_track_frames=1` 时仍只有 1 个命中）。`nan` 要
        真的进到报告里，需要一个把**被拒**判决也一并序列化的消费方，而 Task 30 会怎么
        消费这个字典尚未定。守卫仍然要留：它成本为零，且把隐患挡在产生端。
        """
        if len(xy) < 2:
            return 0.0
        d = np.diff(xy, axis=0)
        return float(np.median(np.hypot(d[:, 0], d[:, 1])))


@dataclass
class Verdict:
    track: Track
    is_target: bool
    v_det_px: float
    v_sky_px: float
    contrast: float
    pixel_locked: bool
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """交付报告用的 JSON 安全字典。

        `stationarity_contrast` 在非有限时（热像素的 v_det 恰为 0，对比度是
        `inf`）强制降为 `None`：`json.dumps` 会把 `inf` 渲染成裸 token
        `Infinity`，那不是合法 JSON，严格解析器会拒收；`allow_nan=False` 时
        `json.dumps` 直接抛 `ValueError`。Task 30 序列化这个字典，所以在这里
        闭掉。
        """
        contrast = float(self.contrast)
        return {
            "is_target": self.is_target,
            "frames": [self.track.start, self.track.end],
            "n_frames": self.track.length,
            "v_det_px_per_frame": self.v_det_px,
            "v_sky_px_per_frame": self.v_sky_px,
            "stationarity_contrast": contrast if math.isfinite(contrast) else None,
            "pixel_locked": self.pixel_locked,
            "peak_flat_ratio": self.track.peak_flat_ratio,
            "drift_det_px": self.track.drift_det_px,
            "xy_start": self.track.xy_det[0].tolist(),
            "xy_end": self.track.xy_det[-1].tolist(),
            "reasons": list(self.reasons),
        }


def build_tracks(
    detections: dict[int, SourceTable],
    frames,
    registration: RegistrationResult,
    *,
    radius: float = 8.0,
    min_frames: int = 10,
    lock_span_px: float = 1.0,
) -> list[Track]:
    """在探测器坐标系中做最近邻链接，从每一帧的每一个源播种。

    `claimed` 集合记下已完成轨迹上的每个 `(帧号, 源序号)`，使一个源不会同时属
    于两条轨迹。这不是可选项：每一帧都重新播种，去掉它之后合成场景的 4 条轨迹
    会碎成大量互相重叠的片段。

    诚实的局限（不在本任务范围内修，改了会动到已实测的基线）：链接是**贪心最近
    邻、无跳帧容忍**的，且只向前延伸——漏检一帧就把轨迹断成两截。数据集 B 的目标
    在 54 帧里帧帧被探到，所以这条路径上从未触发；但更暗的目标只要有一帧掉到
    4 sigma 以下，就会被切成两段都不足 `min_frames` 的碎片而整体丢失。

    返回的轨迹按长度降序排列，`find_targets` 依赖这个顺序。

    调用方须保证 `frames` 按时间升序且连续——本函数与 `register_sequence` 都不
    校验它。
    """
    frames = list(frames)
    trees = {i: cKDTree(detections[i].xy) for i in frames if len(detections[i])}
    claimed: set[tuple[int, int]] = set()
    tracks: list[Track] = []

    for si, seed_frame in enumerate(frames):
        table = detections[seed_frame]
        for k in range(len(table)):
            if (seed_frame, k) in claimed:
                continue
            path = [(seed_frame, k)]
            cx, cy = float(table.x[k]), float(table.y[k])
            for nxt in frames[si + 1 :]:
                tree = trees.get(nxt)
                if tree is None:
                    break
                dist, idx = tree.query([cx, cy])
                if dist >= radius:
                    break
                idx = int(idx)
                path.append((nxt, idx))
                cx = float(detections[nxt].x[idx])
                cy = float(detections[nxt].y[idx])
            if len(path) < min_frames:
                continue
            claimed.update(path)
            idxs = [i for i, _ in path]
            xy_det = np.array(
                [[detections[f].x[j], detections[f].y[j]] for f, j in path], dtype=np.float64
            )
            xy_sky = np.array(
                [registration.to_sky(f, xy_det[n : n + 1])[0] for n, (f, _) in enumerate(path)],
                dtype=np.float64,
            )
            tracks.append(
                Track(
                    frames=idxs,
                    xy_det=xy_det,
                    xy_sky=xy_sky,
                    flux=np.array([detections[f].flux[j] for f, j in path], dtype=np.float64),
                    peak=np.array([detections[f].peak[j] for f, j in path], dtype=np.float64),
                    elongation=np.array(
                        [detections[f].elongation[j] for f, j in path], dtype=np.float64
                    ),
                    lock_span_px=lock_span_px,
                )
            )

    tracks.sort(key=lambda t: -t.length)
    return tracks


def classify(
    track: Track,
    *,
    v_det_max: float = 8.0,
    v_sky_min: float = 30.0,
    lock_span_px: float = 1.0,
) -> Verdict:
    """对一条轨迹下三证据判决。

    关键字名与配置键刻意不同名：这里是 `v_det_max` / `v_sky_min` / `lock_span_px`，
    配置里是 `target.v_det_max_px` / `target.v_sky_min_px` / `target.lock_span_px`
    （前两个带 `_px` 后缀），由 `find_targets` 桥接。把任一侧改成与另一侧同名会
    让 `conf.get` 静默退回默认值，而直接调用 `classify` 的测试照样通过。

    已知的小瑕疵（保留现状，改它超出本任务范围）：本函数会**写回**所传入 `Track`
    的 `lock_span_px` 字段。`Track` 是普通 `@dataclass` 而非 frozen，所以能写；
    但同一条轨迹用不同门限判两次时，它只会静默保留最后一次的值。
    """
    track.lock_span_px = lock_span_px
    v_det = track.v_det_px
    v_sky = track.v_sky_px
    locked = track.pixel_locked
    contrast = v_sky / v_det if v_det > 0 else float("inf")

    reasons: list[str] = []
    if v_det > v_det_max:
        reasons.append(f"探测器系速度 {v_det:.2f} px/帧 超过上限 {v_det_max}，判为恒星")
    if v_sky < v_sky_min:
        reasons.append(f"天球系速度 {v_sky:.2f} px/帧 低于下限 {v_sky_min}，未见相对运动")
    if locked:
        reasons.append("像素锁定：坐标跨帧几乎不变，判为传感器缺陷而非空间目标")

    ok = not reasons
    if ok:
        reasons.append(
            f"双系确认通过：探测器系 {v_det:.2f} px/帧、天球系 {v_sky:.2f} px/帧，"
            f"静止性对比度 {contrast:.1f}x，坐标连续变化"
        )
    return Verdict(
        track=track,
        is_target=ok,
        v_det_px=v_det,
        v_sky_px=v_sky,
        contrast=contrast,
        pixel_locked=locked,
        reasons=reasons,
    )


def find_targets(
    detections: dict[int, SourceTable],
    frames,
    registration: RegistrationResult,
    *,
    config: dict | None = None,
) -> list[Verdict]:
    conf = dict(config or {})
    tracks = build_tracks(
        detections,
        frames,
        registration,
        radius=conf.get("track_radius_px", 8.0),
        min_frames=conf.get("min_track_frames", 10),
        lock_span_px=conf.get("lock_span_px", 1.0),
    )
    verdicts = [
        classify(
            t,
            v_det_max=conf.get("v_det_max_px", 8.0),
            v_sky_min=conf.get("v_sky_min_px", 30.0),
            lock_span_px=conf.get("lock_span_px", 1.0),
        )
        for t in tracks
    ]
    hits = [v for v in verdicts if v.is_target]
    hits.sort(key=lambda v: -v.track.length)
    return hits
