from __future__ import annotations

import inspect

import numpy as np
import pytest

from src.calib.hotpixel import (
    HotCluster,
    _pixel_stats,
    build_from_sequence,
    build_hotpixel_map,
    reject_hot,
)
from src.config import load_config
from src.dataio.fits_loader import FrameSequence

# 五个实测传感器缺陷簇中心 (cx, cy)，0-based，x=列 / y=行。
# 数据集 A（2026-03-09）与数据集 B（2026-07-22）相隔约 4.5 个月，两次独立观测复现
# 同一批位置——这是"传感器固有缺陷"而非"天体"的确证。
#
# 取值来自 flat_ratio=0.10 的实测。该阈值位于一段实测平台的中部：0.07 至 0.15 都给出
# 同样这 5 簇，0.05 只给 4 簇（漏掉 (312, 3206)），0.20 跳到 18 簇、0.30 跳到 242 簇。
# 因此 0.10 是量出来的，不是为了凑答案挑的，且两侧各有约 0.03/0.05 的裕度。
# 注意 (312, 3206) 实测 flat_ratio 0.0743，在原 brief 的 0.05 下会被漏掉，
# 但它的 median 是 134（背景 6），且在 4 个月前的数据集 A 同一像素复现，是真缺陷。
EXPECTED_CLUSTER_CENTERS = [
    (0.0, 0.0),
    (244.1, 3173.6),
    (312.0, 3206.5),
    (327.0, 3210.1),
    (2192.0, 3223.0),
]
CENTER_TOL_PX = 2.0


class _StubSequence:
    """最小 FrameSequence 替身：build_from_sequence 只用到 image(index)。

    同时记录读取次序，用来断言 max_frames 真的截断了帧列表。
    """

    def __init__(self, frames):
        self._frames = list(frames)
        self.reads: list[int] = []

    def image(self, index: int) -> np.ndarray:
        self.reads.append(index)
        return np.asarray(self._frames[index], dtype=np.float64)


def _poisson_frames(n: int, shape=(16, 16), mean: float = 6.0, seed: int = 11):
    """泊松底噪帧（整数、非负），用于走 uint16 转换路径的用例。

    这里不能用高斯噪声：N(6, 3.8) 会产生负像素，会被 build_from_sequence 的
    uint16 守卫正当地拒绝，从而掩盖用例真正想测的东西。
    """
    rng = np.random.default_rng(seed)
    return [rng.poisson(mean, size=shape).astype(np.float64) for _ in range(n)]


@pytest.fixture()
def synthetic_cube():
    """20 帧：3 个数值锁定的热像素簇 + 1 颗每帧移动的恒星 + 1 个位置固定但亮度起伏的源。

    最后那个源是必需的：移动亮源只在少数帧经过某像素，其时间轴 median 仍停在背景上，
    因此**光靠亮度判据就能排除它**——只有它在场时，删掉 flat_ratio 判据不会让任何
    用例失败（已实测：仅保留亮度判据时全部合成用例照过）。位置固定、亮度起伏的源
    （闪烁的恒星）median 高、极差大，是唯一只有 flat_ratio 能挡住的东西。
    """
    rng = np.random.default_rng(7)
    n, ny, nx = 20, 128, 128
    cube = rng.normal(6.0, 3.8, size=(n, ny, nx))
    hot = [(20, 30, 900.0), (21, 30, 850.0), (80, 90, 1200.0)]
    for x, y, value in hot:
        cube[:, y, x] = value
    for k in range(n):
        cube[k, 60, 5 + 4 * k] = 1500.0  # 每帧移动 4 px 的亮源
        cube[k, 100, 100] = 400.0 + 100.0 * (k % 9)  # 位置固定、亮度起伏的源
    return cube, hot


VARIABLE_SOURCE_XY = (100, 100)


def test_finds_locked_pixels_only(synthetic_cube):
    cube, hot = synthetic_cube
    hpm = build_hotpixel_map(cube, bkg_median=6.0, bkg_sigma=3.8, dilate=0)
    for x, y, _ in hot:
        assert hpm.mask[y, x]
    # 移动的亮源不得入选
    assert not hpm.mask[60, 5]
    assert not hpm.mask[60, 5 + 4 * 19]
    # 位置固定但亮度起伏的源同样不得入选：它每帧都远高于阈值，只有 flat_ratio 能挡住
    vx, vy = VARIABLE_SOURCE_XY
    assert not hpm.mask[vy, vx]
    # 除三个热像素外不得有任何多余入选像素——否则"只找锁定像素"这句话是空的
    assert hpm.mask.sum() == 3
    assert hpm.mask.dtype == np.bool_


def test_variable_source_is_bright_enough_to_need_flat_ratio(synthetic_cube):
    """前置条件检查：那个起伏源确实亮到能通过亮度判据。

    否则 test_finds_locked_pixels_only 里针对它的那条断言会因为"它本来就不够亮"
    而通过，flat_ratio 判据依旧无人看守。
    """
    cube, _ = synthetic_cube
    vx, vy = VARIABLE_SOURCE_XY
    series = cube[:, vy, vx]
    assert series.min() > 6.0 + 5.0 * 3.8, "起伏源必须每帧都超过亮度阈值"
    med = float(np.median(series))
    assert (series.max() - series.min()) / med > 0.10, "起伏源的极差比必须超过 flat_ratio"


def test_adjacent_pixels_merge_into_one_cluster(synthetic_cube):
    cube, _ = synthetic_cube
    hpm = build_hotpixel_map(cube, bkg_median=6.0, bkg_sigma=3.8, dilate=0)
    assert len(hpm.clusters) == 2
    npix = sorted(c.npix for c in hpm.clusters)
    assert npix == [1, 2]
    # 两像素簇的质心必须落在 (20, 30) 与 (21, 30) 之间，而不是任一端点
    merged = next(c for c in hpm.clusters if c.npix == 2)
    assert merged.cx == pytest.approx(20.5)
    assert merged.cy == pytest.approx(30.0)


def test_clusters_sorted_by_size_then_position(synthetic_cube):
    """簇按 npix 降序排列：报告与调试都依赖这个稳定次序。"""
    cube, _ = synthetic_cube
    hpm = build_hotpixel_map(cube, bkg_median=6.0, bkg_sigma=3.8, dilate=0)
    assert [c.npix for c in hpm.clusters] == [2, 1]


def test_dilation_grows_mask(synthetic_cube):
    cube, _ = synthetic_cube
    tight = build_hotpixel_map(cube, bkg_median=6.0, bkg_sigma=3.8, dilate=0)
    grown = build_hotpixel_map(cube, bkg_median=6.0, bkg_sigma=3.8, dilate=1)
    assert grown.mask.sum() > tight.mask.sum()
    # 膨胀只动 mask，不得改变簇的数量或中心
    assert [(c.cx, c.cy, c.npix) for c in grown.clusters] == [
        (c.cx, c.cy, c.npix) for c in tight.clusters
    ]
    # 4 连通膨胀 1 次：孤立像素 1->5，两像素簇 2->8，合计 13
    assert grown.mask.sum() == 13


def test_to_dict_exposes_all_cluster_fields(synthetic_cube):
    cube, _ = synthetic_cube
    hpm = build_hotpixel_map(cube, bkg_median=6.0, bkg_sigma=3.8, dilate=0)
    d = hpm.to_dict()
    assert d["n_pixels"] == 3
    assert d["n_clusters"] == 2
    assert set(d["clusters"][0]) == {"cx", "cy", "npix", "median_value", "flat_ratio"}
    merged = next(c for c in d["clusters"] if c["npix"] == 2)
    # 簇内两像素中位值 900 与 850，取中位数 875；数值锁定故 flat_ratio 为 0
    assert merged["median_value"] == pytest.approx(875.0)
    assert merged["flat_ratio"] == pytest.approx(0.0)


def test_constant_frames_produce_no_clusters():
    """整幅恒定在背景中位数上的立方体不得产出任何簇。

    这种立方体的 (max-min)/median 恒为 0，"数值锁定"一项永远满足，唯一挡住它的是
    median > bkg_median + n_sigma * bkg_sigma。该判据一旦被删掉或写反，1024 个
    像素会并成一个巨簇——这条断言就是为它设的。

    （原 brief 的 test_reject_hot_filters_by_radius 把这件事和 reject_hot 混在一起测，
    结果两头都没测到，见 test_reject_hot_radius_controls_which_points_survive 的说明。）
    """
    hpm = build_hotpixel_map(np.full((5, 32, 32), 6.0), bkg_median=6.0, bkg_sigma=3.8)
    assert hpm.clusters == []
    assert not hpm.mask.any()


def test_rejects_malformed_cube():
    with pytest.raises(ValueError):
        build_hotpixel_map(np.zeros((4, 4)), bkg_median=6.0, bkg_sigma=3.8)
    with pytest.raises(ValueError):
        build_hotpixel_map(np.zeros((0, 4, 4)), bkg_median=6.0, bkg_sigma=3.8)


def test_pixel_stats_chunking_is_exact(synthetic_cube):
    """分块计算逐像素 median / span 必须与整块计算逐位一致。

    分块存在的唯一理由是内存：np.median 会复制一份输入，30 帧 4096^2 的 uint16
    立方体本身就是 1 GB，整块走会再多 1 GB。分块若算错，代价是错误的热像素图，
    所以这里用 max_chunk_bytes=1（退化为逐行）与整块结果做逐位比对。
    """
    cube, _ = synthetic_cube
    ref_med, ref_span = _pixel_stats(cube)
    med, span = _pixel_stats(cube, max_chunk_bytes=1)
    assert np.array_equal(med, ref_med)
    assert np.array_equal(span, ref_span)
    # 同时钉住两个量的定义本身
    assert np.array_equal(ref_med, np.median(cube, axis=0))
    assert np.array_equal(ref_span, cube.max(axis=0) - cube.min(axis=0))


def test_reject_hot_radius_controls_which_points_survive(synthetic_cube):
    """半径必须真的参与判断：同一批点在 radius=4 与 radius=12 下结果不同。

    原 brief 的 test_reject_hot_filters_by_radius 传的是空簇列表（它构造的
    `np.full((5,32,32), 6.0) + np.eye(32)*0.0` 是一幅恒定图，`*0.0` 让 eye 项成为
    空操作），因此走的是 reject_hot 里 `if not clusters` 的早退分支，从未碰到半径
    比较，也和 test_reject_hot_with_no_clusters_keeps_all 完全重复；它还把测点放在
    (100, 100)——超出 32x32 图幅。故此处重写为真正带簇的用例。
    """
    cube, _ = synthetic_cube
    hpm = build_hotpixel_map(cube, bkg_median=6.0, bkg_sigma=3.8, dilate=0)
    assert hpm.clusters, "前置条件：必须真有簇，否则本用例退化为空测试"
    # 簇中心为 (80, 90) 与 (20.5, 30)；第一个点距 (80, 90) 恰 10 px
    xy = np.array([[80.0, 100.0], [80.0, 90.0], [1000.0, 1000.0]])
    tight = reject_hot(xy, hpm.clusters, radius=4.0)
    loose = reject_hot(xy, hpm.clusters, radius=12.0)
    assert tight.tolist() == [True, False, True]
    assert loose.tolist() == [False, False, True]


def test_reject_hot_removes_near_cluster():
    clusters = [HotCluster(cx=244.0, cy=3174.0, npix=7, median_value=900.0, flat_ratio=0.01)]
    xy = np.array([[244.0, 3174.0], [248.0, 3176.0], [300.0, 3174.0]])
    keep = reject_hot(xy, clusters, radius=8.0)
    assert keep.tolist() == [False, False, True]


def test_reject_hot_boundary_is_exclusive():
    """恰好等于 radius 的点算命中：钉住 `dist > radius` 而不是 `>=`。"""
    clusters = [HotCluster(cx=100.0, cy=100.0, npix=3, median_value=900.0, flat_ratio=0.01)]
    xy = np.array([[108.0, 100.0], [108.001, 100.0]])
    keep = reject_hot(xy, clusters, radius=8.0)
    assert keep.tolist() == [False, True]


def test_reject_hot_uses_nearest_of_many_clusters():
    """多簇时必须取最近距离：只比第一个簇会漏掉后面的簇。"""
    clusters = [
        HotCluster(cx=10.0, cy=10.0, npix=5, median_value=900.0, flat_ratio=0.01),
        HotCluster(cx=500.0, cy=600.0, npix=2, median_value=800.0, flat_ratio=0.02),
    ]
    xy = np.array([[10.0, 10.0], [501.0, 601.0], [300.0, 300.0]])
    keep = reject_hot(xy, clusters, radius=8.0)
    assert keep.tolist() == [False, False, True]


def test_reject_hot_with_no_clusters_keeps_all():
    xy = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert reject_hot(xy, [], radius=8.0).all()


def test_reject_hot_handles_empty_point_set():
    keep = reject_hot(np.empty((0, 2)), [], radius=8.0)
    assert keep.shape == (0,)
    assert keep.dtype == np.bool_


def test_build_from_sequence_finds_locked_pixel():
    frames = _poisson_frames(6)
    for f in frames:
        f[4, 5] = 900.0
    seq = _StubSequence(frames)
    hpm = build_from_sequence(seq, range(6), dilate=0)
    assert hpm.mask[4, 5]
    assert hpm.mask.sum() == 1
    assert [(c.cx, c.cy) for c in hpm.clusters] == [(5.0, 4.0)]


def test_max_frames_caps_frames_read():
    """max_frames 是内存闸门：必须真的截断读取，而不只是写在签名里。"""
    frames = _poisson_frames(10)
    for f in frames:
        f[4, 5] = 900.0
    seq = _StubSequence(frames)
    hpm = build_from_sequence(seq, range(10), max_frames=3, dilate=0)
    assert seq.reads == [0, 1, 2]
    assert hpm.mask[4, 5]


def test_empty_frame_indices_rejected():
    with pytest.raises(ValueError):
        build_from_sequence(_StubSequence([]), [])


def test_negative_pixel_rejected_instead_of_wrapping():
    """负像素必须显式报错，而不是静默绕成 65535 凭空造出一个热簇。

    build_from_sequence 按 uint16 累积立方体以控制内存（4096^2 float64 单帧 128 MB）。
    有符号→无符号转换会把 -1 绕成 65535；若该像素在多数帧都是负值，绕完之后
    median 极高、span 为 0，恰好同时满足两条热像素判据，得到一个完全虚构的传感器缺陷。
    实测（临时移除守卫）：把 (7, 9) 每帧置 -1.0 后得到 1 簇，中心 (9.0, 7.0)、
    median_value 65535.0、flat_ratio 0.0。
    实测数据集 A/B 共 55 帧无任何负像素，所以这条守卫当前不会误伤真数据。
    """
    frames = _poisson_frames(5)
    for f in frames:
        f[7, 9] = -1.0
    with pytest.raises(ValueError, match="负值"):
        build_from_sequence(_StubSequence(frames), range(5))


def test_negative_pixel_in_first_frame_rejected():
    """首帧是单独读取的（要拿它做背景统计），守卫不能只写在循环里。"""
    frames = _poisson_frames(5)
    frames[0][7, 9] = -1.0
    with pytest.raises(ValueError, match="负值"):
        build_from_sequence(_StubSequence(frames), range(5))


def test_non_finite_pixel_rejected():
    """NaN / inf 转 uint16 的结果是未定义的：`nan < 0` 为假，光靠负值检查兜不住。"""
    frames = _poisson_frames(5)
    frames[2][3, 3] = np.nan
    with pytest.raises(ValueError, match="非有限"):
        build_from_sequence(_StubSequence(frames), range(5))

    frames = _poisson_frames(5)
    frames[2][3, 3] = np.inf
    with pytest.raises(ValueError, match="非有限"):
        build_from_sequence(_StubSequence(frames), range(5))


def test_pixel_above_uint16_range_rejected():
    """超过 65535 的像素同样会绕回小值，必须一起挡住。"""
    frames = _poisson_frames(5)
    for f in frames:
        f[3, 3] = 70000.0
    with pytest.raises(ValueError, match="65535"):
        build_from_sequence(_StubSequence(frames), range(5))


def test_defaults_match_config():
    """函数默认值必须与 src/config/default.yaml 的 hotpixel 节一致。

    实数据用例不显式传阈值，所以默认值漂移会被它们抓到；但合成用例都显式传参，
    max_frames 与 reject_radius_px 更是没有任何用例会因漂移而失败，这条专门补位。
    """
    cfg = load_config()["hotpixel"]

    def defaults(fn):
        return {
            name: p.default
            for name, p in inspect.signature(fn).parameters.items()
            if p.default is not inspect.Parameter.empty
        }

    map_defaults = defaults(build_hotpixel_map)
    for key in ("n_sigma", "flat_ratio", "dilate"):
        assert map_defaults[key] == cfg[key], f"build_hotpixel_map 默认值与配置不一致: {key}"

    seq_defaults = defaults(build_from_sequence)
    for key in ("n_sigma", "flat_ratio", "dilate", "max_frames"):
        assert seq_defaults[key] == cfg[key], f"build_from_sequence 默认值与配置不一致: {key}"

    assert defaults(reject_hot)["radius"] == cfg["reject_radius_px"]


def _assert_expected_centers(hpm, label: str) -> None:
    found = sorted((c.cx, c.cy) for c in hpm.clusters)
    for want in EXPECTED_CLUSTER_CENTERS:
        assert any(
            abs(cx - want[0]) <= CENTER_TOL_PX and abs(cy - want[1]) <= CENTER_TOL_PX
            for cx, cy in found
        ), f"{label} 未找到已知热像素簇 {want}；实测簇中心 {found}"
    # 数量也要钉住：只查"包含"挡不住阈值放松导致的过检（0.20 会跳到 18 簇）
    assert len(found) == len(EXPECTED_CLUSTER_CENTERS), f"{label} 簇数不符：实测 {found}"


def _nearest(hpm, cx: float, cy: float) -> HotCluster:
    """按坐标取簇。刻意不用 round() 建字典：round(3206.5) 在 Python 里是 3206
    而非 3207（银行家舍入），簇中心恰好落在 .5 上时会静默取错簇。
    """
    best = min(hpm.clusters, key=lambda c: (c.cx - cx) ** 2 + (c.cy - cy) ** 2)
    assert abs(best.cx - cx) <= CENTER_TOL_PX and abs(best.cy - cy) <= CENTER_TOL_PX, (
        f"({cx}, {cy}) 附近无簇，最近的是 ({best.cx}, {best.cy})"
    )
    return best


def test_dataset_b_reproduces_known_clusters(dataset_b_dir):
    seq = FrameSequence.from_directory(dataset_b_dir)
    hpm = build_from_sequence(seq, list(range(16, 46)))
    _assert_expected_centers(hpm, "数据集 B")

    # (312, 3206.5) 是把 flat_ratio 从 0.05 提到 0.10 的直接原因：它实测 0.0743，
    # 在 0.05 下会被漏掉。这条断言让"为什么是 0.10"在阈值被改回时立刻显形。
    assert 0.05 < _nearest(hpm, 312.0, 3206.5).flat_ratio < 0.10
    # 两个大簇必须成团，不能碎成单像素（在 flat_ratio=0.10 下实测 9 与 8 像素；
    # 簇边缘有 ratio 恰为 0.1000 和 0.1085 的像素，阈值稍动 npix 就变 10/9，故只卡下界）
    assert _nearest(hpm, 327.0, 3210.1).npix >= 7
    assert _nearest(hpm, 244.1, 3173.6).npix >= 7
    # (0, 0) 是卡死的角点像素，正是 background_stats 在 frame 30 报出 max=26978 的来源
    assert _nearest(hpm, 0.0, 0.0).median_value > 20000.0
    # (2192, 3223) 单像素、中位值约 2612，远高于背景 6
    assert _nearest(hpm, 2192.0, 3223.0).median_value > 2000.0


def test_dataset_a_reproduces_same_clusters(dataset_a_dir):
    """相隔 4 个月的独立观测应给出同一批缺陷位置——这是传感器缺陷的确证。"""
    seq = FrameSequence.from_directory(dataset_a_dir)
    hpm = build_from_sequence(seq, list(range(5, 36)))
    _assert_expected_centers(hpm, "数据集 A")


def test_reject_hot_removes_real_detections_at_cluster_sites(dataset_b_dir):
    """闭环：实数据簇 + 落在簇上的伪探测点，剔除后只剩真点。

    这是本任务对下游的全部承诺——热像素图必须能真的把伪探测拿掉。
    """
    seq = FrameSequence.from_directory(dataset_b_dir)
    hpm = build_from_sequence(seq, list(range(16, 26)))
    assert hpm.clusters
    fake = np.array([[c.cx, c.cy] for c in hpm.clusters])
    real = np.array([[2047.5, 1024.0], [800.0, 800.0]])
    xy = np.vstack([fake, real])
    keep = reject_hot(xy, hpm.clusters)
    assert not keep[: len(fake)].any()
    assert keep[len(fake) :].all()
