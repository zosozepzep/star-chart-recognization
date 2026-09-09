from __future__ import annotations

import inspect

import numpy as np
import pytest
from photutils.utils.exceptions import NoDetectionsWarning

from src.calib.background import model_background
from src.calib.hotpixel import HotCluster
from src.dataio.fits_loader import FrameSequence
from src.detect.segmentation import SourceTable, detect_sources_in_frame, detect_sequence


def make_frame(sources, ny=256, nx=256, seed=11, sigma_psf=1.6):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:ny, 0:nx]
    img = rng.normal(6.45, 3.80, size=(ny, nx))
    for cx, cy, amp in sources:
        img += amp * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma_psf ** 2))
    return img


def test_recovers_planted_sources():
    truth = [(50.0, 60.0, 800.0), (150.0, 40.0, 400.0), (200.0, 190.0, 250.0)]
    img = make_frame(truth)
    model = model_background(img, box=64, backend="cpu")
    tbl = detect_sources_in_frame(img, model, n_sigma=5.0, npixels=5, frame=3)
    assert len(tbl) == 3
    assert tbl.frame == 3
    order = np.argsort(tbl.x)
    got = np.column_stack([tbl.x[order], tbl.y[order]])
    want = np.array([[50.0, 60.0], [150.0, 40.0], [200.0, 190.0]])
    # 实测最大偏差 0.0343 px，约 15 倍余量
    assert np.abs(got - want).max() < 0.5
    # 实测流量 12621.9 / 6063.8 / 3761.4、面积 64 / 45 / 41
    assert tbl.npix[order].tolist() == [64, 45, 41]


def test_flux_ordering_matches_amplitude_ordering():
    """AMENDMENT (裁决 27c)：同时走 n >= len 分支并钉死别名隔离。"""
    img = make_frame([(50.0, 60.0, 800.0), (150.0, 40.0, 200.0)])
    model = model_background(img, box=64, backend="cpu")
    tbl = detect_sources_in_frame(img, model, n_sigma=5.0, npixels=5)
    bright = tbl.brightest(1)
    assert len(bright) == 1
    assert bright.x[0] == pytest.approx(50.0, abs=0.5)
    allof = tbl.brightest(5)
    assert len(allof) == 2
    assert allof is not tbl
    allof.x[0] = -999.0
    assert tbl.x[0] != -999.0


def test_lower_threshold_finds_more():
    """AMENDMENT (裁决 23)——原任务书用 amp=26，**实测会失败**。

    暗源 (150,40) 对 800-amp 源的完整扫描，box=64、npixels=5，走正确的残差式阈值：
        amp  8, 12         -> n5=1 n4=1 n3=1   （任何阈值都太暗）
        amp 14,15,16,17    -> n5=1 n4=1 n3=2   <- 3σ 跨界带，本测试用它
        amp 18,20,22       -> n5=1 n4=2 n3=2   <- 4σ 跨界带
        amp 24,26,30       -> n5=2 n4=2 n3=2   （5σ 就已找到）
    原任务书的 amp=26 给出 n5=2、n3=2，于是 `n3 > n5` 断言的是 2 > 2。
    amp=16 落在 3σ 带中间：n5=1、n4=1、n3=2。这个种子上噪声本身到 3σ 都不贡献
    额外探测，所以计数差**就是**那颗植入的源。这张表要留在 docstring 里：
    它是唯一能阻止后来的改动把振幅漂出带外而无人察觉的东西。
    """
    img = make_frame([(50.0, 60.0, 800.0), (150.0, 40.0, 16.0)])
    model = model_background(img, box=64, backend="cpu")
    n5 = len(detect_sources_in_frame(img, model, n_sigma=5.0, npixels=5))
    n4 = len(detect_sources_in_frame(img, model, n_sigma=4.0, npixels=5))
    n3 = len(detect_sources_in_frame(img, model, n_sigma=3.0, npixels=5))
    assert n5 == 1
    assert n4 == 1          # AMENDMENT (R154)：4σ 也钉死，否则 amp 漂到 18 无人察觉
    assert n3 == 2
    assert n3 > n5


def test_hot_clusters_are_rejected():
    """AMENDMENT (裁决 24)——热簇挪离对角线，到 (100, 140)。

    实测（正确的残差式阈值路径）：探测到 (50.01, 59.99) 与 (100.02, 139.99)；
    传入正确顺序的 (x, y) 保留 **1** 个，传入交换后的 (y, x) 保留 **2** 个。
    `center_of_mass` 返回 (row, col) = (y, x)，所以 hotpixel→detect 的每一处
    边界都是交换陷阱，而本测试是唯一的守卫。**这个不对称是故意的，
    不要把它"整理"回对称坐标。**
    """
    img = make_frame([(50.0, 60.0, 800.0), (100.0, 140.0, 900.0)])
    model = model_background(img, box=64, backend="cpu")
    hot = [HotCluster(cx=100.0, cy=140.0, npix=3, median_value=900.0, flat_ratio=0.01)]
    tbl = detect_sources_in_frame(
        img, model, n_sigma=5.0, npixels=5, hot_clusters=hot, reject_radius=8.0
    )
    assert len(tbl) == 1
    assert tbl.x[0] == pytest.approx(50.0, abs=0.5)
    assert tbl.y[0] == pytest.approx(60.0, abs=0.5)


def test_columns_are_plain_arrays_not_astropy_columns():
    """R155：原任务书用 `isinstance(arr, np.ndarray)`，对未转换的列**是瞎的**。

    实测 photutils 2.0.2 交回来的六列：
        xcentroid/ycentroid/segment_flux/max_value -> Column,   unit=None
        elongation                                 -> Quantity, unit=dimensionless
        area                                       -> Quantity, unit=pix2, dtype=float64
    六者 `isinstance(..., np.ndarray)` **全为 True**（Column/Quantity 都是 ndarray
    子类），前五列 `dtype == float64` 也为真。所以原断言在"不做转换"的变异体上
    五项全过，只有 `npix.dtype == int64` 偶然抓到它（area 是 float64）。
    Task 6a 的修复轮已经为同一个断言形式付过一次代价。

    能真正判别的是 `type(...) is np.ndarray`。注意 `hasattr(arr, "unit")` 单独用
    **不够**：无单位的 Column 也有 `unit` 属性（值为 None），实测 hasattr 为 True。
    两个检查都要，且互不替代。
    """
    img = make_frame([(50.0, 60.0, 800.0)])
    model = model_background(img, box=64, backend="cpu")
    tbl = detect_sources_in_frame(img, model)
    for name in ("x", "y", "flux", "peak", "elongation", "npix"):
        arr = getattr(tbl, name)
        # `type(...) is np.ndarray` 是刻意的：Column 与 Quantity 都是 ndarray 子类，
        # isinstance 在未转换的列上会通过。
        assert type(arr) is np.ndarray, f"{name} 仍是 astropy 列子类"
        assert not hasattr(arr, "unit"), f"{name} 泄漏了单位"
    for arr in (tbl.x, tbl.y, tbl.flux, tbl.peak, tbl.elongation):
        assert arr.dtype == np.float64
    assert tbl.npix.dtype == np.int64
    assert tbl.xy.shape == (len(tbl), 2)
    assert type(tbl.xy) is np.ndarray and not hasattr(tbl.xy, "unit")
    from scipy.spatial import cKDTree

    cKDTree(tbl.xy).query(np.array([[50.0, 60.0]]))


def test_naive_column_merge_really_does_raise():
    """R156：这道守卫存在的理由——但顺序敏感，实测如下。

    `np.column_stack` 的报错取决于**参数顺序**，实测（同一张 2 行表）：
        list(table.itercols())         [xcentroid 先] -> 不报错，返回 Quantity(2,6)
        list(table.itercols())[::-1]   [area 先]      -> UnitConversionError
        [area, elongation]                            -> UnitConversionError
        [xcentroid, area]                             -> 不报错
        [area, xcentroid]                             -> UnitConversionError
        list(table)                    [迭代**行**]    -> 不报错，shape (1, 2)
    原任务书让实现者断言 `np.column_stack(list(table))` 会抛异常——`list(table)`
    迭代的是**行不是列**，而且它不抛。所以本测试固定用会抛的那一对，
    并且**如果哪天 photutils 不再附带单位，它必须响亮地失败**，因为那意味着
    这道守卫过时了、该重新评估，而不是继续静默留着。
    """
    from astropy.units import UnitConversionError
    from photutils.segmentation import SourceCatalog, detect_sources

    from src.detect.segmentation import _COLUMNS

    img = make_frame([(50.0, 60.0, 800.0), (150.0, 40.0, 400.0)])
    model = model_background(img, box=64, backend="cpu")
    seg = detect_sources(model.subtract(img), model.threshold(5.0), npixels=5)
    raw = SourceCatalog(model.subtract(img), seg).to_table(list(_COLUMNS))
    # area 带 pix2、xcentroid 无单位：这一对合并必抛
    with pytest.raises(UnitConversionError):
        np.column_stack([raw["area"], raw["xcentroid"]])


def test_empty_frame_yields_empty_table():
    """AMENDMENT (裁决 27b)：收紧，并把 None 分支钉成原因。

    实测：seed=3、128²、box=64 的纯噪声图上 n_sigma=20 时 detect_sources 返回 None
    并发 NoDetectionsWarning。原任务书只查 len==0 与 xy.shape，那样即使
    SourceTable.empty 返回了错误 dtype 或不齐的列也会通过。
    """
    rng = np.random.default_rng(3)
    img = rng.normal(6.45, 3.80, size=(128, 128))
    model = model_background(img, box=64, backend="cpu")
    with pytest.warns(NoDetectionsWarning):
        tbl = detect_sources_in_frame(img, model, n_sigma=20.0, npixels=5)
    assert len(tbl) == 0
    assert tbl.xy.shape == (0, 2)
    for arr in (tbl.x, tbl.y, tbl.flux, tbl.peak, tbl.elongation):
        assert arr.shape == (0,) and arr.dtype == np.float64
    assert tbl.npix.shape == (0,) and tbl.npix.dtype == np.int64
    assert tbl.to_records() == []
    assert len(tbl.select(np.zeros(0, dtype=bool))) == 0
    assert len(tbl.brightest(3)) == 0


def test_select_and_records_roundtrip():
    """经真实探测路径过一遍 select/to_records。

    注：select 的布尔/整数双契约与错长布尔掩码的中文 ValueError 已由 6a 的
    tests/detect/test_source_table.py 单元覆盖，此处不重复。
    """
    img = make_frame([(50.0, 60.0, 800.0), (150.0, 40.0, 400.0)])
    model = model_background(img, box=64, backend="cpu")
    tbl = detect_sources_in_frame(img, model, frame=7)
    sub = tbl.select(tbl.x < 100.0)
    assert len(sub) == 1 and sub.frame == 7
    recs = tbl.to_records()
    assert len(recs) == 2
    assert set(recs[0]) == {"frame", "x", "y", "flux", "peak", "elongation", "npix"}


def test_threshold_is_relative_to_the_residual_not_the_raw_image():
    """R153：把 threshold() 当绝对阈值直接和原始图比，实测 3σ 会给出 52488 个源
    （正确值 169）。两种正确写法必须给出同一组数——实测 95/116/169 逐个相等。
    这条测试用合成图钉住量级，真实数据的等号由 dataset B 那条测试钉。
    """
    img = make_frame([(50.0, 60.0, 800.0), (150.0, 40.0, 400.0), (200.0, 190.0, 250.0)])
    model = model_background(img, box=64, backend="cpu")
    n = len(detect_sources_in_frame(img, model, n_sigma=3.0, npixels=5))
    # 纯噪声不该在 3σ 上炸出成百上千个源；实测这张图 3σ 给 3 个
    assert n == 3


def test_dataset_b_frame30_threshold_counts(dataset_b_dir):
    """复现 spec 记录的阈值曲线。

    控制器实测（photutils 2.0.2，默认 box=128、npixels=5，残差式与绝对式两条路径
    逐个相等）：5σ→95、4σ→116、3σ→169，与 spec 记录完全一致。原任务书给的是
    ±15% 容差；实测既然是整数级复现，就钉死等号，宽容差会让 95 漂到 109 都算通过。
    """
    seq = FrameSequence.from_directory(dataset_b_dir)
    img = seq.image(30)
    model = model_background(img, backend="cpu")
    n5 = len(detect_sources_in_frame(img, model, n_sigma=5.0, npixels=5))
    n4 = len(detect_sources_in_frame(img, model, n_sigma=4.0, npixels=5))
    n3 = len(detect_sources_in_frame(img, model, n_sigma=3.0, npixels=5))
    assert n5 == 95
    assert n4 == 116
    assert n3 == 169        # AMENDMENT (R154)：3σ 也钉死，spec 记录里有这个数
    assert n4 > n5


def test_dataset_b_exposure_changes_within_the_sequence(dataset_b_dir):
    """R157：逐帧背景建模的物理理由必须被测试钉住，而不只是写在 docstring 里。

    实测 dataset B 的 80 帧只有两个曝光值：f00-f07 为 80.0 ms，f08-f79 为 30.0 ms。
    取值用 `seq.headers[i].exposure_ms`——FrameSequence 没有 header() 方法，
    headers 是 FrameHeader dataclass 的列表，不支持 `in` 也不支持 .get()。
    这条测试若失败，说明数据集换了，逐帧建模的理由需要重新论证。
    """
    seq = FrameSequence.from_directory(dataset_b_dir)
    exposures = [h.exposure_ms for h in seq.headers]
    assert len(exposures) == 80
    assert sorted(set(exposures)) == [30.0, 80.0]
    assert exposures[7] == 80.0
    assert exposures[8] == 30.0


def test_detect_sequence_keys_and_frame_numbers(dataset_b_dir):
    """R158：原任务书一条 detect_sequence 测试都没有。**只用 3 帧**——每帧约 2 s。

    键必须**恰好**是请求的帧集合（不是超集也不是子集），每张表的 .frame 必须
    等于它自己的键（mutation 7 会把 frame=f 写成 frame=-1），每张表非空。
    """
    seq = FrameSequence.from_directory(dataset_b_dir)
    frames = [10, 20, 30]
    out = detect_sequence(seq, frames, n_sigma=5.0, npixels=5)
    assert set(out) == set(frames)
    for f in frames:
        assert out[f].frame == f
        assert len(out[f]) > 0
        assert type(out[f].xy) is np.ndarray


def test_signature_defaults_match_config(cfg):
    """本任务书 Step 5 变异 5a 的补漏（实测发现的空转）。

    变异 5a（把签名默认 `npixels: int = 5` 改成 1）**在任务书原定的 12 条测试下
    整套 136 项全绿**：唯一钉 npixels 的 `test_dataset_b_frame30_threshold_counts`
    每次调用都显式传 `npixels=5`，于是签名默认从不被求值；而合成图对 npixels
    完全不敏感（同一张 3 源图实测 npixels=1..5 一律给 3 个源，所以合成用例也钉不住）。
    任务书变异 5a 引用的实测数字 1066 只在 npixels 被**忽略**时出现——那是变异 5b，
    实测确实杀掉了，但它与 5a 不是同一处改动。

    因此这里直接对签名默认取值，并与 src/config/default.yaml 的 detect 节比对：
    任务书 Step 3 要求「默认值必须与 detect 段一致」，此前没有任何测试守卫它。
    """
    detect_cfg = cfg["detect"]
    assert detect_cfg["npixels"] == 5
    assert detect_cfg["report_n_sigma"] == 5.0
    reject_radius = cfg["hotpixel"]["reject_radius_px"]
    assert reject_radius == 8.0
    for fn in (detect_sources_in_frame, detect_sequence):
        params = inspect.signature(fn).parameters
        assert params["n_sigma"].default == detect_cfg["report_n_sigma"]
        assert params["npixels"].default == detect_cfg["npixels"]
        assert params["reject_radius"].default == reject_radius
