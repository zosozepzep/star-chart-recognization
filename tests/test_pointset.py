# -*- coding: utf-8 -*-
"""点集入参校验器 ``as_xy`` 的测试。

被保护的性质：**列数错的表格不得被静默摊平成另一组点**。原先十个模块各写
``np.asarray(xy, dtype=np.float64).reshape(-1, 2)``，而 ``reshape`` 接受任何
元素数为偶数的形状——这是一条完全无声的通路，结果不报错、量级正常、下游一路
算到底。所以本文件每一条拒绝断言都同时记下**旧写法的返回值**，否则读者无法
判断守卫拦的是不是一个真实存在的危害（R381：「输入危险」与「守卫拦得住」
是两条各自要验的断言）。

判据是**逐形状**的而不是 ``ndim == 2``：扁平的 ``(2N,)`` 在本项目里合法且有
现成调用方，一律要求二维会是签名语义变更。
"""
from __future__ import annotations

import numpy as np
import pytest

from src.pointset import as_xy


# --------------------------------------------------------------------------
# 接受的四类形状
# --------------------------------------------------------------------------


def test_standard_n_by_2_passes_through_unchanged():
    """``(N, 2)`` 是唯一的标准形，原样接受且不复制成别的形状。"""
    arr = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    out = as_xy(arr)
    assert out.shape == (3, 2)
    assert out.dtype == np.float64
    assert out.tolist() == [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]


def test_flat_2n_is_read_as_alternating_xy():
    """扁平 ``(2N,)`` 按 ``[x0, y0, x1, y1, ...]`` 解读——这是本项目既有的合法用法。

    这一条是「不能一律要求 ``ndim == 2``」的锚：把判据写成 ``ndim == 2`` 会让
    现成调用方全部转红，那是签名语义变更而不是修洞。
    """
    out = as_xy([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert out.shape == (3, 2)
    assert out.tolist() == [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]


@pytest.mark.parametrize(
    "empty", [np.zeros(0), np.zeros((0, 2)), [], np.zeros((0, 5))]
)
def test_empty_inputs_normalise_to_0_by_2(empty):
    """空输入一律归一到 ``(0, 2)``，调用方的 ``len()``/切片/KDTree 都按第 0 维工作。

    ``(0, 5)`` 也在此列且**不抛错**：它没有任何元素，所以「第二维必须是 2」这条
    判据没有可被误读的坐标可言，归一成 ``(0, 2)`` 比抛错更有用。
    """
    out = as_xy(empty)
    assert out.shape == (0, 2)
    assert out.dtype == np.float64


def test_integer_input_is_converted_to_float64():
    """整型输入必须转成 ``float64``——全局约束要求点集一律 ``float64``。

    不转型的后果是下游的亚像素质心运算被整除或提前截断。
    """
    out = as_xy(np.array([[1, 2], [3, 4]], dtype=np.int64))
    assert out.dtype == np.float64


def test_two_by_two_is_unambiguous_under_both_rules():
    """``(2, 2)`` 在「二维」与「扁平」两条规则下解读相同，不存在歧义。

    ``(1, 2)`` 是一个点而不是两个：这是另一处容易读错的地方，一并钉住。
    """
    assert as_xy(np.array([[1.0, 2.0], [3.0, 4.0]])).tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert as_xy(np.array([[1.0, 2.0]])).shape == (1, 2)


# --------------------------------------------------------------------------
# 拒绝的三类形状——每条都记下旧 reshape 写法的返回值
# --------------------------------------------------------------------------


def test_multi_column_table_is_not_silently_flattened_into_more_points():
    """``(N, 4)`` 必须抛错，**不得**被摊平成 ``2N`` 个点。

    这是本模块存在的全部理由。误把 ``x, y, flux, snr`` 整表传进来时，实测旧写法
    ``np.asarray(arr).reshape(-1, 2)`` 把 ``(2, 4)`` 变成 ``(4, 2)``：

        [[1, 2, 3, 4],        ->   [[1, 2],
         [5, 6, 7, 8]]              [3, 4],     <- 把 flux/snr 当成了坐标
                                    [5, 6],
                                    [7, 8]]

    点数从 2 变成 4，且第二个点 ``(3, 4)`` 是两个非坐标量。**不报错、量级正常、
    一路算到底**——所以只有入口守卫拦得住。
    """
    bad = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    # 先证危害存在：旧写法确实静默把 2 个点变成 4 个。
    old = np.asarray(bad, dtype=np.float64).reshape(-1, 2)
    assert old.shape == (4, 2)
    assert old.tolist() == [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
    # 再证守卫拦得住。
    with pytest.raises(ValueError, match="第二维必须是 2"):
        as_xy(bad)


def test_three_column_table_is_rejected_too():
    """``(N, 3)`` 同理：实测旧写法把 ``(2, 3)`` 变成 **3** 个点。

    ``(N, 3)`` 比 ``(N, 4)`` 更隐蔽，因为 3 是奇数、点数变化的比例不是整倍，
    人眼更难从下游的计数上察觉。
    """
    bad = np.arange(6.0).reshape(2, 3)
    assert np.asarray(bad).reshape(-1, 2).shape == (3, 2)
    with pytest.raises(ValueError, match="第二维必须是 2"):
        as_xy(bad)


def test_odd_length_flat_input_is_rejected():
    """奇数长度的一维输入抛 ``ValueError``。

    这一档旧写法**本来就会抛** ``ValueError``（``reshape`` 无法均分），所以守卫
    在「是否产出数值」上是等价的；保留它的价值只在消息——带形状与实参名的中文
    消息比 ``cannot reshape array of size 5 into shape (2)`` 有诊断价值。
    """
    with pytest.raises(ValueError, match="元素数必须是偶数"):
        as_xy(np.arange(5.0))


def test_three_dimensional_input_is_rejected():
    """三维及以上抛错——实测旧写法把 ``(2, 2, 2)`` 摊成 4 个点。"""
    bad = np.arange(8.0).reshape(2, 2, 2)
    assert np.asarray(bad).reshape(-1, 2).shape == (4, 2)
    with pytest.raises(ValueError, match=r"必须是 \(N, 2\)"):
        as_xy(bad)


def test_error_message_names_the_offending_argument():
    """``name`` 必须进消息：十个调用点共用一条消息时，指名实参才有用。

    「点集形状非法」远不如「dst_xy 形状非法」有诊断价值；这条钉住那个形参没被
    忽略掉（默认值也一并验，否则漏传时消息会是空的）。
    """
    with pytest.raises(ValueError, match="dst_xy 形状非法"):
        as_xy(np.arange(8.0).reshape(2, 4), name="dst_xy")
    with pytest.raises(ValueError, match="点集 形状非法"):
        as_xy(np.arange(8.0).reshape(2, 4))


# --------------------------------------------------------------------------
# 十个调用点都真的过了这个校验器
# --------------------------------------------------------------------------


def test_all_pointset_entry_points_reject_a_four_column_table():
    """五个模块的入口都必须拒绝 ``(N, 4)``——否则某处漏接了校验器。

    逐个点名而不是只测 ``as_xy`` 本身：``as_xy`` 绿而某个模块仍留着
    ``reshape(-1, 2)``，那个模块的洞就还开着。这条是「十处都换过了」的守卫。
    """
    from src.register.solver import vote_correspondences
    from src.register.transform import apply_transform, fit_similarity, similarity_matrix
    from src.target.trajectory import angular_rates
    from src.validate.repeatability import cross_frame_repeatability

    bad = np.arange(8.0).reshape(2, 4)
    good = np.array([[0.0, 0.0], [1.0, 1.0]])
    identity = similarity_matrix()

    with pytest.raises(ValueError, match="形状非法"):
        apply_transform(identity, bad)
    with pytest.raises(ValueError, match="形状非法"):
        fit_similarity(bad, good)
    with pytest.raises(ValueError, match="形状非法"):
        vote_correspondences(bad, good)
    with pytest.raises(ValueError, match="形状非法"):
        angular_rates(bad, [0.0, 1.0], 6.179047)
    with pytest.raises(ValueError, match="形状非法"):
        cross_frame_repeatability([bad, bad], match_radius=3.0)


def test_pointset_module_does_not_import_truth():
    """真值隔离（硬约束）：``src/pointset.py`` 被 ``src/target/`` import，不得引真值。

    这个校验器是新增的共用模块，恰好被真值隔离约束覆盖的三个包全部依赖，所以它
    自己必须干净——否则 ``src/target/`` 会经由它拿到通往 ``.DAT`` 的路径。
    """
    import pathlib

    import src.pointset as mod

    text = pathlib.Path(mod.__file__).read_text(encoding="utf-8")
    assert "truth" not in text
