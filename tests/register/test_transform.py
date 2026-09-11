from __future__ import annotations

import numpy as np
import pytest

from src.register.transform import (
    apply_transform,
    decompose,
    fit_similarity,
    invert,
    residuals,
    rms,
    similarity_matrix,
)

CENTER = (2047.5, 2047.5)


def test_identity_is_noop():
    M = similarity_matrix()
    xy = np.array([[10.0, 20.0], [3000.0, 4000.0]])
    assert np.allclose(apply_transform(M, xy), xy)


def test_pure_translation():
    M = similarity_matrix(tx=100.0, ty=-50.0)
    got = apply_transform(M, np.array([[0.0, 0.0]]))
    assert got[0] == pytest.approx([100.0, -50.0])


def test_rotation_about_center_leaves_center_fixed():
    M = similarity_matrix(rotation_deg=30.0, center=CENTER)
    got = apply_transform(M, np.array([list(CENTER)]))
    assert got[0] == pytest.approx(list(CENTER), abs=1e-9)


def test_rotation_90_degrees():
    M = similarity_matrix(rotation_deg=90.0, center=(0.0, 0.0))
    got = apply_transform(M, np.array([[1.0, 0.0]]))
    assert got[0] == pytest.approx([0.0, 1.0], abs=1e-9)


def test_scale_about_center():
    M = similarity_matrix(scale=2.0, center=(0.0, 0.0))
    got = apply_transform(M, np.array([[3.0, 4.0]]))
    assert got[0] == pytest.approx([6.0, 8.0])


def test_decompose_roundtrip():
    """scale/rotation_deg 原样往返，但 tx/ty 不与输入相等（Ruling 248）。

    `similarity_matrix` 把绕 center 的旋转折叠进平移列；`decompose` 原样读出该列，
    因此只有 center=(0, 0) 时 tx/ty 才等于输入实参。这里 center=CENTER 且带旋转，
    读出的 tx 与输入 6495 相差 -111.457693 px（= 6495 - 6383.5423066），是绕
    CENTER 旋转 -3.325° 折进平移列的量，不是拟合误差。
    """
    M = similarity_matrix(scale=0.99822, rotation_deg=-3.325, tx=6495.0, ty=-540.0, center=CENTER)
    d = decompose(M)
    assert d["scale"] == pytest.approx(0.99822, rel=1e-9)
    assert d["rotation_deg"] == pytest.approx(-3.325, abs=1e-9)
    assert d["tx"] == pytest.approx(6383.5423066, abs=1e-4)
    assert d["ty"] == pytest.approx(-414.3719726, abs=1e-4)


def test_decompose_roundtrip_exact_at_origin_center():
    """center=(0, 0) 时 tx/ty 与输入实参逐位相等（Ruling 248 第 2 点）。"""
    M = similarity_matrix(scale=1.02, rotation_deg=7.0, tx=-126.0, ty=18.0, center=(0.0, 0.0))
    d = decompose(M)
    assert d["tx"] == -126.0
    assert d["ty"] == 18.0


def test_decompose_similarity_matrix_reconstruction_identity():
    """similarity_matrix(**decompose(M), center=(0, 0)) 重建 M（Ruling 248 第 3 点）。

    这不是逐位相等的性质：200 个随机变换里只有 145/200 是 bit-exact，最差误差
    6.661e-16。用 atol=1e-12 而非 == ，四个数量级的余量覆盖两个代表性变换，同时仍能
    被 mutation 4（tx 读成 matrix[0,2] - center[0]）以 ~1e14 倍的误差杀死。
    """
    M_b = similarity_matrix(scale=0.99822, rotation_deg=-3.325, tx=6495.0, ty=-540.0, center=CENTER)
    d_b = decompose(M_b)
    rebuilt_b = similarity_matrix(**d_b, center=(0.0, 0.0))
    assert np.allclose(rebuilt_b, M_b, atol=1e-12)

    M_c = similarity_matrix(scale=2.0, rotation_deg=-7.0, tx=30.0, ty=-9.0, center=CENTER)
    d_c = decompose(M_c)
    rebuilt_c = similarity_matrix(**d_c, center=(0.0, 0.0))
    assert np.allclose(rebuilt_c, M_c, atol=1e-12)


def test_decompose_rotation_range_is_closed():
    """decompose 的 rotation_deg 落在闭区间 [-180, 180]（Ruling 252）。

    `arctan2(sin(180°), cos(180°))` 落在 +π 而不是 -π（sin(π) 是 +1.22e-16 不是
    -0.0），所以两端都可达；半开区间 `-180 <= r < 180` 在输入 180.0 时会失败。
    """
    d180 = decompose(similarity_matrix(rotation_deg=180.0, center=(0.0, 0.0)))
    assert d180["rotation_deg"] == pytest.approx(180.0, abs=1e-9)

    d_neg180 = decompose(similarity_matrix(rotation_deg=-180.0, center=(0.0, 0.0)))
    assert d_neg180["rotation_deg"] == pytest.approx(-180.0, abs=1e-9)

    d181 = decompose(similarity_matrix(rotation_deg=181.0, center=(0.0, 0.0)))
    assert d181["rotation_deg"] == pytest.approx(-179.0, abs=1e-9)

    d359 = decompose(similarity_matrix(rotation_deg=359.0, center=(0.0, 0.0)))
    assert d359["rotation_deg"] == pytest.approx(-1.0, abs=1e-9)

    for r in (d180["rotation_deg"], d_neg180["rotation_deg"], d181["rotation_deg"], d359["rotation_deg"]):
        assert -180.0 <= r <= 180.0


def test_fit_similarity_recovers_known_transform():
    rng = np.random.default_rng(1)
    src = rng.uniform(0.0, 4096.0, size=(60, 2))
    M = similarity_matrix(scale=1.0012, rotation_deg=0.35, tx=-126.0, ty=18.0, center=CENTER)
    dst = apply_transform(M, src)
    fitted = fit_similarity(src, dst)
    assert np.allclose(fitted, M, atol=1e-6)
    assert rms(fitted, src, dst) < 1e-8


def test_fit_similarity_is_robust_to_small_noise():
    rng = np.random.default_rng(2)
    src = rng.uniform(0.0, 4096.0, size=(80, 2))
    M = similarity_matrix(rotation_deg=0.2, tx=-126.0, ty=5.0, center=CENTER)
    dst = apply_transform(M, src) + rng.normal(0.0, 0.3, size=src.shape)
    fitted = fit_similarity(src, dst)
    assert decompose(fitted)["rotation_deg"] == pytest.approx(0.2, abs=0.02)
    assert rms(fitted, src, dst) < 0.6


def test_fit_similarity_requires_two_points():
    with pytest.raises(ValueError):
        fit_similarity(np.array([[1.0, 2.0]]), np.array([[3.0, 4.0]]))


def test_fit_similarity_with_exactly_two_points_succeeds():
    """n=2 时 fit_similarity 成功恢复变换（Ruling 251）。

    rms 的量级随点的坐标大小与条件数变化三个数量级（1.4e-14 ~ 6e-11），不是函数的
    固有属性而是这一对点的属性。这里选用的 (10,20)-(3000,2500) 实测 rms 1.39e-11，
    bound 取 1e-6 留了约五个数量级的余量，故意松，避免把边界钉在某一对点的偶然值上；
    mutation 7（n<2 改 n<3）会让这个测试直接因 raise 失败，所以松的 bound 不会白费。
    """
    src = np.array([[10.0, 20.0], [3000.0, 2500.0]])
    M = similarity_matrix(scale=1.0012, rotation_deg=0.35, tx=-126.0, ty=18.0, center=CENTER)
    dst = apply_transform(M, src)
    fitted = fit_similarity(src, dst)
    assert np.allclose(fitted, M, atol=1e-6)
    assert rms(fitted, src, dst) < 1e-6


def test_fit_similarity_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        fit_similarity(np.zeros((5, 2)), np.zeros((4, 2)))


def test_fit_similarity_error_messages_are_chinese():
    """两处 ValueError 消息均为中文（全局约束）。"""
    with pytest.raises(ValueError, match="点数不一致: 5 vs 4"):
        fit_similarity(np.zeros((5, 2)), np.zeros((4, 2)))
    with pytest.raises(ValueError, match="相似变换拟合至少需要 2 个对应点"):
        fit_similarity(np.array([[1.0, 2.0]]), np.array([[3.0, 4.0]]))


def test_fit_similarity_on_coincident_points_does_not_raise_but_is_arbitrary():
    """重合点不报错，返回一个自称完美拟合、实际任意的变换（Ruling 39）。

    src/dst 都退化成单点，lstsq 用 rcond=None 给出最小范数解：rms 在自身输入上 ~0，
    但套到无关点上会给出一个编造的答案。这是已知局限，写成文档：真正防护是 Task 10
    solve_pair 的 min_inliers 门限（过滤到至少 30 个匹配点后才会调用 fit_similarity），
    这里不加秩检验——lstsq 已经处理了通用情形，且逐对秩检验会在流水线第二耗时的循环里
    白花运行时间去防一个下游不会发生的场景。
    """
    src = np.array([[10.0, 10.0], [10.0, 10.0]])
    dst = np.array([[20.0, 20.0], [20.0, 20.0]])
    fitted = fit_similarity(src, dst)
    assert np.all(np.isfinite(fitted))
    assert rms(fitted, src, dst) < 1e-10

    d = decompose(fitted)
    assert d["scale"] == pytest.approx(1.990050, abs=1e-5)

    unrelated = apply_transform(fitted, np.array([[500.0, 900.0]]))
    assert unrelated[0] == pytest.approx([995.124, 1791.144], abs=1e-3)


def test_fit_similarity_on_collinear_points():
    """共线点集也能恢复变换——相似变换只需要 2 个不重合点即可确定（Ruling 251）。"""
    src = np.array([[0.0, 0.0], [100.0, 0.0], [200.0, 0.0], [300.0, 0.0]])
    M = similarity_matrix(scale=1.0012, rotation_deg=0.35, tx=-126.0, ty=18.0, center=CENTER)
    dst = apply_transform(M, src)
    fitted = fit_similarity(src, dst)
    assert np.allclose(fitted, M, atol=1e-6)
    assert rms(fitted, src, dst) < 1e-6


def test_fit_similarity_cannot_represent_reflection():
    """4 参数模型不能吸收镜像翻转，rms 保持很大（Ruling 249）。

    src 是原点处 100x50 矩形，dst 是 y 取反后的同一组点——这是纯反射，行列式为负。
    相似变换只有各向同性缩放 + 旋转两个自由度，无法表示行列式变号，所以最小二乘会
    落在一个折衷解上，rms 恰好是 20*sqrt(5)（闭式而非小数字面量，说明数字从哪来，
    对应 R243 的写法）。而 6 参数仿射能精确表示这个翻转（见下一测试），两者对比正是
    模块文档里选 4 参数而不是 6 参数的论据。
    """
    src = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 50.0], [0.0, 50.0]])
    dst = src * np.array([1.0, -1.0])
    fitted = fit_similarity(src, dst)
    d = decompose(fitted)
    assert d["scale"] == pytest.approx(0.6, abs=1e-9)
    assert d["rotation_deg"] == pytest.approx(0.0, abs=1e-9)
    assert rms(fitted, src, dst) == pytest.approx(20.0 * np.sqrt(5.0), rel=1e-9)


def test_affine_6param_can_represent_reflection_that_4param_cannot():
    """同一份反射数据，6 参数仿射拟合的 rms 远小于 4 参数相似变换（Ruling 249）。

    直接用 lstsq 拟合 6 参数仿射（不经过本模块的接口，本模块不提供仿射拟合）作对照：
    diag 应该是 [+1, -1]，rms 落到 ~1.6e-14，与上一测试的 44.72 相差 9 个数量级。
    """
    src = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 50.0], [0.0, 50.0]])
    dst = src * np.array([1.0, -1.0])

    n = len(src)
    A = np.zeros((2 * n, 6), dtype=np.float64)
    b = np.zeros(2 * n, dtype=np.float64)
    A[0::2, 0] = src[:, 0]
    A[0::2, 1] = src[:, 1]
    A[0::2, 4] = 1.0
    A[1::2, 2] = src[:, 0]
    A[1::2, 3] = src[:, 1]
    A[1::2, 5] = 1.0
    b[0::2] = dst[:, 0]
    b[1::2] = dst[:, 1]
    p, *_ = np.linalg.lstsq(A, b, rcond=None)
    a11, a12, a21, a22, tx, ty = p
    affine = np.array([[a11, a12, tx], [a21, a22, ty], [0.0, 0.0, 1.0]])

    fitted_affine = src @ affine[:2, :2].T + affine[:2, 2]
    diff = fitted_affine - dst
    affine_rms = float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))

    assert affine_rms == pytest.approx(1.598e-14, abs=1e-12)
    assert a11 == pytest.approx(1.0, abs=1e-9)
    assert a22 == pytest.approx(-1.0, abs=1e-9)


def test_residuals_shape_and_values():
    M = similarity_matrix(tx=1.0)
    src = np.array([[0.0, 0.0], [10.0, 10.0]])
    dst = np.array([[1.0, 0.0], [11.0, 13.0]])
    r = residuals(M, src, dst)
    assert r.shape == (2,)
    assert r[0] == pytest.approx(0.0)
    assert r[1] == pytest.approx(3.0)


def test_residuals_on_empty_input():
    """空输入的 residuals 形状是 (0,)（Ruling 40）。"""
    M = similarity_matrix()
    r = residuals(M, np.zeros((0, 2)), np.zeros((0, 2)))
    assert r.shape == (0,)


def test_rms_on_empty_input_is_nan():
    """空输入的 rms 是 nan，不是 0.0（Ruling 40）。

    在 Task 10 的 rms_max_px 报告里，一个静默的 0.0 会被误读成完美拟合；用 nan
    才能让下游区分「没有匹配点」和「拟合得极好」。
    """
    M = similarity_matrix()
    result = rms(M, np.zeros((0, 2)), np.zeros((0, 2)))
    assert np.isnan(result)


def test_invert_roundtrip():
    M = similarity_matrix(scale=1.02, rotation_deg=7.0, tx=30.0, ty=-9.0, center=CENTER)
    xy = np.array([[100.0, 200.0], [3000.0, 900.0]])
    back = apply_transform(invert(M), apply_transform(M, xy))
    assert np.allclose(back, xy, atol=1e-8)


def test_invert_singular_matrix_raises():
    """scale=0 的相似矩阵不可逆，invert 抛出 LinAlgError（Ruling 40）。

    RegistrationResult.to_detector 会对累积矩阵调用 invert，退化链条必须响亮地失败
    而不是静默返回垃圾。
    """
    M = similarity_matrix(scale=0.0)
    with pytest.raises(np.linalg.LinAlgError):
        invert(M)


def test_apply_transform_on_empty():
    got = apply_transform(similarity_matrix(), np.zeros((0, 2)))
    assert got.shape == (0, 2)


def test_apply_transform_reshapes_flat_array_of_valid_size():
    """长度为偶数的一维数组被静默 reshape 成 (N, 2)（Ruling 40）。"""
    M = similarity_matrix()
    got = apply_transform(M, np.array([1.0, 2.0, 3.0, 4.0]))
    assert np.allclose(got, [[1.0, 2.0], [3.0, 4.0]])


def test_apply_transform_rejects_flat_array_of_invalid_size():
    """长度不是偶数时抛中文 ``ValueError``（Ruling 40 的性质，消息已换）。

    Ruling 40 钉的性质是「奇数长度的扁平输入必须抛错」，它没有变。变的是消息
    来源：原先是 numpy 的 ``cannot reshape array of size 3 into shape (2)``，
    现在点集校验集中到 ``src.pointset.as_xy``，消息带上了实参名与形状。
    逐字匹配 numpy 的内部措辞本来就把测试绑在了 numpy 的版本上，
    换成本项目自己的消息反而更稳。
    """
    M = similarity_matrix()
    with pytest.raises(ValueError, match=r"xy 形状非法: \(3,\)"):
        apply_transform(M, np.array([1.0, 2.0, 3.0]))


def test_composition_matches_sequential_application():
    A = similarity_matrix(rotation_deg=1.0, tx=10.0, center=CENTER)
    B = similarity_matrix(scale=1.001, ty=-4.0, center=CENTER)
    xy = np.array([[500.0, 600.0]])
    assert np.allclose(apply_transform(B @ A, xy), apply_transform(B, apply_transform(A, xy)))
