"""交付成果图：8 张图，全部标注中文。

字体守卫为什么是字形级的
------------------------

只检查「字体名非空」且「名字在 ``fontManager.ttflist`` 里」是**无效的守卫**。
实测（本镜像，matplotlib 3.10.9，20 个可见字体，见 ``probe/t26-font.txt``）：
``DejaVu Sans`` 两条都过，而它对本模块实际要画的汉字**一个字形都没有**
（``FIGURE_CJK`` 全部缺失）。那样的图每个汉字都是空心方框，而断言全绿。

唯一可靠的判据是字形索引：``FT2Font.get_char_index(ord(ch)) == 0`` 意味着该
字符在该字体里不存在。本镜像 20 个字体里只有 ``SimHei``
（``/usr/share/fonts/truetype/simhei.ttf``，28562 字形）覆盖 ``FIGURE_CJK``；
其余 19 个（DejaVu 5 + STIX 7 + cm 7）全部 100% 缺失。

两个必须绕开的缺失字形（SimHei 也没有，索引都是 0）
--------------------------------------------------

- ``U+2212 MINUS SIGN``：matplotlib 的负刻度默认用它。ASCII 连字符
  ``U+002D`` 在 SimHei 里是索引 16，所以 ``axes.unicode_minus = False``
  是**载荷代码**，不是风格偏好。
- ``U+00B2 SUPERSCRIPT TWO``：面积单位一律写 ASCII 的 ``arcsec^2``。
  ``FIGURE_CJK`` 里不得出现 ``²``。

存在的符号（实测索引）：``″``=269、``σ``=183、``Δ``=145、``±``=102。

``probe=True`` 的用途
---------------------

八个 ``plot_*`` 统一多一个 ``probe`` 关键字。默认返回 ``Path``；
``probe=True`` 返回 dict，含 ``path``、``all_labels``（本图所有标题/轴标签/
图例/文本块的文字）以及该图特有的量。

用返回值而不是模块级 ``last_axis_info()``：模块级可变状态在并行测试下互相
污染，且只能服务最后一次调用。要探的量在 ``plt.close(fig)`` **之前**抄进
普通 dict，所以不持有 figure、不泄漏。

口径约束（写进图上的文字，答辩直接引用）
----------------------------------------

- 图 5 的位置轴是**恒星系像素**，不是赤经赤纬；MVP 不报位置角
  （``TrajectoryPoint`` 的 ``ra_deg``/``dec_deg``/``pa_deg`` 恒为 ``None``）。
- 图 6 只报自相容区间，不声称与外部星历一致。
- 图 7 第 0 行的剔除依据是**数值量级**，不是曝光响应；剔除数量如实报出。
- 图 8 的天光面亮度与 ``fwhm_arcsec`` 是**真值定标量**，只作条件命题、
  **不报小数**（整数 ADU 量化地板 0.18 mag，零点散度 0.22 mag）。
- 极限星等是**经验暗端百分位**，不是「最暗可复现源」。
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg", force=True)   # 必须在任何 pyplot import 之前

import matplotlib.pyplot as plt     # noqa: E402
import numpy as np                  # noqa: E402
from matplotlib import font_manager as fm   # noqa: E402
from matplotlib.ft2font import FT2Font      # noqa: E402

from src.config import load_config  # noqa: E402

# 本模块全部图表实际会画到的汉字与特殊符号，去重后按码点排序（307 个）。
# 由 probe/t26-collect-cjk.py 从八张图（连同它们的退化分支，共 117 处标签）
# probe=True 的 all_labels 里收集而来，不是手编的样本 ——
# test_figure_cjk_is_actually_what_the_figures_draw 会逐字复核这个超集关系。
# 注意这里**没有** `²`：SimHei 的 U+00B2 字形索引是 0，面积单位写 `arcsec^2`。
# 同理 `°`(U+00B0) 与 `×`(U+00D7) 已从标签里换掉（写「度」与 ASCII `x`），
# 它们不属于 CJK 也不属于本项目钉死的符号白名单。
FIGURE_CJK = (
    "±σ″、一三上下不与个中为之乎乘于交亮仅从仰件伏会传伪伸位低例供依倍值"
    "像元充光入全其内决准出分列判到剔加动化匹区半单历原参叉双反取变叠口只"
    "可号合同后向否含周命和器图圆圈在地均坐型填声复外多大天如始字宁定实容"
    "宽寸对小少尺展差已布带帧平并应度弧影径必态性总恒感成所才折报拉拖按据"
    "探推提搜收故效敛散数整方无时星是普景暗曝曲最有期未本机束条来板极架查"
    "标格档模止步残母比毫汇法流测源漂演灰点热焦状独率现球理用由界百的目相"
    "看真确示秒称移空立端第等算簇系素索约级纯线经结给绝续缀缺置考背自至致"
    "荐落行被见观视角解计认证该读调赤起超距跟踪轨轮轴输达过运近远连迹逐通"
    "速道部都配量钟锁长门间阈限除陷集零静非面顶须颗题饱首验骗高（），：；"
)

# 候选字体的模块级兜底，仅当配置缺 `viz.font_candidates` 时使用。
# 内容必须与 src/config/default.yaml 一致，且**不含 DejaVu Sans**
# （它是方框源，见模块 docstring）。
_FALLBACK_CANDIDATES: tuple[str, ...] = (
    "SimHei",
    "Noto Sans CJK SC",
    "Noto Sans CJK JP",
    "WenQuanYi Zen Hei",
)


class FontUnavailable(RuntimeError):
    """候选字体全部不可用（不存在，或缺失本项目要画的字形）。"""


def _viz_cfg() -> dict[str, Any]:
    """读 `viz` 配置块。阈值与候选表只有这一份真相。"""
    config = load_config()
    section = config.get("viz")
    if not isinstance(section, dict):
        raise ValueError("配置缺少 viz 块，无法确定字体候选与 dpi")
    return section


@lru_cache(maxsize=64)
def _installed_font_names() -> frozenset[str]:
    """matplotlib 当前可见的字体名集合。"""
    return frozenset(f.name for f in fm.fontManager.ttflist)


@lru_cache(maxsize=64)
def _font_file(font_name: str) -> str | None:
    """字体名 -> 字体文件路径；不存在时返回 None（不回落到默认字体）。"""
    try:
        return fm.findfont(
            fm.FontProperties(family=font_name), fallback_to_default=False
        )
    except Exception:
        return None


@lru_cache(maxsize=256)
def _missing_glyphs(font_name: str, sample: str) -> tuple[str, ...]:
    """缓存一份逐字符字形查询结果（FT2Font 打开文件是这里唯一的开销）。"""
    path = _font_file(font_name)
    if path is None:
        return tuple(sample)
    try:
        face = FT2Font(path)
    except Exception:
        return tuple(sample)
    return tuple(ch for ch in sample if face.get_char_index(ord(ch)) == 0)


def glyphs_missing(font_name: str, sample: str = FIGURE_CJK) -> list[str]:
    """返回 `sample` 里该字体没有字形的字符。

    字形索引 0 = 该字符在该字体里不存在，这是唯一可靠的判据；只比对字体名字
    会放过 DejaVu Sans（实测它缺 `FIGURE_CJK` 的全部汉字）。字体本身不存在时
    视作全部缺失。
    """
    return list(_missing_glyphs(font_name, sample))


def setup_matplotlib(candidates: Sequence[str] | None = None) -> str:
    """选中文字体、定 rcParams，返回选中的字体名。

    两段判定：候选既要出现在 `fontManager.ttflist` 里，又要通过
    `glyphs_missing(...) == []` 的字形级检查。全不满足时抛 `FontUnavailable`,
    消息里列出**每个**尝试过的名字及其被拒原因（不存在 / 缺 N 个字形），
    否则排查只能靠猜。
    """
    if candidates is None:
        configured = _viz_cfg().get("font_candidates")
        names: tuple[str, ...] = (
            tuple(str(n) for n in configured)
            if isinstance(configured, (list, tuple)) and configured
            else _FALLBACK_CANDIDATES
        )
    else:
        names = tuple(str(n) for n in candidates)

    installed = _installed_font_names()
    rejected: list[str] = []
    chosen: str | None = None
    for name in names:
        if name not in installed:
            rejected.append(f"{name}（未安装）")
            continue
        missing = glyphs_missing(name, FIGURE_CJK)
        if missing:
            rejected.append(
                f"{name}（缺 {len(missing)}/{len(FIGURE_CJK)} 个字形："
                f"{''.join(missing[:8])}）"
            )
            continue
        chosen = name
        break

    if chosen is None:
        raise FontUnavailable(
            "候选字体全部不可用，图上的中文会画成空心方框。逐个原因："
            + "；".join(rejected)
            + "。请在镜像里安装含汉字的字体（如 SimHei 或 fonts-wqy-zenhei）"
            "并加入 src/config/default.yaml 的 viz.font_candidates"
        )

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [chosen]
    # U+2212 在 SimHei 里字形索引为 0（ASCII 连字符是 16），负刻度必须换回 ASCII。
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = _dpi()
    plt.rcParams["savefig.dpi"] = _dpi()
    return chosen


def _dpi() -> int:
    """出图 dpi，来自 `viz.dpi`。"""
    return int(_viz_cfg()["dpi"])


def _collect_labels(fig) -> list[str]:
    """抄出这张图上的每一处文字：标题、轴标签、图例、文本块。

    必须在 `plt.close(fig)` 之前调用 —— 返回的是普通字符串列表，
    不持有任何 matplotlib 对象。
    """
    labels: list[str] = [t.get_text() for t in fig.texts]
    for ax in fig.axes:
        labels.append(ax.get_title())
        labels.append(ax.get_xlabel())
        labels.append(ax.get_ylabel())
        labels.extend(t.get_text() for t in ax.texts)
        legend = ax.get_legend()
        if legend is not None:
            labels.extend(t.get_text() for t in legend.get_texts())
    return [text for text in labels if text]


def _save(fig, path: str | Path) -> Path:
    """存盘并**关闭** figure。

    `plt.close(fig)` 不是清洁工作：一次出 8 张图、循环出多张时漏掉它会撞上
    matplotlib 的 `More than 20 figures have been opened` 警告并把图对象攒在
    内存里（4096² 的图各自持有一份像素缓存）。
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=_dpi())
    plt.close(fig)
    return out


def _finish(fig, path: str | Path, *, probe: bool, extra: dict[str, Any] | None = None):
    """统一收尾：抄探针量 -> 存盘关闭 -> 按 `probe` 返回 Path 或 dict。"""
    labels = _collect_labels(fig)
    n_axes = len(fig.axes)
    out = _save(fig, path)
    if not probe:
        return out
    info: dict[str, Any] = {"path": out, "all_labels": labels, "n_axes": n_axes}
    if extra:
        info.update(extra)
    return info


def _legend(ax) -> None:
    """只在确实有带标签的图元时画图例，避免 matplotlib 的空图例警告。"""
    handles, labels = ax.get_legend_handles_labels()
    if handles and labels:
        ax.legend(loc="best", fontsize=9)


def _marker_sizes(values: np.ndarray, *, lo: float = 18.0, hi: float = 90.0) -> np.ndarray:
    """把流量映射成散点面积。空数组与零极差都必须给出合法 sizes。"""
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return np.zeros(0, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0 or np.ptp(finite) <= 0.0:
        return np.full(values.shape, lo, dtype=np.float64)
    low, high = float(finite.min()), float(finite.max())
    scaled = (np.clip(values, low, high) - low) / (high - low)
    return lo + (hi - lo) * scaled


def _stretch(image: np.ndarray) -> tuple[float, float]:
    """显示用的百分位拉伸区间。空图像或全非有限值时退回 (0, 1)。"""
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = (float(v) for v in np.percentile(finite, (1.0, 99.5)))
    if not (hi > lo):
        return lo, lo + 1.0
    return lo, hi


# ===========================================================================
# 图 1：原图 + 探测源叠加
# ===========================================================================

def plot_detections(
    image: np.ndarray,
    table: Any,
    path: str | Path,
    *,
    target_xy: tuple[float, float] | None = None,
    hot_clusters: Iterable[Any] | None = None,
    probe: bool = False,
):
    """原始帧灰度底图 + 探测源圈选，可选叠加目标位置与热像素簇。

    空源表是常态（弱帧、高阈值档），必须画出图并在图上写明「0 个」，
    不得静默画一张空图 —— `np.percentile` 在空流量数组上抛 IndexError,
    `_marker_sizes` 与图例都按空表验过。
    """
    setup_matplotlib()
    image = np.asarray(image, dtype=np.float64)
    n_sources = len(table)
    vmin, vmax = _stretch(image)

    fig, ax = plt.subplots(figsize=(10.0, 8.0))
    ax.imshow(image, cmap="gray", origin="upper", vmin=vmin, vmax=vmax,
              interpolation="nearest")
    ax.scatter(table.x, table.y, s=_marker_sizes(table.flux), facecolors="none",
               edgecolors="#1f9d55", linewidths=1.1,
               label=f"探测源 {n_sources} 个（圈大小按流量）")

    n_hot = 0
    if hot_clusters is not None:
        clusters = list(hot_clusters)
        n_hot = len(clusters)
        ax.scatter([c.cx for c in clusters], [c.cy for c in clusters],
                   marker="x", s=70, color="#d64545", linewidths=1.6,
                   label=f"热像素簇 {n_hot} 个（已从探测中剔除）")

    has_target = target_xy is not None
    if has_target:
        ax.scatter([target_xy[0]], [target_xy[1]], marker="o", s=260,
                   facecolors="none", edgecolors="#f0a020", linewidths=2.2,
                   label="双系确认的目标")

    ax.set_title(f"原图与探测源叠加：{n_sources} 个源")
    ax.set_xlabel("探测器系像素坐标 x（列，0 起算）")
    ax.set_ylabel("探测器系像素坐标 y（行，0 起算）")
    ax.text(0.02, 0.02, f"灰度拉伸区间 {vmin:.1f} ~ {vmax:.1f} ADU（1 至 99.5 百分位）",
            transform=ax.transAxes, fontsize=9, color="#f5f5f5",
            va="bottom", ha="left")
    _legend(ax)
    fig.tight_layout()
    return _finish(fig, path, probe=probe, extra={
        "n_sources": n_sources,
        "n_hot_clusters": n_hot,
        "has_target": has_target,
    })


# ===========================================================================
# 图 2：阈值-复现率曲线
# ===========================================================================

def plot_threshold_curve(curve: Any, path: str | Path, *, title: str | None = None,
                         probe: bool = False):
    """阈值扫描的复现率与纯度双曲线，并标出 `recommended()` 选中的档位。

    「约 10² 颗可复现源」这个结论的全部依据就是被标出的那一个点，图上不标
    等于图没讲完。`recommended()` 在没有任何档达标时抛 `ValueError`
    （单点低复现率曲线就是这种情形），此时如实写明「无达标档位」而不是崩。
    """
    setup_matplotlib()
    points = list(curve.points)
    sigmas = np.array([p.n_sigma for p in points], dtype=np.float64)
    repro = np.array([p.reproducibility for p in points], dtype=np.float64)
    purity = np.array([p.purity for p in points], dtype=np.float64)
    counts = np.array([p.n_reproducible for p in points], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(9.0, 6.0))
    ax.plot(sigmas, repro, marker="o", color="#1f77b4", label="复现率（分母为参考帧源数）")
    ax.plot(sigmas, purity, marker="s", color="#8c564b", linestyle="--",
            label="纯度（分母为逐帧平均探测数）")
    ax.set_xlabel("探测阈值（背景标准差的倍数，σ）")
    ax.set_ylabel("比例")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)

    ax_count = ax.twinx()
    ax_count.plot(sigmas, counts, marker="^", color="#2ca02c", linestyle=":",
                  label="可复现源数（颗）")
    ax_count.set_ylabel("可复现源数（颗）")

    marked_sigma: float | None = None
    try:
        best = curve.recommended()
    except ValueError:
        ax.text(0.5, 0.5, "无达标档位：所有阈值的复现率都低于门限",
                transform=ax.transAxes, ha="center", va="center", fontsize=11,
                color="#a33")
    else:
        marked_sigma = float(best.n_sigma)
        ax.axvline(marked_sigma, color="#d62728", linewidth=1.4, alpha=0.85,
                   label=f"推荐阈值 {marked_sigma:.1f}σ（单调合格后缀的最低档）")
        ax.annotate(
            f"推荐 {marked_sigma:.1f}σ：复现率 {best.reproducibility:.3f}、"
            f"可复现 {best.n_reproducible} 颗",
            xy=(marked_sigma, best.reproducibility),
            xytext=(0.42, 0.28), textcoords="axes fraction", fontsize=10,
            arrowprops={"arrowstyle": "->", "color": "#d62728"},
        )

    ax.set_title(title or f"阈值与复现率曲线（{len(curve.frames)} 帧，"
                          f"匹配半径 {curve.match_radius_px:.1f} px）")
    handles, labels = ax.get_legend_handles_labels()
    extra_handles, extra_labels = ax_count.get_legend_handles_labels()
    if handles or extra_handles:
        ax.legend(handles + extra_handles, labels + extra_labels,
                  loc="lower right", fontsize=9)
    fig.tight_layout()
    return _finish(fig, path, probe=probe, extra={
        "marked_sigma": marked_sigma,
        "n_points": len(points),
    })


# ===========================================================================
# 图 3：配准残差分布
# ===========================================================================

def plot_registration_residuals(result: Any, path: str | Path, *, probe: bool = False):
    """逐对配准的残差与内点数**并列**两个子图。

    只看 RMS 小说明不了配准好：内点数塌到 3 个时 RMS 必然很小。两个量一起看
    才是证据，所以这张图恒为两个子图。0 对（单帧数据集）时 `report()` 的 rms
    是 nan、inliers 是 0，此时写明「无配准对」而不是崩。
    """
    setup_matplotlib()
    report = result.report()
    pairs = list(result.pairs)
    rms = np.array([p.rms_px for p in pairs], dtype=np.float64)
    inliers = np.array([p.n_inliers for p in pairs], dtype=np.float64)
    frames_from = np.array([p.frame_from for p in pairs], dtype=np.float64)

    fig, (ax_rms, ax_in) = plt.subplots(2, 1, figsize=(10.0, 8.0), sharex=False)

    if pairs:
        ax_rms.hist(rms, bins=min(20, max(4, len(pairs) // 2)), color="#1f77b4",
                    edgecolor="white",
                    label=f"{len(pairs)} 对，中位 {report['rms_median_px']:.3f} px")
        ax_rms.axvline(report["rms_median_px"], color="#d62728", linewidth=1.4,
                       label=f"中位残差 {report['rms_median_px']:.3f} px")
        ax_rms.axvline(report["rms_max_px"], color="#f0a020", linestyle="--",
                       linewidth=1.3, label=f"最大残差 {report['rms_max_px']:.3f} px")
        ax_in.plot(frames_from, inliers, marker="o", color="#2ca02c",
                   label=f"内点数（中位 {report['inliers_median']:.0f}，"
                         f"最少 {report['inliers_min']}）")
        ax_in.axhline(report["inliers_min"], color="#d62728", linestyle=":",
                      linewidth=1.3, label=f"最少内点 {report['inliers_min']}")
    else:
        for ax in (ax_rms, ax_in):
            ax.text(0.5, 0.5, "无配准对：该数据集只有单帧，无法给出逐对残差",
                    transform=ax.transAxes, ha="center", va="center", fontsize=12,
                    color="#a33")

    ax_rms.set_title(f"配准残差分布（参考帧 {report['reference_frame']}，"
                     f"{report['n_frames']} 帧、{report['n_pairs']} 对）")
    ax_rms.set_xlabel("逐对残差 RMS（px）")
    ax_rms.set_ylabel("对数")
    ax_rms.grid(alpha=0.3)
    _legend(ax_rms)

    ax_in.set_title("逐对内点数：残差小必须与内点多同时成立，单看残差会被少内点的解骗过")
    ax_in.set_xlabel("起始帧号")
    ax_in.set_ylabel("内点数（对）")
    ax_in.grid(alpha=0.3)
    _legend(ax_in)

    fig.tight_layout()
    return _finish(fig, path, probe=probe, extra={
        "rms_median_px": report["rms_median_px"],
        "rms_max_px": report["rms_max_px"],
        "inliers_min": report["inliers_min"],
        "n_pairs": report["n_pairs"],
    })


# ===========================================================================
# 图 4：双坐标系目标判决（本项目主亮点）
# ===========================================================================

def plot_dual_frame_verdict(verdict: Any, path: str | Path, *, probe: bool = False):
    """探测器系与恒星系两条轨迹**并列**，附判决理由。

    这张图的全部意义在于两条轨迹的对比：机架在跟踪目标，所以目标在探测器系
    近乎静止；而它相对恒星背景高速运动，所以在恒星系里划出一条长线。只画一条
    等于把主亮点画没了。

    热像素的 `v_det` 恰为 0，静止性对比度是 `inf`，格式化后写出「inf」是可接受
    的；但任何把它喂给 `set_xlim` 之类的路径都会崩，所以对比度只进文字。
    """
    setup_matplotlib()
    track = verdict.track
    det = np.asarray(track.xy_det, dtype=np.float64)
    sky = np.asarray(track.xy_sky, dtype=np.float64)

    fig, (ax_det, ax_sky) = plt.subplots(1, 2, figsize=(12.0, 6.0))
    ax_det.plot(det[:, 0], det[:, 1], marker="o", markersize=3.5, color="#1f77b4",
                label=f"探测器系轨迹（{track.length} 帧，总漂移 {track.drift_det_px:.2f} px）")
    ax_sky.plot(sky[:, 0], sky[:, 1], marker="o", markersize=3.5, color="#d62728",
                label=f"恒星系轨迹（总漂移 {track.drift_sky_px:.1f} px）")

    ax_det.set_title(f"探测器系：机架在跟踪，目标近乎静止（{track.v_det_px:.2f} px/帧）")
    ax_sky.set_title(f"恒星系：相对恒星背景高速运动（{track.v_sky_px:.1f} px/帧）")
    for ax in (ax_det, ax_sky):
        ax.set_xlabel("像素坐标 x（列，0 起算）")
        ax.set_ylabel("像素坐标 y（行，0 起算）")
        ax.grid(alpha=0.3)
        ax.margins(0.15)
        _legend(ax)

    contrast = float(verdict.contrast)
    head = "判为空间目标" if verdict.is_target else "判为非目标"
    lines = [
        f"判决：{head}",
        f"逐帧位移中位数：探测器系 {verdict.v_det_px:.2f} px/帧、"
        f"恒星系 {verdict.v_sky_px:.1f} px/帧",
        f"静止性对比度：{contrast:.1f} 倍" if np.isfinite(contrast)
        else "静止性对比度：inf（探测器系完全不动，热像素的特征）",
        f"像素锁定：{'是' if verdict.pixel_locked else '否'}",
    ]
    reasons = list(verdict.reasons)
    lines.extend(f"理由 {k}：{text}" for k, text in enumerate(reasons, start=1))
    fig.text(0.01, 0.01, "\n".join(lines), fontsize=9, va="bottom", ha="left",
             family="sans-serif")

    fig.suptitle("双坐标系交叉确认：三条独立证据同时成立才判为空间目标", fontsize=13)
    fig.tight_layout(rect=(0.0, 0.16, 1.0, 0.96))
    # 从**实际画上去的 Line2D** 数出来，不用手递增的计数器：
    # 计数器与绘制是两处代码，删掉一次 plot 而忘记减计数器时它会说谎。
    n_tracks_drawn = sum(len(ax.lines) for ax in (ax_det, ax_sky))
    return _finish(fig, path, probe=probe, extra={
        "n_tracks_drawn": n_tracks_drawn,
        "drift_det_px": track.drift_det_px,
        "drift_sky_px": track.drift_sky_px,
        "is_target": bool(verdict.is_target),
        "n_reasons": len(reasons),
        "contrast": contrast,
    })


# ===========================================================================
# 图 5：目标轨迹 + 角速度
# ===========================================================================

def plot_trajectory(trajectory: Any, path: str | Path, *, probe: bool = False):
    """恒星系轨迹与逐步角速度。

    **位置轴是恒星系像素，不是赤经赤纬**：本项目的 FITS 头没有 WCS、没有焦距、
    没有像元尺寸，`ra_deg`/`dec_deg`/`pa_deg` 在 MVP 里恒为 `None`
    （见 `src.target.trajectory.TrajectoryPoint` 的 docstring）。位置角同理不报。

    图上标的角速度均值取 `mean_rate_arcsec_s`，它走 `_independent_rates`、
    **排除 points[0]** 那个前向填充值；自己按全部 n 点重算会差 0.26 ″/s。
    """
    setup_matplotlib()
    points = list(trajectory.points)
    x_sky = np.array([p.x_sky for p in points], dtype=np.float64)
    y_sky = np.array([p.y_sky for p in points], dtype=np.float64)
    frames = np.array([p.frame for p in points], dtype=np.float64)
    rates = np.array([p.rate_arcsec_s for p in points], dtype=np.float64)
    mean_rate = float(trajectory.mean_rate_arcsec_s)
    std_rate = float(trajectory.rate_std_arcsec_s)

    fig, (ax_pos, ax_rate) = plt.subplots(1, 2, figsize=(12.0, 6.0))
    ax_pos.plot(x_sky, y_sky, marker="o", markersize=3.5, color="#d62728",
                label=f"恒星系轨迹（{len(points)} 帧）")
    if len(points):
        ax_pos.scatter([x_sky[0]], [y_sky[0]], marker="s", s=70, color="#1f77b4",
                       label=f"起始帧 {points[0].frame}")
        ax_pos.scatter([x_sky[-1]], [y_sky[-1]], marker="D", s=70, color="#2ca02c",
                       label=f"结束帧 {points[-1].frame}")
    ax_pos.set_title("目标轨迹（恒星系像素坐标，非赤道坐标）")
    ax_pos.set_xlabel("恒星系像素坐标 x（列，0 起算）")
    ax_pos.set_ylabel("恒星系像素坐标 y（行，0 起算）")
    ax_pos.grid(alpha=0.3)
    ax_pos.margins(0.12)
    _legend(ax_pos)

    ax_rate.plot(frames, rates, marker="o", markersize=3.5, color="#1f77b4",
                 label="逐步角速度")
    ax_rate.axhline(mean_rate, color="#d62728", linewidth=1.4,
                    label=f"均值 {mean_rate:.2f} ″/s（{max(len(points) - 1, 0)} 个独立步，"
                          f"不含首点填充值）")
    if np.isfinite(mean_rate) and np.isfinite(std_rate):
        ax_rate.axhspan(mean_rate - std_rate, mean_rate + std_rate, color="#d62728",
                        alpha=0.12, label=f"± 1σ 散度 {std_rate:.2f} ″/s")
    ax_rate.set_title(f"角速度：均值 {mean_rate:.2f} ″/s，"
                      f"弧长 {trajectory.total_arc_arcsec:.0f} ″")
    ax_rate.set_xlabel("帧号")
    ax_rate.set_ylabel("角速度（″/s）")
    ax_rate.grid(alpha=0.3)
    _legend(ax_rate)

    fig.text(0.01, 0.01,
             "位置轴为恒星系像素：本轮无绝对定向来源（无 WCS、无焦距、无像元尺寸），"
             "不报赤道坐标与方位角",
             fontsize=9, va="bottom", ha="left")
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
    return _finish(fig, path, probe=probe, extra={
        "mean_rate_arcsec_s": mean_rate,
        "rate_std_arcsec_s": std_rate,
        "n_points": len(points),
    })


# ===========================================================================
# 图 6：高度反演区间
# ===========================================================================

def plot_height_interval(estimate: Any, path: str | Path, *, probe: bool = False):
    """高度反演的**自相容区间**，画成带宽而不是一个单值。

    把区间中点画成「测得的高度」是答辩现场的原理性质疑：那个中点只供单值展示
    （见 `OrbitEstimate` 的 docstring）。所以这张图的载荷是带宽本身，两个端点
    逐字标出。不收敛时两端点都是 `None`，写明「未收敛」而不是 TypeError。

    图上不写 `estimate.note`：交付的 note 里带「未与公开根数比对」这类字样，
    而口径要求图面完全不提外部根数 —— 少说比解释更安全。
    """
    setup_matplotlib()
    low = estimate.height_min_km
    high = estimate.height_max_km
    width = estimate.band_width_km
    has_band = low is not None and high is not None

    fig, ax = plt.subplots(figsize=(9.0, 6.0))
    if has_band:
        low_km, high_km = float(low), float(high)
        ax.axhspan(low_km, high_km, xmin=0.12, xmax=0.88, color="#1f77b4", alpha=0.28,
                   label=f"自相容高度区间，带宽 {float(width):.2f} km")
        for value, name, color in ((low_km, "下界", "#2ca02c"), (high_km, "上界", "#d62728")):
            ax.axhline(value, xmin=0.12, xmax=0.88, color=color, linewidth=1.6)
            ax.annotate(f"{name} {value:.2f} km", xy=(0.88, value),
                        xytext=(0.90, value), textcoords=("axes fraction", "data"),
                        va="center", fontsize=10, color=color)
        if estimate.height_km is not None:
            ax.axhline(float(estimate.height_km), xmin=0.12, xmax=0.88, color="#666666",
                       linestyle=":", linewidth=1.2,
                       label=f"区间中点 {float(estimate.height_km):.2f} km"
                             "（仅供单值展示，不是测量值）")
        margin = max(0.08 * float(width), 20.0)
        ax.set_ylim(low_km - margin, high_km + margin)
        band_low, band_high, band_width = low_km, high_km, float(width)
    else:
        ax.text(0.5, 0.5, "反演未收敛：角速度落在圆轨道模型的可解区间之外，不给出高度",
                transform=ax.transAxes, ha="center", va="center", fontsize=12,
                color="#a33")
        ax.set_ylim(0.0, 1.0)
        band_low = band_high = band_width = None

    ax.set_xlim(0.0, 1.0)
    ax.set_xticks([])
    ax.set_title(f"高度反演：只报自相容区间（角速度 {estimate.rate_arcsec_s:.2f} ″/s，"
                 f"仰角 {estimate.elevation_deg:.1f} 度）")
    ax.set_ylabel("轨道高度（km）")
    ax.set_xlabel(f"圆轨道自相容反演，搜索区间 "
                  f"{estimate.bracket_km[0]:.0f} ~ {estimate.bracket_km[1]:.0f} km")
    ax.grid(alpha=0.3, axis="y")
    if estimate.period_min_min is not None and estimate.period_max_min is not None:
        ax.text(0.02, 0.02,
                f"对应周期区间 {float(estimate.period_min_min):.1f} ~ "
                f"{float(estimate.period_max_min):.1f} 分钟",
                transform=ax.transAxes, fontsize=9, va="bottom", ha="left")
    ax.text(0.02, 0.96, "仅报自相容区间，不声称与外部星历一致",
            transform=ax.transAxes, fontsize=9, va="top", ha="left", color="#444444")
    _legend(ax)
    fig.tight_layout()
    return _finish(fig, path, probe=probe, extra={
        "band_low_km": band_low,
        "band_high_km": band_high,
        "band_width_km": band_width,
        "converged": bool(estimate.converged),
    })


# ===========================================================================
# 图 7：热像素 / 传感器缺陷
# ===========================================================================

def plot_hotpixels(
    hotpixel_map: Any,
    path: str | Path,
    *,
    shape: tuple[int, int] | None = None,
    report: Any | None = None,
    probe: bool = False,
):
    """缺陷簇在探测器上的位置分布。

    只读 `.clusters`，从不碰 `.mask`（4096² 的 bool 掩膜实测 16.0 MiB，
    画位置分布不需要它），所以 `shape` 由调用方给出或从掩膜形状推断。

    带 `report=` 时把读出伪影的剔除数量如实写在图上：剔除依据是**数值量级**
    （实测 px(0,0) 是同帧中部行中位数的数千倍），不是曝光响应。
    """
    setup_matplotlib()
    clusters = list(hotpixel_map.clusters)
    if shape is None:
        shape = tuple(int(v) for v in np.asarray(hotpixel_map.mask).shape[:2])
    ny, nx = int(shape[0]), int(shape[1])

    fig, ax = plt.subplots(figsize=(10.0, 8.0))
    sizes = _marker_sizes(np.array([c.npix for c in clusters], dtype=np.float64),
                          lo=40.0, hi=180.0)
    ax.scatter([c.cx for c in clusters], [c.cy for c in clusters], s=sizes,
               facecolors="none", edgecolors="#d62728", linewidths=1.6,
               label=f"缺陷簇 {len(clusters)} 个（圈大小按像素数）")
    if not clusters:
        ax.text(0.5, 0.5, "该探测器普查到 0 个缺陷簇", transform=ax.transAxes,
                ha="center", va="center", fontsize=12, color="#2ca02c")

    n_metadata = None
    if report is not None:
        n_metadata = int(report.n_metadata_artifacts)
        ax.axhspan(-0.5, int(report.metadata_rows) - 0.5, color="#f0a020", alpha=0.35,
                   label=f"顶部 {report.metadata_rows} 行状态字（已剔除 {n_metadata} 个伪影）")
        ax.text(0.02, 0.02,
                f"读出伪影 {n_metadata} 个已剔除并如实计入；"
                f"剔除依据为数值量级远超同帧中部行的中位数\n"
                f"背景中位 {report.background_median:.1f} ADU、"
                f"起伏 {report.background_rms:.3f} ADU"
                f"（第 {report.background_frame_index} 帧，"
                f"曝光 {report.background_exposure_ms:.0f} 毫秒）；"
                f"饱和像元 {report.n_saturated} 个",
                transform=ax.transAxes, fontsize=9, va="bottom", ha="left")

    ax.set_xlim(0, nx)
    ax.set_ylim(ny, 0)
    ax.set_title(f"传感器缺陷簇分布（{nx} x {ny} 探测器，{len(clusters)} 个簇）")
    ax.set_xlabel("探测器系像素坐标 x（列，0 起算）")
    ax.set_ylabel("探测器系像素坐标 y（行，0 起算）")
    ax.grid(alpha=0.3)
    _legend(ax)
    fig.tight_layout()
    return _finish(fig, path, probe=probe, extra={
        "n_clusters": len(clusters),
        "n_metadata_artifacts": n_metadata,
        "shape": (ny, nx),
    })


# ===========================================================================
# 图 8：视宁度 + 天光 + 极限星等
# ===========================================================================

def plot_magnitude_histogram(
    magnitudes: np.ndarray,
    path: str | Path,
    *,
    limiting: float | None = None,
    seeing: Any | None = None,
    sky_mag_arcsec2: float | None = None,
    probe: bool = False,
):
    """星等分布直方图，右侧并列视宁度与天光面亮度。

    三条口径写死在标签里：

    - 极限星等是**经验暗端百分位**，不是「最暗可复现源」。
    - 天光面亮度与 `fwhm_arcsec` 都乘了真值定出的板比例与零点，是**真值定标量**,
      只能作条件命题，不得声称独立验证。
    - 天光面亮度**不报小数**：背景中位是整数 ADU，1 ADU 在 6.0 上折 0.18 mag,
      零点自身散度 0.22 mag，小数位没有信息。单位写 ASCII 的 `arcsec^2`
      —— `²` U+00B2 在 SimHei 里字形索引是 0。
    """
    setup_matplotlib()
    values = np.asarray(magnitudes, dtype=np.float64)
    finite = values[np.isfinite(values)]

    fig, (ax, ax_info) = plt.subplots(1, 2, figsize=(12.0, 6.0),
                                      gridspec_kw={"width_ratios": [2.0, 1.0]})
    if finite.size:
        ax.hist(finite, bins=min(30, max(5, finite.size // 10)), color="#1f77b4",
                edgecolor="white", label=f"有效星等 {finite.size} 个")
    else:
        ax.text(0.5, 0.5, "无有效星等：零点不可用，全部为非有限值",
                transform=ax.transAxes, ha="center", va="center", fontsize=12,
                color="#a33")

    if limiting is not None:
        ax.axvline(float(limiting), color="#d62728", linewidth=1.6,
                   label=f"极限星等 {float(limiting):.1f}（经验暗端百分位口径）")
    ax.set_title(f"星等分布与极限星等（{finite.size} 个有效源）")
    ax.set_xlabel("星等（真值零点定标）")
    ax.set_ylabel("源数（个）")
    ax.grid(alpha=0.3)
    _legend(ax)

    ax_info.axis("off")
    lines = ["观测条件汇总"]
    if seeing is not None:
        lines.append(f"合成星像宽度：{float(seeing.fwhm_px):.2f} px"
                     f"（{seeing.n_stars} 颗星，剔除 {seeing.n_rejected} 颗）")
        if seeing.elongation_median is not None:
            lines.append(f"拖长比中位数：{float(seeing.elongation_median):.2f}")
        if seeing.fwhm_arcsec is not None:
            lines.append(f"折算视宁度：{float(seeing.fwhm_arcsec):.2f} ″"
                         "（乘真值定出的板比例，条件命题）")
        lines.append("其中像素宽度与拖长比是纯像素量；角秒值是真值定标量")
    if sky_mag_arcsec2 is not None:
        lines.append(f"天光面亮度：约 {float(sky_mag_arcsec2):.0f} mag/arcsec^2")
        lines.append("给定真值零点下的条件命题；整数 ADU 量化地板 0.18 mag、"
                     "零点散度 0.22 mag，故不报小数位")
    if limiting is not None:
        lines.append(f"极限星等：{float(limiting):.1f}，取暗端经验百分位")
    if len(lines) == 1:
        lines.append("未提供视宁度与天光输入")
    ax_info.text(0.0, 0.98, "\n".join(lines), transform=ax_info.transAxes,
                 fontsize=10, va="top", ha="left")

    fig.tight_layout()
    return _finish(fig, path, probe=probe, extra={
        "n_valid": int(finite.size),
        "limiting": None if limiting is None else float(limiting),
        "sky_mag_arcsec2": None if sky_mag_arcsec2 is None else float(sky_mag_arcsec2),
    })
