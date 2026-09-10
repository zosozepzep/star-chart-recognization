# 实测量记录

本文件累积流水线各步在真实数据上的实测数字，是中文技术文档与答辩预案的唯一取数
来源——文档里的每个数字都必须能在这里找到出处，不允许凭记忆写数。

数据集：
- **A** = `60394_20260309_1437117414637934_PIC`，QIANFAN-16 / NORAD 60394，36 帧，无 `.DAT` 真值
- **B** = `60385_20260722_1485695076008039_PIC_POS`，QIANFAN-7 / NORAD 60385，80 帧，含 `.DAT` 真值

## 机架状态分段

参数取 `src/config/default.yaml` 的 `segment` 默认值：`window=5`、
`std_threshold_deg=0.05`、`min_length=10`。

复现命令：

```bash
bash scripts/dr.sh python -c "
from src.calib.segment import segment_sequence
from src.dataio.fits_loader import FrameSequence
for d in ['data/images/60394_20260309_1437117414637934_PIC',
          'data/images/60385_20260722_1485695076008039_PIC_POS']:
    seq = FrameSequence.from_directory(d)
    print(seq.dataset_id, len(seq), 'cadence %.5f s' % seq.cadence_s())
    for s in segment_sequence(seq.headers):
        print('   %-8s f%02d-f%02d (%2d) daz %+.4f +- %.4f' %
              (s.label, s.start, s.end, s.length, s.daz_mean, s.daz_std))
"
```

实测输出：

```
60394_20260309_1437117414637934_PIC 36 cadence 1.00900 s
   slew     f00-f03 ( 4) daz +0.0002 +- 0.0002
   tracking f04-f35 (32) daz +0.4749 +- 0.0160
60385_20260722_1485695076008039_PIC_POS 80 cadence 1.00900 s
   slew     f00-f15 (16) daz +0.0784 +- 5.3699
   tracking f16-f70 (55) daz +0.2214 +- 0.0144
   slew     f71-f79 ( 9) daz -5.9676 +- 0.0533
```

最长跟踪段：

| 数据集 | 帧数 | 帧间隔 (s) | 最长跟踪段 | 段长 | ΔAZ 均值 (°/帧) | ΔAZ 标准差 |
|--------|------|-----------|-----------|------|----------------|-----------|
| A | 36 | 1.00900 | f04–f35 | 32 | +0.4749 | 0.0160 |
| B | 80 | 1.00900 | f16–f70 | 55 | +0.2214 | 0.0144 |

### 与 `.DAT` 真值的独立交叉校验（数据集 B）

真值文件 `20260721172644_6002_060385_0008.DAT` 恰好 55 行，与分段结果的 55 帧一致；
两端时标逐帧对齐，边界 `(16, 70)` 因此有独立佐证：

| `.DAT` 行 | 真值时标 | 对应帧 | 帧 DATE-OBS |
|-----------|----------|--------|-------------|
| 第 1 行 | 17:26:44.2670 | f16 | 2026-07-21T17:26:44.267 |
| 第 55 行 | 17:27:38.7400 | f70 | 2026-07-21T17:27:38.740 |

### 需要注意的两个实测细节

1. **数据集 B 跟踪段的 `daz_std` 是 0.0144，不是 <0.01。** 段内第一个增量
   `r16 (f16→f70 方向的 f16→f17)` 为 **+0.11899**，明显小于其余增量的
   ~0.2225——这是机架停稳前的余量。剔除这一个增量后，其余 53 个增量的标准差仅
   **0.0032**。f16 依真值
   时标确属跟踪段，故保留该帧、按实测修正标准差预期，而非挪动分段边界。

2. **数据集 A 确有一个可用的长跟踪段：f04–f35，共 32 帧。** 开头 f00–f03 的
   |ΔAZ| ≤ 0.0005°/帧（机架静止待命），被 `min_length=10` 自然排除，无需区分
   "静止"与"慢速跟踪"。A 的跟踪速率 +0.4749°/帧约为 B 的 2.1 倍，段内标准差
   0.0160 与 B 同量级，仍远低于 0.05 阈值。A 的跟踪段一直延续到序列末帧（f35），
   末尾没有回摆。

## 热像素（传感器缺陷）识别

参数取 `src/config/default.yaml` 的 `hotpixel` 默认值：`n_sigma=5.0`、
`flat_ratio=0.10`、`dilate=1`、`max_frames=40`、`reject_radius_px=8.0`。
帧区间取各数据集最长跟踪段的前 30/31 帧：B 用 f16–f45，A 用 f05–f35。

复现命令：

```bash
SC_IMAGE=star-chart:cpu-interim bash scripts/dr.sh python -c "
import resource
from src.calib.hotpixel import build_from_sequence
from src.dataio.fits_loader import FrameSequence
for d, rng in [('data/images/60385_20260722_1485695076008039_PIC_POS', range(16, 46)),
               ('data/images/60394_20260309_1437117414637934_PIC', range(5, 36))]:
    hpm = build_from_sequence(FrameSequence.from_directory(d), list(rng))
    print(d.split('/')[-1], len(hpm.clusters), 'clusters', int(hpm.mask.sum()), 'mask px')
    for c in hpm.clusters:
        print('   npix %2d  (%8.3f, %8.3f)  med %9.1f  ratio %.4f'
              % (c.npix, c.cx, c.cy, c.median_value, c.flat_ratio))
    print('   peak RSS %.3f GiB'
          % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024))
"
```

数据集 B（f16–f45）实测 **5 簇、seed 21 像素、`dilate=1` 后 mask 57 像素**，
按输出次序（npix 降序，同 npix 时 cx 升序）：

| npix | cx | cy | median_value | flat_ratio |
|------|-----|-----|--------------|-----------|
| 9 | 327.000 | 3210.111 | 162.0 | 0.0603 |
| 8 | 244.000 | 3173.500 | 148.2 | 0.0444 |
| 2 | 312.000 | 3206.500 | 134.5 | 0.0743 |
| 1 | 0.000 | 0.000 | 26981.5 | 0.0009 |
| 1 | 2192.000 | 3223.000 | 2612.0 | 0.0065 |

数据集 A（f05–f35，相隔约 4.5 个月的独立观测）给出同一批位置，同样是
**5 簇、seed 21 像素、mask 57 像素**，逐簇对照：

| npix | cx | cy | median_value | flat_ratio | 与 B 的位置差 |
|------|-----|-----|--------------|-----------|--------------|
| 9 | 327.000 | 3210.111 | 165.0 | 0.0424 | 0 |
| 8 | 244.250 | 3173.500 | 149.5 | 0.0468 | 0.25 px（cx） |
| 2 | 312.000 | 3206.500 | 137.5 | 0.0619 | 0 |
| 1 | 0.000 | 0.000 | 26983.0 | 0.0009 | 0 |
| 1 | 2192.000 | 3223.000 | 2609.0 | 0.0046 | 0 |

位置最大互差 0.25 px、中位值互差不超过 3 个计数——这是"传感器固有缺陷"而非"天体"
的确证：两次观测相隔约 4.5 个月，任何天体都不可能落在同一批像素上。
(0, 0) 那个卡死角点中位值约 26982，正是 Task 4 `background_stats` 在 frame 30
报出 `max=26978` 的来源，任何"全图最亮像素"式逻辑都必须先过热像素剔除。

`flat_ratio` 阈值扫描（数据集 B，f16–f45，本轮重新实测）：

| flat_ratio | 簇数 | seed 像素 | 说明 |
|-----------|------|----------|------|
| 0.05 | 4 | 10 | 漏掉 (312, 3206.5)，它实测 0.0743 |
| 0.07 | 5 | 17 | 平台下沿 |
| 0.10 | 5 | 21 | **默认值**，平台中部 |
| 0.15 | 5 | 25 | 平台上沿 |
| 0.20 | 18 | 42 | 跳变 |
| 0.30 | 242 | 270 | 开始收进真实恒星 |

取 0.10 是因为它位于 0.07–0.15 这段平台的中部，两侧各有约 0.03/0.05 裕度。
平台内簇数恒为 5、位置不变，变的只是簇边缘的 seed 像素数（17→25）。
被漏掉的 (312, 3206.5) 中位值 134.5（背景 6），且在 4 个月前的数据集 A 同一像素
复现，是无可争议的真缺陷——这就是不取原设想的 0.05 的理由。

### 两个边界像素

判据 `ratio < flat_ratio` 是严格小于，边界上确有像素：

| 像素 (x, y) | median | span | ratio |
|-------------|--------|------|-------|
| (245, 3174) | 100.0 | 10.0 | **0.100000000**（恰等于阈值，被排除） |
| (326, 3211) | 64.5 | 7.0 | 0.108527 |

`<` 改成 `<=` 的后果可观测：(244.0, 3173.5) 簇由 8 像素变 9、质心移到
(244.111, 3173.556)，seed 21→22、mask 57→58，且它会顶掉 (327, 3210.1) 成为首簇。
由 `test_dataset_b_flat_ratio_boundary_is_strict` 看守。

### 内存实测

| 量 | 值 |
|----|-----|
| 30 帧 4096² uint16 立方体 | 0.938 GiB（1.007 GB） |
| 40 帧（`max_frames` 上限）立方体 | 1.250 GiB |
| `med` + `span` 两个 4096² float64 | 0.250 GiB |
| `_pixel_stats` 分块上限 | 256 MiB |
| 30 帧时每块行数 / 块数 | **1092 行 / 4 块**（末块 820 行；4096 = 3×1092 + 820） |
| `build_from_sequence(B, f16–f45)` 峰值 RSS | **1.760 GiB**（1.890 GB），基线 0.087 GiB |
| `build_from_sequence(A, f05–f35)` 峰值 RSS | 1.807 GiB（同进程内接着 B 跑，31 帧） |

峰值 RSS 由 `resource.getrusage(RUSAGE_SELF).ru_maxrss` 在容器内量得，B 单独运行
两次分别为 1.759 / 1.760 GiB。
注：`.superpowers` 下的 Task 5 报告曾记"每块约 21 行"，那是推算数、错约 50 倍，
已按 `256*1024*1024 // (30*4096*2) = 1092` 重算并实测复核；同一处早先记的 1.845 GB
峰值与本次 1.890 GB 属同一量级（`ru_maxrss` 受分配器与页回收影响，有百分之几的
运行间差异），结论"峰值远低于不分块时的水平"不变。

## 配准

参数取 `src/config/default.yaml` 的 `register` 默认值：`center=[2047.5, 2047.5]`、
`rot_range_deg=1.5`、`rot_step_deg=0.05`（`np.arange` 给出 61 步扫描）、
`shift_max_px=400.0`、`vote_bin_px=4.0`、`match_radius_px=5.0`、`min_inliers=30`。
输入点集取数据集 B 最长跟踪段 f16–f70（55 帧、54 对），热像素掩膜由段前 30 帧
（f16–f45）建立，探测用 `n_sigma=4.0`、`npixels=5`。

复现命令：

```bash
SC_IMAGE=star-chart:cpu-interim bash scripts/dr.sh python -c "
import json
from src.calib.hotpixel import build_from_sequence
from src.calib.segment import tracking_segment
from src.dataio.fits_loader import FrameSequence
from src.detect.segmentation import detect_sequence
from src.register.solver import register_sequence
seq = FrameSequence.from_directory('data/images/60385_20260722_1485695076008039_PIC_POS')
seg = tracking_segment(seq.headers)
frames = list(range(seg.start, seg.end + 1))
hpm = build_from_sequence(seq, frames[:30])
dets = detect_sequence(seq, frames, n_sigma=4.0, npixels=5, hot_clusters=hpm.clusters)
rep = register_sequence(dets, frames, reference=seg.start).report()
print(json.dumps({k: v for k, v in rep.items() if k != 'pairs'},
                 ensure_ascii=False, indent=2, default=float))
"
```

实测 `report()`（去掉逐对的 `pairs` 明细）：

```json
{
  "reference_frame": 16,
  "n_frames": 55,
  "n_pairs": 54,
  "inliers_min": 74,
  "inliers_max": 93,
  "inliers_median": 84.0,
  "rms_max_px": 1.8566921552753293,
  "rms_median_px": 0.7363272298653143,
  "cumulative": {
    "scale": 0.9982173045106925,
    "rotation_deg": -3.324711228702002,
    "tx": 6494.953129137156,
    "ty": -539.8055332030586
  }
}
```

汇总：

| 量 | 实测 |
|----|------|
| 段 / 帧数 / 对数 | f16–f70 / 55 / 54 |
| 探测源数（每帧） | 100–124，中位 114 |
| 内点 min / median / max | 74 / 84.0 / 93 |
| `rms_max_px` | **1.8567**（全部来自段首一对 f16→f17） |
| `rms_median_px` | **0.7363** |
| 其余 53 对 rms 区间 | **0.6229–0.8322**，中位 0.7358 |
| 逐对旋转 min / median / max | −0.1046° / **−0.0608°** / −0.0547° |
| 累计（f16→f70）scale | 0.9982173 |
| 累计旋转 | −3.32471° |
| 累计平移 tx / ty | +6494.953 / −539.806 px |
| `register_sequence` 耗时 | **0.399 s / 54 对 = 0.0074 s/对** |
| `detect_sequence` 耗时（同一次运行） | 90.8 s |

### `rms_max_px = 1.8567` 的归因：段首一对跨机架整定过渡

`rms_max_px` 的 1.8567 **完全由第一对 f16→f17 贡献**，其余 53 对落在
0.6229–0.8322。原因是几何而非质心质量：f16 跨在机架整定的尾部，
`azimuth_rates` 实测 f15→f16 = +0.549830、**f16→f17 = +0.118990**、
f17→f18 = +0.213910，之后稳定在 ~0.2214。段内 54 个增量的均值是 0.221442037，
所以 f16→f17 只有段内均值的 **53.7%**。这一对的场移也随之异常：
`shift_px[0] = 61.38 px`，其余 53 对全在 118.91–123.61 px。跟踪段 f16–f70 内
`exposure_ms` 恒为 30.0，所以这不是曝光时长造成的假象（f00–f07 是 80.0，但那几帧
在摆扫段内，不参与配准）。

这一对的残差是整体抬升而不是几个离群点：p50 = 1.6615，剔掉最差 5%（74→70 点）
重新拟合也只降到 1.6580。真正的判据是自由度对比——**同一组 74 对匹配点，4 参数
相似变换 rms 1.8567，6 参数仿射 rms 0.6618，比值 0.357**；而在健康对上 6 参数
没有收益（f17→f18 比值 0.9992、f32→f33 0.9865）。残差还呈强线性空间结构：
`corr(x, res_y) = -0.9090`、`corr(y, res_x) = -0.9449`，而同轴项只有
`corr(x, res_x) = -0.1186`、`corr(y, res_y) = +0.1039`，两轴残差均值都是 0——
交叉项强、同轴项弱、均值为零，这是切变的指纹。也就是说帧间映射在这一对上带
真实切变，4 参数模型按定义吸收不了它。

这组 1.8567 / 0.6618 对比，正是 `src/register/transform.py` 模块 docstring 里
"4 参数吸收不了切变"那段论证的真实数据佐证；它比 Task 9 用镜像点集造出的
`rms = 20√5 ≈ 44.7214` 更有说服力，因为它出自生产数据而非合成反射。

**这不是配准缺陷。** 缺陷的源头在上游分段：`tracking_segment` 用的是整段
`daz_std` 判据，它把 f16 这个首个整定帧一并收进了跟踪段（同一事实在本文件
「机架状态分段」一节已经记过：剔除 f16→f17 这一个增量后其余 53 个增量的标准差
只有 0.0032，而含它是 0.0144）。MVP 内不改 Task 2 的分段逻辑，因此段首这一对
的残差抬升被如实记录并由分层断言看守，而不是靠放宽 `rms_max_px` 的上界掩掉。

守护测试 `test_dataset_b_registration_matches_measured_baseline` 因此不用单一的
`rms_max_px < 1.5`，而是分层断言：段首对 `< 2.0`、其余 53 对全部 `< 1.0`、
汇总 median `< 1.0`、且 `rms_max_px` 必须恰等于段首对的 rms（钉住"最大值来自
段首"这一归因本身）。段首对**不设下界**——将来质心精度改进使它降到 0.9 时测试
不该因此变红，强度由 `max(rest) < 1.0` 与 median 承担。

