# 太空目标识别系统

从地基望远镜的 FITS 序列里找出人造天体，并给出可复现的定量结论。

核心方法是**双坐标系目标判决**：把每一帧配准到同一天球参考系后，恒星在天球系
静止、在探测器系随机架扫动；人造目标反过来——在探测器系近乎不动、在天球系高速
掠过。数据集 B 实测两个速度差 **68.4 倍**（探测器系 1.84 px/帧、天球系
126.10 px/帧），判决不依赖任何真值文件。

所有数字都在容器里实测过，逐条记在 [`docs/reports/measurements.md`](docs/reports/measurements.md)。
本 README 里出现的每个数，你都可以用 `examples/` 下的脚本自己跑出来。

---

## 快速开始

### 0. 前置条件

需要 Docker，以及 `data/images/` 下的两个数据集目录。**宿主机不需要装 Python 依赖**
（astropy / photutils / scipy 只在镜像里）。

当前可用镜像只有一个：

```bash
docker images | grep star-chart
# star-chart:cpu-interim   21GB
```

`scripts/dr.sh` 的默认镜像名是 `star-chart:cpu`，**它不存在**。所以下面每条命令
都显式带 `SC_IMAGE=star-chart:cpu-interim`。

> **不要执行 `docker build` 或 `docker-compose up --build`。** 详见「已知限制」。

### 1. 冒烟测试：读数据（约 3 秒）

```bash
SC_IMAGE=star-chart:cpu-interim bash scripts/dr.sh python examples/01_load.py
```

应看到：

```
=== 历元 B ===
  dataset_id     = 60385_20260722_1485695076008039_PIC_POS
  帧数           = 80
  首帧 DATE-OBS  = 2026-07-21T17:26:28.122
  真值 .DAT      = 20260721172644_6002_060385_0008.DAT
  跟踪段         = f16-f70，共 55 帧
```

### 2. 主链：探测 → 配准 → 目标判决（约 105 秒）

```bash
SC_IMAGE=star-chart:cpu-interim bash scripts/dr.sh python examples/02_detect_register_target.py
```

关键输出：

```
判为目标的航迹数 = 1
  v_det_px_per_frame     = 1.843980065992662      探测器系速度
  v_sky_px_per_frame     = 126.09585119222203     天球系速度
  stationarity_contrast  = 68.38243727127353      两者之比
配准 54 对，内点 74..93（中位 84），rms 中位 0.7363 最大 1.8567 px
```

`rms_max` 1.8567 出在段首对 f16→f17：机架尚未停稳。这是需要跳过段首帧的实测依据。

### 3. 其余五步

| 脚本 | 内容 | 耗时 | 需要真值 |
|---|---|---|---|
| `examples/01_load.py` | 读序列、头部、跟踪段 | 3 s | 否 |
| `examples/02_detect_register_target.py` | 探测 + 配准 + 双系判决 | 105 s | 否 |
| `examples/03_truth_photometry.py` | 真值比对、星等零点、极限星等 | 108 s | **是** |
| `examples/04_trajectory_orbit.py` | 派生 1/2：角速度、高度反演 | 105 s | 是（板比例） |
| `examples/05_sensor_seeing.py` | 派生 4/5：传感器健康、观测条件 | 60 s | 仅天光面亮度 |
| `examples/06_repeatability.py` | 阈值-复现率曲线（两个数据集） | 100 s | 否 |
| `examples/07_figures.py` | 出 8 张成果图到 `output/figures/` | 155 s | 是 |

顺序无关，每个脚本自成一体（代价是主链会重跑）。**建议顺序 01 → 02 → 07**：
07 一次跑完能看到全部结论 + 8 张图。

### 4. 测试套件（约 14 分钟）

```bash
SC_IMAGE=star-chart:cpu-interim bash scripts/dr.sh \
  python -m pytest tests -q --junitxml=suite.xml -p no:cacheprovider
```

只认 `suite.xml` 里 `<testsuite>` 标签的属性，不要读终端汇总行：

```xml
<testsuite name="pytest" errors="0" failures="0" skipped="6" tests="569" ...>
```

当前基线 **569 通过 / 0 失败 / 0 错误 / 6 跳过**（5 skip + 1 xfail，逐条原因见
「已知限制」）。必须用 `python -m pytest` 而不是裸 `pytest`，否则 `PYTHONPATH`
不生效。

---

## 实测结论一览

全部来自数据集 B（80 帧、含 `.DAT` 真值）。数据集 A（36 帧、无真值）只作
「同站不同历元」的对照。

### 星图识别

| 量 | 值 | 口径 |
|---|---|---|
| 第 30 帧探测源数 | 169 / 116 / **95** @ 3σ / 4σ / 5σ | 报数阈值是 5σ |
| 5σ 跨帧复现率 | **0.6304**（B）/ 0.8000（A） | 6 帧，≥4 帧命中，3 px 半径 |
| 结构性上限 | 0.8261（B）/ 0.8842（A） | 场滚出画幅，无论探测多好都不可能复现 |
| 极限星等（95 百分位） | **10.2111** | **经验暗端百分位**，不是「最暗可复现源」 |
| 星像宽度 `fwhm_px` | **3.403649**（B）/ 3.409136（A） | 纯像素量，可独立主张 |
| 拖长比中位 | 1.195896 / 1.202717 | 是上一行**可读性的前提**，须一起报 |

**为什么只报 10² 颗而不是 10⁵ 颗**：2σ → 5σ 之间探测数掉 25 倍（B：2325.83 →
91.83），而复现率从 0.070 升到 0.630。2σ 上每帧那 2300 个「源」里只有 161 个能跨帧
复现——**其余 93% 是噪声峰**。跑 `examples/06_repeatability.py` 可以复现整条曲线。

### 星图分析（双坐标系判决）

| 量 | 值 |
|---|---|
| 静止性对比度 | **68.38243727127353** |
| 探测器系速度 / 天球系速度 | 1.843980065992662 / 126.09585119222203 px/帧 |
| 54 帧探测器系总漂移 | 95.80648318010832 px |
| 起点 → 终点（探测器系） | (2271.5947, 1958.8500) → (2365.8030, 1976.2765) |

真值比对（`.DAT`，54 帧全部配对，`dt_max_s = 0.0`）：

| 量 | 值 |
|---|---|
| 板比例 | **6.179046984546331 ± 0.012843265127332462** ″/px |
| 星等零点 | **15.335125319985362 ± 0.21790718677520077** |
| 星等相关系数 | 0.9486520886198294 |

### 四个派生数据集

| 派生 | 量 | 值 |
|---|---|---|
| 1 角速度 | 平均角速度 | **770.150746** ″/s（散度 5.908433，相对 0.77%） |
| 2 高度反演 | 自相容区间 | **[213.4410634932108, 848.3591755656954] km**，宽 634.9181 |
| 2 | 轨道速度 | 7.5955657040170586 km/s |
| 4 传感器健康 | 跨历元共有缺陷簇 | **4 / 4**，持久性 1.0，Jaccard 1.0 |
| 4 | 最大配对偏移 | 0.33529240969646845 px（两簇逐位相同） |
| 5 观测条件 | 天光面亮度 | **17.34**（B）/ 17.54（A）mag/arcsec² |

两个历元相隔 134.158023 天，4 个缺陷簇全部复现、两个位置逐位相同——这是「缺陷来自
硅而不是当晚环境」的证据。

---

## 口径约束（引用这些数字时不可违）

这几条不是措辞偏好，是实测边界。越过它们，结论就站不住。

1. **高度只报自相容区间 `[213.44, 848.36] km`，不报「与公开根数吻合」。**
   真实两行根数还没抄进 `docs/reference/tle.txt`，四条 TLE 交叉校验测试是 SKIPPED。
   （848.70 是派发前的预估值，勿用。）
2. **`fwhm_arcsec` 与天光面亮度是「真值定标量」，只能作条件命题。** 两者都乘了由
   真值定出的板比例（天光面亮度还用了真值零点），不得声称独立观测，更不得说它们
   「验证了真值」——真值是它们的输入，用输出验证输入是循环论证。只有 `fwhm_px`
   与 `elongation_median` 可作独立主张。
3. **天光面亮度不写小数位。** 背景中位是整数 ADU，1 ADU 在 6.0 上就折 0.18 mag；
   零点自身另有 0.22 mag 散度。两项都比小数位大一个量级。
4. **极限星等是经验暗端百分位**，不是探测完备性极限，也不是「最暗可复现源」。
   它只对单帧源表取百分位，全程没有跨帧比对或注入-回收。
5. **轨迹图的位置轴是恒星系像素，不是赤经赤纬。** FITS 头没有 WCS、没有焦距、
   没有像元尺寸，`ra_deg` / `dec_deg` / `pa_deg` 在当前范围里恒为 `None`；位置角同理不报。
6. **只能说「真值仅用于验证与星等零点，识别链本身不读真值」**，不能说「架构级真值
   隔离」——那个守卫（Task 13）尚未落地。前一句读代码就能验：`src/detect/`、
   `src/register/`、`src/target/`、`src/validate/repeatability.py` 都不 import
   `src.validate.truth`。
7. **数据集 A 有没有可探测目标，是个还没测的事实。** 全部目标结论都出自数据集 B。

---

## 板比例尺约定（会影响你自己跑出来的数）

真值比对给出的全精度板比例是 **6.179046984546331** ″/px，但项目全程用四舍五入的
**6.179**：

| 用哪个 | 平均角速度 |
|---|---|
| `SCALE = 6.179`（**交付口径**） | **770.150746** ″/s |
| 6.179046984546331（全精度） | 770.156602 ″/s |

两者都对，但必须选一个。`examples/` 里一律用 6.179，与 `docs/reports/measurements.md`
和测试里的 `SCALE` 常量一致。如果你自己写脚本时传了全精度值，得到的所有下游数
（角速度、高度区间、`fwhm_arcsec`）都会与报告差在末几位——那不是 bug。

同理，`examples/05_sensor_seeing.py` 的帧预算从
`src/config/default.yaml` 的 `sensor_health.report_frames` 读（出厂 **20**），
硬写 30 会得到 4 个簇但偏移变成 0.0/0.157/0.335/0.5——20 是簇数收敛平台的起点
（5 帧 → 115 簇、10 → 8、15 → 6、20 → 4、25/30/36 → 4），不是随手选的数字。

---

## 代码结构

```
src/
  pointset.py            点集形状校验（as_xy）：全项目点集统一 (N,2) float64 [x, y]
  dataio/fits_loader.py  FITS 序列读取（头部三处非标准写法要 verify('silentfix')）
  calib/
    segment.py           从 AZ/EL 时序切出稳定跟踪段
    background.py        photutils Background2D 背景与底噪
    hotpixel.py          热像素/缺陷簇普查（时间轴统计 + 数值锁定判据）
  detect/
    segmentation.py      阈值分割源探测
    centroid.py          质心与孔径流量
    psf.py               二维高斯 PSF 拟合
  register/
    transform.py         相似变换模型
    solver.py            旋转-平移投票 + 最小二乘精化，链式累积到参考帧
  target/
    dual_frame.py        **双坐标系目标判决**（本项目主亮点）
    trajectory.py        目标轨迹与逐步角速度（派生 1）
  analysis/
    orbit.py             圆轨道高度反演（派生 2）
    sensor_health.py     传感器健康报告 + 视宁度 + 天光面亮度（派生 4/5）
  astrometry/
    photometry.py        仪器星等、真值零点标定、极限星等
  validate/
    repeatability.py     跨帧可复现性（**禁止 import truth**）
    truth.py             .DAT 真值读取与轨迹比对
  viz/figures.py         8 张成果图 + 中文字体守卫
  config/default.yaml    全部阈值，模块体内不留裸数字
```

坐标系约定：**0-based，x = 列，y = 行，画面中心 (2047.5, 2047.5)**。

真值隔离（代码层面，非架构守卫）：`src/detect/`、`src/register/`、`src/target/`、
`src/validate/repeatability.py` 均不 import `src.validate.truth`。`photometry.py`
的零点标定是**显式允许**的真值用途。

---

## 已知限制

### 镜像从未构建过

`docker/Dockerfile.cpu` **没有被构建过一次**。当前镜像 `star-chart:cpu-interim`
里的 SimHei 中文字体是手工 `COPY` 进去的，**不属于任何 apt 包**——Dockerfile 里的
`fonts-noto-cjk` 在这个镜像里实际不存在。

后果：**首次真实 `docker build` 之后，必须重跑 `tests/viz/test_figures.py` 的四条
字体测试**，否则 `setup_matplotlib()` 会抛 `FontUnavailable`，8 张图一张也画不出来。
容器内无网络，所以这一步没法在当前环境里关闭。

字体守卫是**字形级**的，不是名字级：`FIGURE_CJK` 的 307 个字符里 SimHei 缺 0 个，
DejaVu Sans 缺 304 个。只查字体名会成功选中一个把每个汉字画成空心方框的字体。

### 6 个跳过的测试

| 数量 | 原因 |
|---|---|
| 4 | TLE 交叉校验：容器无网络，`docs/reference/tle.txt` 尚未抄入真实两行根数 |
| 1 | GPU 后端：本机无 NVIDIA 驱动 |
| 1 (xfail) | 数据集 B 5σ 复现率实测 **0.6304**，低于任务书门限 0.85 |

最后一条是**如实记录的实测结果，不是放宽阈值**。结构性上限（场滚出画幅）是 0.8261,
门限 0.85 在这个数据集上不可达。

### 当前范围之外

按 [`scope-mvp.md`](.superpowers/sdd/2026-09-08-太空目标识别系统/scope-mvp.md)（未入库）
的裁剪，以下推到后续阶段：统一命令行入口、CSV/JSON/.DAT 输出、滑窗叠加、
像素系↔赤道系定向标定、盲板求解、光变曲线、SExtractor 交叉匹配、跨数据集回归。

所以现在**没有 CLI**——`examples/` 下的七个脚本就是入口。

---

## 常见问题

**`Unable to find image 'star-chart:cpu'`**
`scripts/dr.sh` 的默认镜像名不存在。加 `SC_IMAGE=star-chart:cpu-interim`。

**`WARNING: File may have been truncated`**
数据本身的 FITS 长度字段与实际字节数差 448 字节，astropy 的正常提示，已在
`pytest.ini` 里过滤。不影响像素数据。

**`WARNING: The NVIDIA Driver was not detected`**
基础镜像带 CUDA，本机无 GPU。背景建模的 `backend: auto` 会自动落到 CPU。

**`ModuleNotFoundError: No module named 'src'`**
用了裸 `pytest` 或没经过 `scripts/dr.sh`。`PYTHONPATH=/workspace` 由该脚本注入。

**跑出来的数与本 README 差在末几位**
先查板比例尺约定（上文）。若差在第 3 位以后且不是那两个值之一，请报出来。
