# 太空目标识别系统

处理官方天基图像与历史地基 FITS 序列，输出星点候选、星等估计和运动轨迹。

**第 19 届官方数据已适配**：将 15 张 FITS 放在 `data/rst19/`，执行：

```powershell
.\scripts\dr.ps1 python -m src.cli --input data/rst19 --reference-frame 7 --catalog docs/reference/rst19-gaia-dr3.csv --output output/rst19-run
```

`7` 表示第 8 张图，演示暂用此帧；比赛指定其他图时修改帧号。天基辅助数据自动校验，
运行不需要联网。实测 26 条运动候选、294 行轨迹位置；该帧 12,842 个跨帧恒星候选，
最暗候选约 14.53 等（Gaia G 参照估计）。判据、限制、官网参考资料用途及复现说明见
[`官方天基数据实测`](docs/reports/rst19-2026-09-23.md)。运行环境仍使用现有约 1.01 GB 的 CPU 镜像。

历史地基流程的方法是**双坐标系目标判决**：把每一帧配准到同一天球参考系后，恒星在天球系
静止、在探测器系随机架扫动；人造目标反过来——在探测器系近乎不动、在天球系高速
掠过。数据集 B 实测两个速度差 **68.4 倍**（探测器系 1.84 px/帧、天球系
126.10 px/帧），判决不依赖任何真值文件。

历史指标逐条记在 [`docs/reports/measurements.md`](docs/reports/measurements.md)，
可用 `examples/` 下的脚本复现；新增比赛入口的实测见
[`交付验证记录`](docs/reports/delivery-2026-09-23.md)。

**源码提交（压缩后 ≤200 MB）**：运行 `python scripts/package_source.py`，生成
`dist/star-chart-source.zip`。打包范围、数据放置和运行环境见
[`docs/delivery.md`](docs/delivery.md)。原始 FITS、演示图片和 Docker 镜像单独管理。
日常目录用途和历史文件恢复方法见 [`目录说明`](docs/workspace.md)。

---

## 快速开始

### 0. 前置条件

需要 Docker，以及 `data/rst19/` 官方数据；运行下文历史教程时另需 `data/images/`
下的两个地基数据集目录。**宿主机不需要装 Python 依赖**
（astropy / photutils / scipy 只在镜像里）。

首次运行先构建 CPU 镜像（Python 3.11、固定版本依赖、Noto 中文字体）：

```bash
docker build -f docker/Dockerfile.cpu -t star-chart:cpu .
```

`scripts/dr.sh` 和 Windows 的 `scripts/dr.ps1` 优先使用 `star-chart:cpu`；
尚未构建时可自动使用已有的 `star-chart:cpu-interim`（21 GB）。设置 `SC_IMAGE`
可明确指定其他已安装镜像。脚本会先检查 Docker 和镜像，避免演示时隐式下载。

Windows PowerShell 示例（宿主机不需要科学计算依赖）：

```powershell
.\scripts\dr.ps1 python examples/01_load.py
```

新镜像的验证记录见 [`docs/delivery.md`](docs/delivery.md)。

### 1. 冒烟测试：读数据（约 3 秒）

```bash
bash scripts/dr.sh python examples/01_load.py
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

### 比赛入口：指定图像、最暗恒星候选与结果导出

```powershell
.\scripts\dr.ps1 python -m src.cli --input data/images/60385_20260722_1485695076008039_PIC_POS --reference-frame 30 --calibrate-from-truth
```

Linux / Git Bash 将 `.\scripts\dr.ps1` 换成 `bash scripts/dr.sh`。
默认在 `output/run-日期-时间/` 输出 `report.json`、`stars.csv`、`targets.csv` 和
`faintest-star.png`；目标表包含全部
判定成功的目标及逐帧坐标。已有非空输出目录会被保护，再次运行请指定新目录。

`--reference-frame` 是按观测时间排序的 0-based 帧号。默认核验该帧附近 6 帧，至少
4 帧一对一匹配后才计入恒星候选；最暗者从这些候选的固定孔径流量中实际选取，并标注位置。
只有显式提供 `--calibrate-from-truth` 才在识别完成后读取 `.DAT` 拟合零点；否则
定标星等为 `null`。这个新指标与下文保留的历史“95 百分位星等”不同，详见
[`比赛核查记录`](docs/competition-review.md) 和 [`本次实测`](docs/reports/delivery-2026-09-23.md)。

### 2. 主链：探测 → 配准 → 目标判决（约 105 秒）

```bash
bash scripts/dr.sh python examples/02_detect_register_target.py
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
bash scripts/dr.sh python -m pytest tests -q --junitxml=output/suite.xml -p no:cacheprovider
```

历史基线 `probe/fr-full2.xml`（已收入本地 `archive/2026-09-23-project-cleanup.zip`）的 `<testsuite>` 标签为：

```xml
<testsuite name="pytest" errors="0" failures="0" skipped="6" tests="569" ...>
```

历史基线是 **569 条测试总计，563 通过 / 0 失败 / 0 错误 / 6 跳过**
（5 skip + 1 xfail，逐条原因见「已知限制」）。JUnit 的 `tests` 包含跳过项，
不能将 569 写成通过数。新增交付检查和新镜像的结果见 `docs/delivery.md`。
使用 `python -m pytest` 从项目根目录运行，确保 `src` 可被导入。

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
6. **真值仅用于验证与星等零点，识别链本身不读真值。**
   `tests/test_architecture.py` 检查推理入口、探测/配准/跟踪与恒星确认模块的传递导入，
   防止通过间接依赖引入 `src.validate.truth`。它是源码导入约束，不是安全沙箱。
7. **数据集 A 无真值。** 新入口在 A 上检出 1 条候选轨迹（31 帧），仅能主张双坐标系
   判据通过；B 的真值验证结论不能移用于 A。详见本次实测记录。

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
  cli.py                 可指定数据、帧号的比赛入口，CSV/JSON 与最暗恒星标注
  pipeline.py            不依赖真值的识别流水线
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
    star_catalog.py      一对一跨帧恒星确认与最暗候选选取
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

真值隔离（传递导入测试）：`src/detect/`、`src/register/`、`src/target/`、
`src/validate/repeatability.py` 均不 import `src.validate.truth`。`photometry.py`
的零点标定是**显式允许**的真值用途。

---

## 已知限制

### 运行环境与中文字体

历史实测使用 `star-chart:cpu-interim`，其中的 SimHei 是手工复制的字体。
当前 CPU Dockerfile 改用轻量 Python 基础镜像，并通过 `fonts-noto-cjk` 安装字体。
重建或替换镜像后，运行 `tests/viz/test_figures.py` 检查字体与图表；构建状态及
实测大小统一记在 [`docs/delivery.md`](docs/delivery.md)。

字体守卫是**字形级**的，不是名字级：`FIGURE_CJK` 的 307 个字符里 SimHei 缺 0 个，
DejaVu Sans 缺 304 个。只查字体名会成功选中一个把每个汉字画成空心方框的字体。

### 6 个跳过的测试

| 数量 | 原因 |
|---|---|
| 4 | TLE 交叉校验：`docs/reference/tle.txt` 尚未提供相应观测历元的真实两行根数 |
| 1 | GPU 后端：本机无 NVIDIA 驱动 |
| 1 (xfail) | 数据集 B 5σ 复现率实测 **0.6304**，低于任务书门限 0.85 |

最后一条是**如实记录的实测结果，不是放宽阈值**。结构性上限（场滚出画幅）是 0.8261,
门限 0.85 在这个数据集上不可达。

### 当前范围之外

当前算法范围未包含：含赤经赤纬的 `.DAT` 输出、滑窗叠加、
像素系↔赤道系定向标定、盲板求解、光变曲线、SExtractor 交叉匹配、跨数据集回归。

统一入口为 `python -m src.cli`，`examples/` 下的七个脚本保留为历史指标的复现教程。
本次新增入口仍面向现有地基 FITS 格式，不声称已验证规则中的 15 张天基图像。

---

## 常见问题

**`Unable to find image 'star-chart:cpu'`**
先运行上面的 CPU 构建命令。启动脚本会在默认镜像缺失时使用已有 interim 镜像；
显式设置了 `SC_IMAGE` 时只使用所指定的镜像。

**`WARNING: File may have been truncated`**
数据本身的 FITS 长度字段与实际字节数差 448 字节，astropy 的正常提示，已在
`pytest.ini` 里过滤。不影响像素数据。

**`WARNING: The NVIDIA Driver was not detected`**
旧 interim / GPU 镜像带 CUDA，无显卡时可能提示此信息。新的 CPU 镜像不含 CUDA，
背景建模的 `backend: auto` 会使用 CPU。

**`ModuleNotFoundError: No module named 'src'`**
用了裸 `pytest` 或没经过 `scripts/dr.sh`。`PYTHONPATH=/workspace` 由该脚本注入。

**跑出来的数与本 README 差在末几位**
先查板比例尺约定（上文）。若差在第 3 位以后且不是那两个值之一，请报出来。
