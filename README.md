# 🌌 天文星点检测流水线 (Star Chart Recognization)

本项目用于解决太空目标识别仿真任务，旨在在强噪声、海量恒星背景和复杂伪目标条件下，稳定识别真实天体，并输出可信的物理参数和亚像素级定位精度。当前系统已由规则为主的探测方案完成向**物理约束驱动、可量化评估、可扩展维护的高性能工业级流水线**重构转型。

---

## 🎯 核心能力与特性

1. **高召回率挖掘**：通过动态背景估计提取候选目标，尽最大可能挖掘暗弱星点（不惧强噪底线）。
2. **GPU异构加速**：引入 CuPy 方案进行 GPU 高速预处理，快速求取 Background Map 与 RMS Map。
3. **亚像素级精度 (Sub-pixel Accuracy)**：依靠光度学特征提取与 2D Gaussian PSF 精细模型拟合，保障检测结果高精度。
4. **置信度评分与时空滤噪**：
    - 基于统一评分引擎 (scorer.py) 剔除伪目标。
    - **空间 NMS (非极大值抑制)** 确保不过分割目标。
    - **时序关联 (Temporal Association)**与物理轨迹追踪共同完成最终的裁决。
5. **对齐工程标准**：保留了与天文学标准工具 SExtractor 可视化及量化比对通道。

---

## 🏗️ 目录结构规划

核心检测系统现已解耦并重组为完整的树状功能模块：

`	ext
star-chart-recognization/
├── data/                       # 原始图像与 FITS 序列
├── output/                     # 处理流程中途产出与终版 Catalog (含SExtractor测试对比)
├── src/
│   ├── main_pipeline.py        # 【核心】端到端全自动化流水线 
│   ├── combat_mission.py       # 竞赛或核心批处理运行任务
│   ├── detector/               # 单帧检测引擎模块组
│   │   ├── gpu_preprocessor.py # 基于 CuPy 的 GPU 混合背景推断 
│   │   ├── candidate_generator.py # 最初的基于阈值或 DAO 的粗提取
│   │   ├── feature_extractor.py   # 背景与基本测光特征抓取
│   │   ├── psf_fitter.py          # 二维全局及局部 PSF 高斯拟合
│   │   ├── scorer.py              # 高维联合置信度判定
│   │   └── nms.py                 # 密区非极大值抑制去重
│   ├── tracking/               # 时序追踪裁决模块组
│   │   ├── associator.py       # 跨帧关联分析
│   │   ├── kalman.py           # 卡尔曼滤波位置预期验证
│   │   └── physical_validator.py# 线速度、物理特性等整体审查
│   ├── benchmark/              # 量化基点评估模块
│   └── visualization/          # 对比效果与过程渲染
├── Dockerfile                  # 含CUDA与Python天文核心依赖的构建镜像
├── docker-compose.yml          # 一键开发环境容器编排
├── cross_match_diagnostics.py  # 用于交叉验证对齐效果的工具脚本
└── 规划.md                     # 工程演进与重构路线图
`

---

## 🚀 部署与运行

本项目底层高度依赖 CUDA GPU 算力执行张量推断。为避免本机环境的 Numba, CuPy与 CUDA runtime 版本冲突，**强烈建议使用 Docker 进行一键调试与运行。**

### 1. 启动并进入容器

`bash`
`docker-compose up -d --build`
`docker-compose exec [容器名] /bin/bash`  # 具体名称参考 compose 的 services 配置

环境镜像内已封装最新的**nvidia/cuda:13.x** 以及**OpenCV, CuPy, Astropy, Photutils**等必备框架。)

### 2. 一键启动

在容器环境下进入该项目根目录路径中即可运行：

`bash`
`python src/combat_mission.py`
`

### 3. 基准测试 (SExtractor Benchmark)

如需对自研模型与权威天文库的结果进行定量比对，以明确召回率与虚警率：

`bash` 
`python cross_match_diagnostics.py`

---

## 📜 更新与展望

项目正逐步剔除原有的纯启发阶段算法，向更加纯粹的物理信息系统衍进。
# Changelog
[[2026.05.13]]
