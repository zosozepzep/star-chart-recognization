# 🌌 太空目标识别仿真 (SpaceMapper Pipeline v2.0)

本仓库用于参加**第十九届先进机器人及仿真技术大赛 - 太空目标识别仿真组**。本项目基于 [Spacemapper.cn](http://spacemapper.cn/) 提供的天基 FITS 序列数据，构建了一套从底层物理特征提取到高层时序轨迹关联的全自动处理流水线。

---

# 🚀 核心架构与今日更新 (2026-05-12)

我们已将原有的 Baseline 升级为 **“自适应双引擎 + 时序物理审判”** 架构，解决了海量暗星背景下的虚警问题。

## 1. 算法层面：物理驱动的提纯

- **双引擎探测 (Double-Engine)**：融合了 Engine B (PSF 亚像素拟合) 与 Engine A (形态学分割)，确保高精度定位与非标目标的全面捕获。
    
- **自适应下界雷达**：引入“二阶导数增长率”监控，通过动态阈值自动下探至 $2.5\sigma \sim 3.0\sigma$，成功从背景中挖掘出 65,000+ 颗真实暗星。
    
- **线性度审查 (Linearity Check)**：利用 **NumPy 向量化运算** 对 120+ 候选轨迹进行运动学审查，将误报的布朗运动噪点彻底排除，精准锁定 12 个真实运动目标。
    

## 2. 工程层面：容器化并行流水线

- **Docker 一键式环境**：基于容器化部署，利用 Docker 卷挂载实现物理数据与输出产物的无缝解耦。
    
- **多进程加速 (Concurrency)**：在 `Phase 2` 引入进程池加速，充分榨干 CPU 多核算力，单帧处理效率提升 5-10 倍。
    
- **结构化输出**：自动分类存储图像预览 (`.jpg`) 与科学数据 (`.json`)，支持断点续传。
    

---

# 🏗️ 环境配置 (WSL2 + Docker)

本项目完全运行在容器化环境中，确保了从开发到比赛提交的环境一致性。

## 1. 启动容器

Bash

```
# 构建并启动 OpenCV+Astropy 专用环境
docker-compose up -d --build
```

## 2. 进入开发环境

Bash

```
docker exec -it opencv_gpu_env bash
cd /workspace
```

---

# 📁 优化后的项目结构

Plaintext

```
.
├── data/
│   └── images/              # 原始 FITS 序列 (按 20260309... 命名)
├── src/
│   ├── main_pipeline.py     # 【核心】全自动化主控脚本
│   ├── detector.py          # 自适应双引擎星点提取器
│   ├── tracker.py           # 多帧轨迹关联与线性度审查引擎
│   └── inspect_star.py      # 3D 能量分布物理审查工具
├── output/
│   ├── single_frames/       # 过程产物 (按来源引擎着色预览)
│   │   ├── images/          # Green: Engine B, Cyan: Engine A
│   │   └── json/            # 每帧的结构化物理特征
│   └── final_catalog/       # 最终时序清洗后的总星表 (决赛提交格式)
├── Dockerfile               # 集成 CUDA, OpenCV, Astropy, Photutils
└── docker-compose.yml       # 容器编排配置
```

---

# 🚀 运行流水线

在容器内部执行以下操作，即可自动完成从数据排序到目标认定的全过程：

Bash

```
# 运行自动化流水线
python src/main_pipeline.py
```

**流水线逻辑流：**

1. **Phase 1 (Time Sort)**：自动解析文件名中的毫秒级时间戳，建立绝对时间序列。
    
2. **Phase 2 (Detection)**：多核并发提取，利用双引擎捕获全量星点并进行边缘裁切。
    
3. **Phase 3 (Tracking)**：基于 KD-Tree 进行跨帧关联，剔除瞬态噪点。
    
4. **Phase 4 (Physics Check)**：执行线性度向量化校验，生成最终的 `master_catalog.json`。

---

# 📊 当前进展 (Milestones)

- [x] **基础环境搭建**：Docker + WSL2 GPU 开发环境。
    
- [x] **高性能提取**：实现自适应双引擎算法，支持 60,000+ 级别的暗星捕获。
    
- [x] **时序关联引擎**：完成基于线性度终审的碎片锁定算法，虚警率显著下降。
    
- [ ] **光变曲线分析**：(进行中) 计划提取目标在15帧序列中的亮度波动特征。
    
- [ ] **速度向量计算**：(规划中) 基于物理时间差 $\Delta t$ 导出目标的绝对像素速度。

---

# 📄 许可声明

本项目代码仅供第十九届先进机器人及仿真技术大赛参赛使用。

**UCAS Aerospace Project - Backend Group**

_Last Update: 2026-05-12_