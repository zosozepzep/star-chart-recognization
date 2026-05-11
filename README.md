
# 🌌 太空目标识别仿真 (Space Object Identification Simulation)

本仓库用于参加**第十九届先进机器人及仿真技术大赛 - 太空目标识别仿真组**。本项目基于[](Spacemapper.cn)提供的天基 FITS 图像数据，旨在通过计算机视觉与深度学习技术，实现高精度的星图识别与动目标检测。

---

## 🎯 比赛目标
根据大赛规则，本项目核心任务包括：
* 1. **星图识别**：从 FITS 图像中识别星点、计算总数，并定位星等最低（最暗）的星。
* 2. **星图分析**：通过多帧图像序列识别太空中的运动目标（空间碎片、非合作卫星等）。
* 3. **创新赛项**：从原始数据中提取光变曲线、运动矢量等具有实际工程意义的创新数据 。

## 🛠️ 技术栈
* **OS**: Windows 11 + WSL2 (Ubuntu 22.04)
* **Runtime**: Docker Desktop (Containerization)
* **Language**: Python 3.10
* **Core Libraries**: 
    * `OpenCV`: 图像处理与矩阵运算
    * `Astropy`: FITS 科学数据解析与天文计算
    * `PyTorch`: 深度学习推理 (规划中)
    * `Matplotlib`:科学计算和绘图库
* **Methodology**: Vibe Coding (高效直觉驱动开发)

## 🏗️ 环境配置
本项目完全容器化，确保了开发环境的一致性。

### 1. 克隆仓库
```bash
git clone https://github.com/zosozepzep/star-chart-recognization.git
cd star-chart-recognization
````

### 2. 启动开发环境

确保已安装 Docker Desktop 并开启 WSL2 后端：

Bash

```
docker-compose up -d --build
```

### 3. 进入容器

Bash

```
docker exec -it opencv_gpu_env bash
```

## 📁 项目结构

Plaintext

```
.
├── data/ # 由于文件太大，已忽略
│   └── images/          # 存放原始.fits 数据 [cite: 26]
├── src/
│   ├── star_detect_fits.py  # 星图识别 Baseline (动态阈值法)
│   └── utils/           # 工具函数 (FITS转换, 坐标计算)
├── output/              # 算法输出结果 (可视化图片 & JSON 结构化数据)
├── Dockerfile           # 镜像构建文件 (集成 CUDA, OpenCV, Astropy)
└── docker-compose.yml   # 容器编排配置
```

## 🚀 当前进展 (Milestones)

- [x] **基础环境搭建**：完成基于 Docker + WSL2 的 GPU 加速环境配置。
    
- [x] **星图识别 Baseline**：实现基于 `Astropy` 数据解析与动态统计阈值（$\mu + N\sigma$）的星点提取算法，支持自动计算背景底噪并输出 JSON 数据。
    
- [ ] **动目标检测**：开发基于多帧差分或目标追踪（DeepSORT）的运动目标识别逻辑。
    
- [ ] **创新数据提取**：计划实现太空碎片的光变曲线分析。

## 📄 许可声明

本项目代码仅供参赛使用。数据来源归Spacemapper.cn，先进机器人及仿真技术大赛组委会所有。