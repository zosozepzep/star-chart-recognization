# 源码提交与运行环境

## 提交包

源码压缩后上限为 **200 MB**。在项目根目录执行（仅使用 Python 标准库）：

```bash
python scripts/package_source.py
```

生成 `dist/star-chart-source.zip`，采用更严格的十进制 200,000,000 字节上限。
超限会以非零状态退出，不产生可误提交的压缩包。已有同名包不会被覆盖；后续可用
`--output dist/star-chart-source-v2.zip` 指定新文件。

包内包含 `src/`、`tests/`、`examples/`、启动与打包脚本、Dockerfile、依赖清单、
配置、运行说明及实测报告。`SOURCE_MANIFEST.json` 记录每个文件的大小与 SHA-256；
终端另外输出整个 ZIP 的 SHA-256。同一份文件内容重复打包会得到相同的 ZIP。

原始数据、生成图片/目录、Docker 镜像、Git 历史、探针脚本、已归档的旧版代码、
内部规划和比赛附件不在包内。打包过程不会删除或移动原目录。

**不要直接压缩整个项目，也不要用 `git archive` 代替提交脚本。** 当前 Git 中仍有
历史输出被跟踪，`.gitignore` 不会把它们自动移出索引；打包脚本使用文件类型与目录
白名单独立控制交付范围。若后续需要清理 Git 历史，应另做备份和仓库迁移。

## 数据与环境

官方比赛数据现在使用 `data/rst19/` 下的 15 张 FITS。推荐命令：

```powershell
.\scripts\dr.ps1 python -m src.cli --input data/rst19 --reference-frame 7 --catalog docs/reference/rst19-gaia-dr3.csv --output output/rst19-run
```

参考 Gaia 星表已随源码提供，运行阶段不需要网络。输出包含两份 CSV、JSON 报告、
最暗候选标注和运动轨迹图。当前交付使用 `dist/star-chart-source.zip`，
验收信息见 `dist/submission-receipt.json`。此前三个版本及其原始回执统一存放于
`dist/history/2026-09-23-pre-cleanup/`，旧文件内容和校验值保持不变。
每次选择空输出目录；重新打包时另选不存在的文件名。

以下两个目录用于历史地基教程，不是本届官方天基数据：

解压后保留项目目录结构，把已有比赛数据分别放回：

```text
data/images/60394_20260309_1437117414637934_PIC/
data/images/60385_20260722_1485695076008039_PIC_POS/
```

前者 36 帧 FITS，后者 80 帧 FITS，并包含对应的 `.DAT` 真值。这些数据约 3.66 GiB，
由数据集渠道单独提供。源码 ZIP 本身不等同于含输入数据的离线演示包。

CPU 镜像构建：

```bash
docker build -f docker/Dockerfile.cpu -t star-chart:cpu .
```

本机 `docker image ls` 实测：旧 `star-chart:cpu-interim` **21 GB**，新
`star-chart:cpu` **1.01 GB**，约减少 **95%**。两者都不装入源码 ZIP。

镜像只装主流程所需的 Python 科学计算栈和 Noto CJK 字体。源码与数据通过启动脚本
挂载；`.dockerignore` 将构建上下文限定为依赖文件和 Dockerfile，避免传入数 GiB 数据。
首次构建需要能访问镜像仓库、Debian 软件源和 PyPI。

项目只有一个依赖入口 `requirements.txt`，用于本地安装和 Docker 构建，包含当前
代码与测试使用的八项固定版本依赖。原 `requirements-extra.txt` 仅声明旧在线星表
方案的 `astroquery`，当前代码没有引用；现有 Gaia 定标直接读取随包 CSV，因此已
将这份过期清单归档。本次整理没有新增安装包，现有 CPU 镜像可以继续使用。

Windows PowerShell：

```powershell
.\scripts\dr.ps1 python examples/01_load.py
.\scripts\dr.ps1 python examples/02_detect_register_target.py
.\scripts\dr.ps1 python examples/07_figures.py
.\scripts\dr.ps1 python -m pytest tests -q --junitxml=output/suite.xml -p no:cacheprovider
```

Linux / macOS / Git Bash：

```bash
bash scripts/dr.sh python examples/01_load.py
bash scripts/dr.sh python examples/02_detect_register_target.py
bash scripts/dr.sh python examples/07_figures.py
bash scripts/dr.sh python -m pytest tests -q --junitxml=output/suite.xml -p no:cacheprovider
```

启动脚本优先使用 `star-chart:cpu`，未构建时可自动使用现有的 `star-chart:cpu-interim`。
可用 `SC_IMAGE` 环境变量选择指定镜像；指定镜像不存在则直接报错，不偷偷更换环境。
Compose 的 GPU 服务须显式启用 `--profile gpu`，普通 CPU 运行不会构建 GPU 镜像。
GPU 配方仍为历史可选环境，不是本次提交的推荐运行方式。

若无需 Docker，也可使用 Python 3.11 虚拟环境安装 `requirements.txt`，在项目根目录
以 `python -m examples.01_load` 等模块方式运行，并自行安装 `src/config/default.yaml`
候选表中覆盖所需字形的中文字体。宿主机较新的 Python（例如 3.13）不保证兼容此处
固定版本的依赖，推荐使用经过验证的容器。

## 比赛材料口径

新增比赛入口可运行：

```powershell
.\scripts\dr.ps1 python -m src.cli --input data/images/60385_20260722_1485695076008039_PIC_POS --reference-frame 30 --calibrate-from-truth --output output/contest-B
```

输出文件：`report.json`（参数、判决、校准来源）、`stars.csv`（参考帧候选及命中次数）、
`targets.csv`（全部目标的逐帧 UTC 时标与两坐标系像素位置）、`faintest-star.png`。
`--frames START END` 可指定连续分析段（含末帧）；`--config` 接受完整配置文件；
`--no-plot` 只导出数据；`--verbose` 展开模块日志。不要重复使用已有非空输出目录。

固定孔径半径和核验窗口来自配置。最暗恒星从满足跨帧核验的候选里按指定帧流量
选取，星等不能解释成探测完备性极限。没有 `.DAT`、没有唯一可关联目标，或未指定
`--calibrate-from-truth` 时，`mag` 为 `null`，保留仪器星等。真值定标使用目标与恒星
共同的固定孔径，并归一化到 ADU/s；与历史教程的分割流量零点不能混用。

本地比赛规则记载：省赛交方案 PPT（≤100 MB）和运行 MP4（≤100 MB、≤3 分钟）；
国赛另交运行程序和源代码。本项目用户另明确源码 ZIP 上限为 200 MB。

官方 15 张天基图像已经独立适配并验证，见 `docs/reports/rst19-2026-09-23.md`。
下方 A/B 记录仍指历史地基数据。官网同页的省赛提交说明存在两种表述，其中下载区
列出“PPT、运行视频、源代码”，因此保留源码交付准备，并以最终提交入口要求为准。

## 2026-09-23 验证记录

- 源码打包：标准库测试 4/4 通过，覆盖大文件排除、清单校验、超限失败与既有文件保留。
- Bash / PowerShell 启动脚本：语法检查通过。
- 新 CPU 镜像：构建成功，`pip check` 通过；全套 573 条中 567 通过、5 skipped、1 xfailed，
  无失败或错误，耗时 892.852 秒。记录在本地 `output/validation/cpu-validation.xml`。
- 新增功能单独验证：14/14 通过（包含 4 条打包测试），记录在 `output/validation/delivery-tests.xml`。
- 空场景导出与已有结果保护：2/2 通过，记录在 `output/validation/cli-tests.xml`。
- 数据集 B 新入口实跑通过：1 条目标、54 行目标位置；指定第 30 帧 93 个去坏点探测源，
  71 个跨帧恒星候选，最暗候选 m≈9.14（真值定标），位置 (326.81, 3867.78)。
  详见 `docs/reports/delivery-2026-09-23.md`，本地结果在 `output/contest-B/`。
- 数据集 A 无真值运行通过：1 条候选轨迹、31 行位置，定标星等为 `null`；
  不使用数据集 B 的零点为 A 补值。本地结果在 `output/contest-A/`。
- 历史基线：`probe/fr-full2.xml` 共 569 条，563 通过，5 skipped、1 xfailed，无失败。
  此文件已按原路径收入 `archive/2026-09-23-project-cleanup.zip`，不随源码分发。
  目录清理与恢复方法见 `docs/workspace.md`。
