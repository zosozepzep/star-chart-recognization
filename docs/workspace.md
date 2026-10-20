# 项目目录说明

## 2026-10-20 当前目录与清理

- 当前源码、完整离线 ZIP、校验值、回执及可直接运行的 `offline-kit/` 位于 `dist/cpu-parallel-20261020/`。
- 上一版 `national-final-20261018` 的源码 ZIP、完整 ZIP 及原始回执移至 `archive/releases/final-20261018/`；删除前逐文件确认旧解压副本已由完整 ZIP 覆盖。
- 更早的 `dist/history/` 和 9 月源码交付回执归入 `archive/releases/pre-cpu-parallel-20261020/`，旧报告中的路径可按原相对路径在该目录查找。旧源码 ZIP 已被新版替代，仅保留原始回执。
- 浏览器配置目录、工具依赖缓存、CPU 对照的源码解压副本与 Python 字节码属于可重新生成的临时文件，可清理；`output/cpu-parallel-20261020/` 的实测输出、测试 XML、比较脚本及截图保留。
- 原始 `data/rst19/`、当前源码与测试、科学实验结果、答辩材料、历史设计文档和历史压缩归档保留。

完整路径、删除前 SHA-256、移动映射和清理后校验记录在本地 `archive/cleanup-20261020/`。
以下保留历次整理记录，表中的历史位置不表示当前仍有对应目录。

## 历次目录记录

当前开发与运行使用下列目录：

| 目录 | 内容 |
|---|---|
| `src/` | 当前识别、测光、导出与绘图代码 |
| `tests/` | 当前代码的自动化检查 |
| `examples/` | 原有七步复现教程 |
| `scripts/` | Windows / Bash 启动与源码打包工具 |
| `docker/` | 容器构建文件 |
| `data/` | 原始数据与新下载的数据包，清理时保留 |
| `output/figures/` | 当前八张演示图 |
| `output/contest-A/`、`output/contest-B/` | 已验证的新入口结果 |
| `output/rst19-final/` | 官方 15 帧天基数据的正式分析结果 |
| `output/rst19-reproduced/` | 源码解压、禁网容器的独立复验结果 |
| `output/validation/` | 测试结果 XML |
| `dist/` | 当前源码提交包、SHA-256 和验收记录 |
| `dist/history/` | 旧提交包及各自的原始验收记录 |
| `docs/` | 交付说明、比赛核查、实测与设计文档 |
| `docs/history/` | 旧规划、开发日志和历史设计方案（含 `superpowers/`） |
| `chores/` | 比赛原始说明及参考截图 |
| `archive/` | 本机历史文件压缩归档及恢复清单，不进入源码提交包 |

官方数据位于 `data/rst19/`，运行时自动识别其辅助数据格式。当前最终发行位于
`dist/final-20261018/`，包含源码包、完整离线包、解压后的 `offline-kit/` 和 `release-receipt.json`。
`dist/star-chart-source.zip` 及其 `.sha256`、`submission-receipt.json` 是 2026-09-23 的历史交付。
此前地基初版、目录整理版、官方天基版及回执移入
`dist/history/2026-09-23-pre-cleanup/`。回执中记录的是当时的路径，查找旧文件时
在该历史目录按原文件名定位；内容不改写。

依赖入口统一为根目录 `requirements.txt`。原额外清单只包含未使用的 `astroquery`，
已移入 `archive/retired-dependencies/`；本次路径整理清单见 `archive/workspace-tidy-2026-09-23.json`。
官方分析、独立禁网复验、历史演示结果和原始数据均保留在原位置。

2026-09-23 目录清理将旧版 `legacy/`、`probe/`、`.superpowers/`、根目录诊断图、
旧实验输出和修改前的演示图归入 `archive/2026-09-23-project-cleanup.zip`。
归档保留相对于项目根目录的原路径，因此历史注释中的 `probe/...` 路径可在 ZIP 内查找。
具体操作与校验清单见 `archive/cleanup-manifest.json`。

Python / pytest 缓存、提交包解压验证副本和网页核查时下载的通用前端脚本已清理。
验证副本中的测试结果另存到 `output/validation/package-validation.xml`。
此前生成的 `star-chart-source.zip`、`star-chart-source-v2.zip` 及验收记录现存于上述
历史交付目录。原验收记录中的 `figure_backup` 路径
`output/figures-before-2026-09-23/` 现在位于上述历史 ZIP 内，记录本身保持原样。

共归档 1,268 个文件；连同生成缓存和重复验证目录的清理，扣除归档 ZIP 后释放约
282 MiB。清理后核对了 82 个源码、脚本及原提交包的 SHA-256，以及 124 个原始数据
文件的大小和修改时间，均保持一致。核对结果见 `archive/cleanup-verification.json`。

恢复历史材料时，先把 ZIP 解压到一个新的空目录，再按需取回文件，避免覆盖当前源码。
本次不改写 Git 历史，历史跟踪的旧输出移出工作目录后会在 Git 中显示为删除。

2026-10-18 最终版验收时，工作区中的 `data/images/` 历史地基数据已不存在，
相关 41 项测试明确跳过；官方 `data/rst19/` 完整，并纳入离线演示包。
不把历史目录清理记录中的“当时保留”理解为当前文件仍然存在。
