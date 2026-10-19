# 曝光拖尾重定位实验

本实验接续 2026-10-15 的轨迹复核：阈值分割可能只保留曝光拖尾的一段，
其质心不等于完整曝光中心。实验在原始像素上拟合曝光模型，随后重新关联，
对照三组：`baseline80`（原 80 ADU）、`threshold40`（仅降低到 40 ADU）、
`refined40`（40 ADU 加重定位）。生产入口不调用此实验分支。

## 实现范围

`src/target/trail_refinement.py` 仅接收图像、观测元数据、检测结果与配置。
用初步轨迹的探测器坐标和实际 UTC 拟合速度，将速度和曝光时间固定为线段，
求解线段与圆高斯卷积模型的中心、流量、PSF 宽度和局部背景平面。
高斯宽度从 1.5 px 初始化，在 0.7–3 px 内拟合，不使用注入时的宽度。
所有拟合门限位于 `src/config/trail_refinement.yaml`。

拟合使用解析误差函数模型，注入使用独立的离散曝光积分实现；两者仍属于同一
物理模型族。对真实非高斯 PSF、光变目标及弯曲拖尾的泛化尚未得到验证。

只处理已经形成初步轨迹且期望曝光长度至少 6 px 的观测。
负值、饱和、非有限像素及其邻域被屏蔽；跨帧静止源中心周围 4 px 被屏蔽。
优化不收敛、参数触边、有效像素或信号支持不足、低信噪比、残差过大均拒绝更新，
保留原始检测坐标。每个观测保存接受/拒绝原因和拟合诊断。

更新后使用原配准、关联与曝光拖尾核验。没有填补缺帧，没有在真值位置强制测光，
没有增加未检测到的点。原始峰值、面积和测光保持原口径；拟合流量只作诊断。
未成轨迹的弱目标不会因这一方法被恢复。仍可出现碎片或重复轨迹，评估会保留它们。

## 两阶段复现

先准备上一轮真实背景注入报告；若无该报告，先按
[`真实背景实验`](injection-benchmark.md) 运行 `src.validate.injection`。
旧报告列出的源码和输入 SHA-256 必须与当前文件一致，否则工具拒绝混用版本。

开发阶段只重用旧 `tuning` 的三个种子和裁块：

```powershell
.\scripts\dr.ps1 python -m src.validate.trail_experiment --input data/rst19 --baseline output/injection-20261015/injection.json --output output/new-trail-development
```

两条对照分支的轨迹数量、观测帧、坐标及回收/未归属计数会与旧报告逐例核对。
开发报告完整生成后，方可运行新区域验证：

```powershell
.\scripts\dr.ps1 python -m src.validate.trail_experiment --input data/rst19 --baseline output/injection-20261015/injection.json --phase validation --development-report output/new-trail-development/experiment.json --output output/new-trail-validation
```

验证启动时核对完整开发报告、源码/配置校验值及原始报告校验值。
源码变动需重跑开发阶段。两阶段运行中也核查源码不变，完成前再次核对输入。
新种子固定为 5101、6101、7103；裁块仅通过几何排斥采样选择，在开发计算前落盘，
与所有旧调参/留出裁块及新裁块彼此均不重叠。没有按像素质量或结果挑选区域。
它们依然来自同一官方序列，不是独立天区、独立设备或独立时段验证。

每阶段为 27 个注入场景 × 3 方法，共 81 条记录，另有 3 个未注入对照 × 3 方法。
沿用原 600／1800／5400 ADU 与 2／6／12 px/s 网格，不改注入规则和匹配半径。
再次改进算法后，已查看过的新区域也不能继续当作未见数据。

## 结果与口径

每个输出目录包含：

- `protocol.json`：计算前固定的配置、输入和源码校验值、全部裁块。
- `controls.json`：三方法未注入对照，含拟合失败和拖尾否决证据。
- `cases.jsonl`、`experiment.json`：全部预测、真值、逐观测拟合诊断及评估。
- `recovery.csv`、`recovery.png`、`report.md`：分箱曲线与方法对照。
- `receipt.json`：输出校验值及完成回执；异常保存 `failure.json`。

两阶段结束后，可在宿主机用标准库独立复核输出完整性、原匹配门限、计数和定位误差：

```powershell
python scripts/check_trail_experiment.py --development output/new-trail-development/experiment.json --validation output/new-trail-validation/experiment.json --input data/rst19 --output output/validation/new-trail-receipt.json
```

检查器核对全部源码、输入及输出的 SHA-256，并确认新验证使用了冻结的开发版本。
结果记录见 [`10 月 18 日实测`](reports/trail-refinement-2026-10-18.md)。

一对一回收继续要求至少 4 个时刻且至少 80% 预测观测落在真值 3 px 内。
未注入对照已匹配真值的情况不授予回收。没有完整真实目标标注，新增未归属数
不能称为 FP 或据此计算 Precision。重复可匹配数仅统计同一注入可领取匹配的
额外轨迹，且这些轨迹仍留在原未归属计数内。

`localization_rmse_px` 保留旧评估器的匹配半径内误差。
汇总的 `all_observation_rmse_px` 另计成功匹配轨迹的**全部**观测，包括超出 3 px 的点，
避免仅展示被半径裁掉后的误差。不同方法成功匹配的集合可能不同，须同时报告漏检。
`mean_matched_coverage` 仍为原评估器的半径内可见帧覆盖。

共享流水线时间包含生产基线关联；方法时间不含检测和配准。
`refined40` 的方法时间含初始 40 ADU 关联/核验、拟合及重新关联/核验。
各例三个相同的共享时间不可重复相加；总时间包含重复计算、验证和图文导出。
这些裁块耗时不能当作全幅现场运行时间。
