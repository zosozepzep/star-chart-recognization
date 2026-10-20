# CPU 并行识别

当前发行 `national-cpu-parallel-20261020` 支持逐帧多进程和多线程执行。默认使用多进程，自动选择当前环境可用的 CPU 数量，最多 4 个工作单元，且不超过分析帧数。源码与完整离线包位于 `dist/cpu-parallel-20261020/`；上一版压缩包保留在 `archive/releases/final-20261018/`。

## 使用

网页执行 `scripts/demo.ps1` 启动后，在识别表单中选择「CPU 并行数量」和「CPU 执行方式」。默认「自动」与「多进程」。运行日志和导出的 `report.json` 中的 `execution` 字段会记录实际执行方式、工作单元数量及底层线程限制。

命令行示例：

```powershell
# 4 个进程利用多个 CPU 核心（推荐）
.\scripts\dr.ps1 python -m src.cli --input data/rst19 --reference-frame 7 --workers 4 --output output/parallel-run

# 4 个线程，共享进程内存
.\scripts\dr.ps1 python -m src.cli --input data/rst19 --reference-frame 7 --workers 4 --executor thread --output output/thread-run

# 串行，用于低内存设备或结果对照
.\scripts\dr.ps1 python -m src.cli --input data/rst19 --reference-frame 7 --workers 1 --output output/serial-run
```

`--workers` 和 `--executor` 覆盖 YAML 的 `execution.workers`、`execution.executor`。`0` 表示自动，`1` 表示串行，正整数表示请求的并行数；负数、非整数及未知执行方式会报错。旧的完整 YAML 配置若没有 `execution`，继续串行运行。

## 并行范围与资源

- 地基与天基主流程的读图、背景估计、源检测、固定孔径测光按帧并行；地基热像素图先统一生成。每帧的计算参数与原流程相同。
- 此次接入的是统一 CLI 与网页主流程；`examples/` 中直接调用旧 `detect_sequence` 的历史示例仍按原方式串行执行。
- 结果按输入帧序汇总，再执行原有配准、轨迹关联、核验和导出。后面这些阶段仍串行，因此加速不会与核心数成正比。
- 多进程绕过 Python GIL；多线程中 NumPy/SciPy 的部分运算可以并行，但 Python 循环仍受 GIL 限制。应以目标设备的实测耗时选择模式。
- 正式 FITS 序列只向工作单元传文件路径；合成或内存序列只传该帧像素，不序列化整个序列，也不传验证真值。最多保留与工作单元数相同数量的待处理任务及暂存结果，避免预先复制全序列。
- 并行区域把各工作单元的 BLAS/OpenMP 线程数限制为 1，避免线程层层叠加。线程模式退出时恢复原设置，串行阶段保留原设置。
- **自动数量的上限不是内存检测。** 4096×4096 图像的单张 float64 数组约 128 MiB，背景、掩膜、分割还需多个临时数组；4 个工作单元可能占用数 GiB。内存紧张时设为 1 或 2，不建议直接选择全部逻辑核心。Docker 内还受 Docker Desktop 的 CPU/内存配额约束。
- CPU 并行应配合 `background.backend: cpu` 或无可用 GPU 的 CPU 镜像。GPU 背景计算不在此次并行性能验证范围内。
- 网页的「停止当前任务」会停止整个 Docker 容器或本地计算进程树。工作单元报错会取消待执行任务，等待当前任务退出，并向调用方传播错误。

直接从 Python 脚本调用主流程并使用多进程时，调用必须放在 `if __name__ == '__main__':` 中，以支持 Windows 和 `spawn` 启动方式。在交互环境里可选线程模式或串行模式。

## 环境升级

新增固定依赖 `threadpoolctl==3.6.0`。已有旧 CPU 镜像需要更新；完整构建命令为：

```powershell
docker build -f docker/Dockerfile.cpu -t star-chart:cpu .
```

直接在 Python 环境运行时，安装更新后的 `requirements.txt`。网页服务自身仍只依赖 Python 标准库，科学计算依赖装在工作进程环境中。

验证记录见 [CPU 并行实测](reports/cpu-parallel-2026-10-20.md)。
